// xAI provider — a thin fromEnv over the shared OpenAICompatProvider.
// xAI's only bespoke surface is context-window resolution (xAI exposes no
// context_window via any documented API endpoint) and the per-token pricing
// probe against /v1/language-models; everything else (the generate spine,
// usage mapping, reasoning translation) is the framework's.

import {
    computeCost,
    OpenAICompatProvider,
    parseOptionalInt,
    parseRequiredInt,
    thinkingFromEnv,
    dataCaptureFromEnv,
    parseRequiredFloat,
    providerSource,
    requireEnv,
    type Provider,
} from "@plurnk/plurnk-providers";

const DEFAULT_BASE_URL = "https://api.x.ai/v1";

// Context windows from docs.x.ai/developers/models (July 2026). xAI does not
// expose context_window via any documented API endpoint — /v1/language-models
// returns rich pricing data but no window, /v1/models is OpenAI-sparse.
// Operators can override via PLURNK_PROVIDERS_CONTEXT_SIZE for new aliases not
// yet in the table. Longest prefix match wins.
const CONTEXT_BY_PREFIX: ReadonlyArray<[string, number]> = Object.freeze([
    ["grok-4.20-multi-agent", 2_000_000],
    ["grok-4.1-fast", 2_000_000],
    ["grok-4.20", 1_000_000],
    ["grok-4.3", 1_000_000],
    // Grok Build (grok-build-0.1, alias grok-code-fast-1) — the coding model, 256k.
    ["grok-build", 256_000],
    ["grok-code-fast", 256_000],
]);

const lookupContextByPrefix = (model: string): number | null => {
    let best: { prefix: string; ctx: number } | null = null;
    for (const [prefix, ctx] of CONTEXT_BY_PREFIX) {
        if (model.startsWith(prefix) && (best === null || prefix.length > best.prefix.length)) {
            best = { prefix, ctx };
        }
    }
    return best?.ctx ?? null;
};

// Reasoning capability by family (#35). Grok Build (grok-build-0.1 /
// grok-code-fast-1) is a coding model with NO reasoning channel — xAI 400s on
// `reasoning_effort` for it, whatever the value ("Model … does not support
// parameter reasoningEffort"). Every other Grok reasons. Model-capability, the
// same shape as the context table: send the reasoning param ONLY where the model
// accepts it. A non-reasoning model gets reasoningStyle "none" — no wire param at
// all — so a globally-set THINKING intent can't break the coding alias. This is
// the accurate mapping (the channel doesn't exist), not a silent degradation.
const NON_REASONING_PREFIXES: readonly string[] = ["grok-build", "grok-code-fast"];
const modelReasons = (model: string): boolean => !NON_REASONING_PREFIXES.some((p) => model.startsWith(p));

export default class Xai {
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Provider> {
        const apiKey = requireEnv(env.XAI_API_KEY, "XAI_API_KEY", "xai");
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_PROVIDERS_FETCH_TIMEOUT, "PLURNK_PROVIDERS_FETCH_TIMEOUT", "xai");
        const thinking = thinkingFromEnv(env, "xai");
        const rawBase = env.XAI_BASE_URL !== undefined && env.XAI_BASE_URL.length > 0
            ? env.XAI_BASE_URL
            : DEFAULT_BASE_URL;
        const base = rawBase.replace(/\/$/, "");

        // Context: env override > per-family table > throw. Resolved before the
        // pricing probe so an unknown alias fails fast without a network call.
        const envCtx = parseOptionalInt(env.PLURNK_PROVIDERS_CONTEXT_SIZE, "PLURNK_PROVIDERS_CONTEXT_SIZE", "xai");
        const contextSize = envCtx !== null ? envCtx : lookupContextByPrefix(model);
        if (contextSize === null || !Number.isFinite(contextSize) || contextSize <= 0) {
            throw new Error(
                `xai provider: no context-window known for "${model}". xAI's API does not expose this; ` +
                "either pick an alias matching a known family prefix (grok-4.3, grok-4.20*, etc.) " +
                "or set PLURNK_PROVIDERS_CONTEXT_SIZE explicitly.",
            );
        }

        const pricing = await fetchPricing({ base, apiKey, model, fetchTimeoutMs });

        return new OpenAICompatProvider({
            model,
            url: `${base}/chat/completions`,
            fetchTimeoutMs,
            headers: { Authorization: `Bearer ${apiKey}` },
            contextSize,
            thinking,
            temperature: parseRequiredFloat(env.PLURNK_PROVIDERS_TEMPERATURE, "PLURNK_PROVIDERS_TEMPERATURE", "xai", 0),
            repeatPenalty: parseRequiredFloat(env.PLURNK_PROVIDERS_REPEAT_PENALTY, "PLURNK_PROVIDERS_REPEAT_PENALTY", "xai", 0),
            retryDelayMs: parseRequiredInt(env.PLURNK_PROVIDERS_RETRY_DELAY, "PLURNK_PROVIDERS_RETRY_DELAY", "xai"),
            retryAttempts: parseRequiredInt(env.PLURNK_PROVIDERS_RETRY_ATTEMPTS, "PLURNK_PROVIDERS_RETRY_ATTEMPTS", "xai"),
            // Opt-in data capture (#36), off by default, per-alias-scopable.
            ...dataCaptureFromEnv(env, "xai"),
            reasoningStyle: modelReasons(model) ? "effort" : "none",
            // Per xAI's docs Grok uses cl100k_base. All current Grok variants
            // share the same tokenizer — no per-model dispatch needed.
            // Three-rate cost: cached tokens are a SUBSET of prompt_tokens,
            // billed at the discounted cached rate; the non-cached portion is
            // billed at the full prompt rate. computeCost bills billable output
            // (completion + reasoning) at the completion rate.
            costFor: (usage) => computeCost(usage, {
                input: pricing.prompt_pico_per_token,
                output: pricing.completion_pico_per_token,
                cached: pricing.cached_pico_per_token,
            }),
            source: providerSource("xai"),
        });
    }
}

// xAI exposes three distinct rates per model in pico-dollars per token.
// `cached` is xAI's prompt-cache discount (typically much lower than prompt);
// applied to the subset of prompt_tokens that came from cache.
type XaiPricing = {
    prompt_pico_per_token: number;
    cached_pico_per_token: number;
    completion_pico_per_token: number;
};

// /v1/language-models/{id} returns per-model pricing in pico-dollars/token.
// Falls back to /v1/language-models (list) if the per-id endpoint 404s
// (rare; new alias not yet exposed).
type ModelPricingResponse = {
    id?: string;
    prompt_text_token_price?: number;
    cached_prompt_text_token_price?: number;
    completion_text_token_price?: number;
};
type ListResponse = { models?: ModelPricingResponse[] };

const fetchPricing = async ({
    base, apiKey, model, fetchTimeoutMs,
}: { base: string; apiKey: string; model: string; fetchTimeoutMs: number }): Promise<XaiPricing> => {
    const direct = await fetch(`${base}/language-models/${encodeURIComponent(model)}`, {
        headers: { Authorization: `Bearer ${apiKey}` },
        signal: AbortSignal.timeout(fetchTimeoutMs),
    });
    if (direct.ok) {
        const entry = (await direct.json()) as ModelPricingResponse;
        return toPricing(entry, model);
    }
    if (direct.status !== 404) {
        const body = await direct.text();
        throw new Error(`xAI /language-models/${model} returned ${direct.status}: ${body}`);
    }
    // 404 on per-id endpoint — fall back to list, match by id or alias.
    const list = await fetch(`${base}/language-models`, {
        headers: { Authorization: `Bearer ${apiKey}` },
        signal: AbortSignal.timeout(fetchTimeoutMs),
    });
    if (!list.ok) {
        const body = await list.text();
        throw new Error(`xAI /language-models returned ${list.status}: ${body}`);
    }
    const data = (await list.json()) as ListResponse;
    const entry = data.models?.find((m) => m.id === model
        || (m as { aliases?: string[] }).aliases?.includes(model));
    if (entry === undefined) {
        throw new Error(`xAI /language-models has no entry for "${model}"`);
    }
    return toPricing(entry, model);
};

const toPricing = (entry: ModelPricingResponse, model: string): XaiPricing => {
    if (entry.prompt_text_token_price === undefined || entry.completion_text_token_price === undefined) {
        throw new Error(`xAI /language-models entry for "${model}" missing prompt/completion prices`);
    }
    return {
        prompt_pico_per_token: entry.prompt_text_token_price,
        cached_pico_per_token: entry.cached_prompt_text_token_price ?? entry.prompt_text_token_price,
        completion_pico_per_token: entry.completion_text_token_price,
    };
};
