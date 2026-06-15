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
    reasoningBudgetFromEnv,
    providerSource,
    requireEnv,
    tokenizerFor,
    type Provider,
} from "@plurnk/plurnk-providers";

const DEFAULT_BASE_URL = "https://api.x.ai/v1";

// Context windows from docs.x.ai/developers/models (May 2026). xAI does not
// expose context_window via any documented API endpoint — /v1/language-models
// returns rich pricing data but no window, /v1/models is OpenAI-sparse.
// Operators can override via PLURNK_PROVIDER_CONTEXT_SIZE for new aliases not
// yet in the table. Longest prefix match wins.
const CONTEXT_BY_PREFIX: ReadonlyArray<[string, number]> = Object.freeze([
    ["grok-4.20-multi-agent", 2_000_000],
    ["grok-4.1-fast", 2_000_000],
    ["grok-4.20", 1_000_000],
    ["grok-4.3", 1_000_000],
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

export default class Xai {
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Provider> {
        const apiKey = requireEnv(env.XAI_API_KEY, "XAI_API_KEY", "xai");
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_FETCH_TIMEOUT, "PLURNK_FETCH_TIMEOUT", "xai");
        const reasoningBudget = reasoningBudgetFromEnv(env, "xai");
        const rawBase = env.XAI_BASE_URL !== undefined && env.XAI_BASE_URL.length > 0
            ? env.XAI_BASE_URL
            : DEFAULT_BASE_URL;
        const base = rawBase.replace(/\/$/, "");

        // Context: env override > per-family table > throw. Resolved before the
        // pricing probe so an unknown alias fails fast without a network call.
        const envCtx = parseOptionalInt(env.PLURNK_PROVIDER_CONTEXT_SIZE, "PLURNK_PROVIDER_CONTEXT_SIZE", "xai");
        const contextSize = envCtx !== null ? envCtx : lookupContextByPrefix(model);
        if (contextSize === null || !Number.isFinite(contextSize) || contextSize <= 0) {
            throw new Error(
                `xai provider: no context-window known for "${model}". xAI's API does not expose this; ` +
                "either pick an alias matching a known family prefix (grok-4.3, grok-4.20*, etc.) " +
                "or set PLURNK_PROVIDER_CONTEXT_SIZE explicitly.",
            );
        }

        const pricing = await fetchPricing({ base, apiKey, model, fetchTimeoutMs });

        return new OpenAICompatProvider({
            model,
            url: `${base}/chat/completions`,
            fetchTimeoutMs,
            headers: { Authorization: `Bearer ${apiKey}` },
            contextSize,
            reasoningBudget,
            reasoningStyle: "effort",
            // Per xAI's docs Grok uses cl100k_base. All current Grok variants
            // share the same tokenizer — no per-model dispatch needed.
            countTokens: tokenizerFor("cl100k"),
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
