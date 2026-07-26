// xAI provider — a thin fromEnv over the shared OpenAICompatProvider.
// xAI's only bespoke surface is context-window resolution (xAI exposes no
// context_window via any documented API endpoint) and the per-token pricing
// probe against /v1/language-models; everything else (the generate spine,
// usage mapping, reasoning translation) is the framework's.

import {
    computeCost,
    OpenAICompatProvider,
    contextWindowFromEnv,
    parseRequiredInt,
    reasoningFromEnv,
    dataCaptureFromEnv,
    parseRequiredFloat,
    providerSource,
    requireEnv,
    type Provider,
    envelopeFromEnv,
} from "@plurnk/plurnk-providers";

const DEFAULT_BASE_URL = "https://api.x.ai/v1";

// Context windows from docs.x.ai/developers/models (July 2026). xAI does not
// expose context_window via any documented API endpoint — /v1/language-models
// returns rich pricing data but no window, /v1/models is OpenAI-sparse.
// Operators can override via PLURNK_PROVIDERS_CONTEXT_WINDOW for new aliases not
// yet in the table. Longest prefix match wins.
type ModelFamily = Readonly<{
    prefix: string;
    contextWindow: number;
    reasoningEffort: boolean;
}>;

// One table owns both capabilities so an alias cannot resolve its context from
// one family and its reasoning-control behavior from another. Longest prefix
// wins: grok-build-latest is the 4.5 alias, while versioned grok-build-* names
// identify Grok Build 0.1. Build 0.1 reasons, but its Chat Completions endpoint
// rejects the reasoning_effort control; `false` means omit that field, not that
// the model has no internal reasoning.
const MODEL_FAMILIES: readonly ModelFamily[] = Object.freeze([
    { prefix: "grok-4.20-multi-agent", contextWindow: 2_000_000, reasoningEffort: true },
    { prefix: "grok-build-latest", contextWindow: 500_000, reasoningEffort: true },
    { prefix: "grok-4.1-fast", contextWindow: 2_000_000, reasoningEffort: true },
    { prefix: "grok-4.20", contextWindow: 1_000_000, reasoningEffort: true },
    { prefix: "grok-4.5", contextWindow: 500_000, reasoningEffort: true },
    { prefix: "grok-4.3", contextWindow: 1_000_000, reasoningEffort: true },
    { prefix: "grok-build", contextWindow: 256_000, reasoningEffort: false },
    { prefix: "grok-code-fast", contextWindow: 256_000, reasoningEffort: false },
]);

const lookupModelFamily = (model: string): ModelFamily | null => {
    let best: ModelFamily | null = null;
    for (const family of MODEL_FAMILIES) {
        if (model.startsWith(family.prefix) && (best === null || family.prefix.length > best.prefix.length)) {
            best = family;
        }
    }
    return best;
};

export default class Xai {
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Provider> {
        const apiKey = requireEnv(env.XAI_API_KEY, "XAI_API_KEY", "xai");
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_PROVIDERS_FETCH_TIMEOUT, "PLURNK_PROVIDERS_FETCH_TIMEOUT", "xai");
        const streamIdleTimeoutMs = parseRequiredInt(env.PLURNK_PROVIDERS_STREAM_IDLE_TIMEOUT, "PLURNK_PROVIDERS_STREAM_IDLE_TIMEOUT", "xai");
        const reasoning = reasoningFromEnv(env, "xai");
        const rawBase = env.XAI_BASE_URL !== undefined && env.XAI_BASE_URL.length > 0
            ? env.XAI_BASE_URL
            : DEFAULT_BASE_URL;
        const base = rawBase.replace(/\/$/, "");

        // Context: env override > per-family table > throw. Resolved before the
        // pricing probe so an unknown alias fails fast without a network call.
        const family = lookupModelFamily(model);
        const envCtx = contextWindowFromEnv(env, "xai");
        const contextWindow = envCtx !== null ? envCtx : family?.contextWindow ?? null;
        if (contextWindow === null || !Number.isFinite(contextWindow) || contextWindow <= 0) {
            throw new Error(
                `xai provider: no context-window known for "${model}". xAI's API does not expose this; ` +
                "either pick an alias matching a known family prefix (grok-4.5, grok-4.3, etc.) " +
                "or set PLURNK_PROVIDERS_CONTEXT_WINDOW explicitly.",
            );
        }

        const pricing = await fetchPricing({ base, apiKey, model, fetchTimeoutMs });

        return new OpenAICompatProvider({
            model,
            url: `${base}/chat/completions`,
            fetchTimeoutMs,
            streamIdleTimeoutMs,
            headers: { Authorization: `Bearer ${apiKey}` },
            contextWindow,
            reasoning,
            temperature: parseRequiredFloat(env.PLURNK_PROVIDERS_TEMPERATURE, "PLURNK_PROVIDERS_TEMPERATURE", "xai", 0),
            repeatPenalty: parseRequiredFloat(env.PLURNK_PROVIDERS_REPEAT_PENALTY, "PLURNK_PROVIDERS_REPEAT_PENALTY", "xai", 0),
            // No forced frequency_penalty floor on xai (providers-xai#2). The floor's
            // rationale is repetition/loop prevention under a constraining grammar, and
            // xai runs NO grammar (no rails) on modern models; grok-code-fast-1 rejects
            // the parameter outright. A caller wanting a penalty still passes it via
            // `sampling` (frequency_penalty isn't reserved). Omitting -> OpenAICompat's 0-default.
            // #507: envelope reserves (window-fraction floor, absolute overrides).
            ...envelopeFromEnv(env, "xai"),
            retryDelayMs: parseRequiredInt(env.PLURNK_PROVIDERS_RETRY_DELAY, "PLURNK_PROVIDERS_RETRY_DELAY", "xai"),
            retryAttempts: parseRequiredInt(env.PLURNK_PROVIDERS_RETRY_ATTEMPTS, "PLURNK_PROVIDERS_RETRY_ATTEMPTS", "xai"),
            // Opt-in data capture (#36), off by default, per-alias-scopable.
            ...dataCaptureFromEnv(env, "xai"),
            reasoningStyle: family?.reasoningEffort === false ? "none" : "effort",
            // Per xAI's docs Grok uses cl100k_base. All current Grok variants
            // share the same tokenizer — no per-model dispatch needed.
            // Three-rate cost: cached tokens are a SUBSET of prompt_tokens,
            // billed at the discounted cached rate; the non-cached portion is
            // billed at the full prompt rate. computeCost bills billable output
            // (completion + reasoning) at the completion rate.
            //
            // Long-context tier (#39): when the request's PROMPT exceeds
            // long_context_threshold, xAI bills the WHOLE request at the 2× long
            // rates — verified money-grade (219,131-token prompt reconciled to the
            // long tier's cost_in_usd_ticks exactly, double the base figure). The
            // gauge is prompt tokens (cached ⊆ prompt); the boundary is `>` (the
            // exact ±1-token edge at the threshold is a rounding non-event).
            costFor: (usage) => {
                const long = pricing.long !== null
                    && pricing.longContextThreshold !== null
                    && usage.prompt > pricing.longContextThreshold;
                const rates = long ? pricing.long! : pricing.base;
                return computeCost(usage, { input: rates.prompt, output: rates.completion, cached: rates.cached });
            },
            source: providerSource("xai"),
        });
    }
}

// xAI's price fields are denominated in "usd_ticks", NOT pico-USD (#38). One tick
// is 1e-10 USD = 100 pico — verified money-grade against the API's own authority:
// a live completion's usage.cost_in_usd_ticks equals Σ(tokens × raw_price) exactly
// (129/64-cached prompt + 144 output at 10000/2000/20000 ⇒ 3,658,000), and the
// only scalar giving grok-code-fast-1 a realistic price ($1/$2 per Mtok, matching
// the live 1:2 field ratio) is 1 tick = 1e-10 USD. Treating raw as pico undercharges
// 100×. So costFor() (pico contract) requires raw × TICK_TO_PICO.
const TICK_TO_PICO = 100;

// Three distinct rates per token, converted to pico-USD. `cached` is xAI's
// prompt-cache discount (much lower than prompt), applied to the subset of
// prompt_tokens that came from cache.
type Rates = { prompt: number; cached: number; completion: number };
// `base` always applies; `long` is the 2× tier used when prompt tokens exceed
// `longContextThreshold` (#39). Both null when the model exposes no long tier.
type XaiPricing = {
    base: Rates;
    long: Rates | null;
    longContextThreshold: number | null;
};

// /v1/language-models/{id} returns per-model pricing in usd_ticks/token, with a
// parallel `*_long_context` tier gated by `long_context_threshold` (prompt tokens).
// Falls back to /v1/language-models (list) if the per-id endpoint 404s
// (rare; new alias not yet exposed).
type ModelPricingResponse = {
    id?: string;
    prompt_text_token_price?: number;
    cached_prompt_text_token_price?: number;
    completion_text_token_price?: number;
    prompt_text_token_price_long_context?: number;
    cached_prompt_text_token_price_long_context?: number;
    completion_text_token_price_long_context?: number;
    long_context_threshold?: number;
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
    const base: Rates = {
        prompt: entry.prompt_text_token_price * TICK_TO_PICO,
        cached: (entry.cached_prompt_text_token_price ?? entry.prompt_text_token_price) * TICK_TO_PICO,
        completion: entry.completion_text_token_price * TICK_TO_PICO,
    };
    // The long tier is present only when the entry carries BOTH bounds and rates
    // (#39). A partial entry (threshold without long prices, or vice versa) leaves
    // it off — never a half-configured tier that mis-bills.
    const hasLong = entry.long_context_threshold !== undefined
        && entry.prompt_text_token_price_long_context !== undefined
        && entry.completion_text_token_price_long_context !== undefined;
    const long: Rates | null = hasLong
        ? {
            prompt: entry.prompt_text_token_price_long_context! * TICK_TO_PICO,
            cached: (entry.cached_prompt_text_token_price_long_context ?? entry.prompt_text_token_price_long_context!) * TICK_TO_PICO,
            completion: entry.completion_text_token_price_long_context! * TICK_TO_PICO,
        }
        : null;
    return { base, long, longContextThreshold: hasLong ? entry.long_context_threshold! : null };
};
