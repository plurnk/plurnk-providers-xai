import { encode as encodeCl100k } from "gpt-tokenizer/encoding/cl100k_base";
import { chatCompletionStream, OpenAiHttpError } from "./openaiStream.ts";

const DEFAULT_BASE_URL = "https://api.x.ai/v1";
const DEFAULT_FETCH_TIMEOUT_MS = 600000;

// Context windows from docs.x.ai/developers/models (May 2026). xAI does not
// expose context_window via any documented API endpoint — /v1/language-models
// returns rich pricing data but no window, /v1/models is OpenAI-sparse.
// Operators can override via XAI_CONTEXT_SIZE for new aliases not yet in
// the table. Longest prefix match wins.
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

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ProviderUsage = {
    prompt: number;
    completion: number;
    cached: number;
    total: number;
};

export type ProviderAssistant = {
    content: string;
    reasoning: string | null;
    usage: ProviderUsage;
    finishReason: string | null;
    model: string;
};

export type ProviderResponse = {
    assistant: ProviderAssistant;
    assistantRaw: unknown;
};

// xAI exposes three distinct rates per model in pico-dollars per token.
// `cached` is xAI's prompt-cache discount (typically much lower than prompt);
// applied to the subset of prompt_tokens that came from cache.
export type XaiPricing = {
    prompt_pico_per_token: number;
    cached_pico_per_token: number;
    completion_pico_per_token: number;
};

export type XaiConfig = {
    baseUrl: string;
    apiKey: string;
    model: string;
    contextSize: number;
    fetchTimeoutMs: number;
    // PROVIDERS.md §3.8: numeric budget → reasoning_effort tier translation
    // lives inside generate() so the construction-time config can carry the
    // raw budget for diagnostics.
    reasonBudget: number;
    pricing: XaiPricing;
};

export default class Xai {
    #baseUrl: string;
    #apiKey: string;
    #model: string;
    #contextSize: number;
    #fetchTimeoutMs: number;
    #reasonBudget: number;
    #pricing: XaiPricing;

    constructor(config: XaiConfig) {
        this.#baseUrl = config.baseUrl.replace(/\/$/, "");
        this.#apiKey = config.apiKey;
        this.#model = config.model;
        this.#contextSize = config.contextSize;
        this.#fetchTimeoutMs = config.fetchTimeoutMs;
        this.#reasonBudget = config.reasonBudget;
        this.#pricing = config.pricing;
    }

    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Xai> {
        const apiKey = env.XAI_API_KEY;
        if (apiKey === undefined || apiKey.length === 0) {
            throw new Error("xai provider: XAI_API_KEY must be set");
        }
        const baseUrl = env.XAI_BASE_URL !== undefined && env.XAI_BASE_URL.length > 0
            ? env.XAI_BASE_URL
            : DEFAULT_BASE_URL;
        const fetchTimeoutMs = env.PLURNK_PROVIDER_FETCH_TIMEOUT !== undefined && env.PLURNK_PROVIDER_FETCH_TIMEOUT.length > 0
            ? Number(env.PLURNK_PROVIDER_FETCH_TIMEOUT)
            : DEFAULT_FETCH_TIMEOUT_MS;
        const normalizedBase = baseUrl.replace(/\/$/, "");
        const pricing = await fetchPricing({ baseUrl: normalizedBase, apiKey, model, fetchTimeoutMs });

        // Context: env override > per-family table > throw.
        const envCtx = env.XAI_CONTEXT_SIZE;
        const contextSize = envCtx !== undefined && envCtx.length > 0
            ? Number(envCtx)
            : lookupContextByPrefix(model);
        if (contextSize === null || !Number.isFinite(contextSize) || contextSize <= 0) {
            throw new Error(
                `xai provider: no context-window known for "${model}". xAI's API does not expose this; ` +
                "either pick an alias matching a known family prefix (grok-4.3, grok-4.20*, etc.) " +
                "or set XAI_CONTEXT_SIZE explicitly.",
            );
        }
        return new Xai({
            baseUrl, apiKey, model, contextSize, fetchTimeoutMs,
            reasonBudget: Number(env.PLURNK_REASON ?? "0"),
            pricing,
        });
    }

    get contextSize(): number { return this.#contextSize; }
    get model(): string { return this.#model; }
    get baseUrl(): string { return this.#baseUrl; }
    get pricing(): XaiPricing { return this.#pricing; }

    // Real cl100k_base tokenization via gpt-tokenizer. Per xAI's docs Grok
    // uses cl100k_base (the OpenAI GPT-3.5/4 family encoding). No per-model
    // dispatch needed — all current Grok variants share the same tokenizer.
    countTokens(text: string): number {
        return text.length === 0 ? 0 : encodeCl100k(text).length;
    }

    // Three-rate cost: cached tokens are a SUBSET of prompt_tokens, billed
    // at the discounted cached_rate. The non-cached portion is billed at
    // the full prompt_rate.
    costFor(usage: ProviderUsage): number {
        const nonCachedPrompt = Math.max(0, usage.prompt - usage.cached);
        const promptCost = nonCachedPrompt * this.#pricing.prompt_pico_per_token;
        const cachedCost = usage.cached * this.#pricing.cached_pico_per_token;
        const completionCost = usage.completion * this.#pricing.completion_pico_per_token;
        return Math.round(promptCost + cachedCost + completionCost);
    }

    async generate({ messages, signal }: { messages: ChatMessage[]; signal?: AbortSignal }): Promise<ProviderResponse> {
        const headers: Record<string, string> = {
            Authorization: `Bearer ${this.#apiKey}`,
        };

        const body: Record<string, unknown> = { model: this.#model, messages };
        // PROVIDERS.md §3.8 translation: numeric budget → reasoning_effort
        // tier. Breakpoints documented in README.
        const effort = reasoningEffortFromBudget(this.#reasonBudget);
        if (effort !== null) body.reasoning_effort = effort;

        const timeoutSignal = AbortSignal.timeout(this.#fetchTimeoutMs);
        const effectiveSignal = signal !== undefined ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

        const raw = await chatCompletionStream({
            url: `${this.#baseUrl}/chat/completions`,
            headers,
            body,
            signal: effectiveSignal,
        });

        const usage: ProviderUsage = {
            prompt: raw.usage?.prompt_tokens ?? 0,
            completion: raw.usage?.completion_tokens ?? 0,
            cached: raw.usage?.cached_tokens ?? 0,
            total: raw.usage?.total_tokens ?? 0,
        };

        return {
            assistant: {
                content: raw.content,
                reasoning: raw.reasoning_content.length > 0 ? raw.reasoning_content : null,
                usage,
                finishReason: raw.finish_reason,
                model: raw.model ?? this.#model,
            },
            assistantRaw: raw,
        };
    }
}

// PROVIDERS.md §3.8 budget-to-effort translation. Sibling-owned breakpoints.
const reasoningEffortFromBudget = (budget: number): "low" | "medium" | "high" | null => {
    if (budget <= 0) return null;
    if (budget <= 1000) return "low";
    if (budget <= 4000) return "medium";
    return "high";
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
    baseUrl, apiKey, model, fetchTimeoutMs,
}: { baseUrl: string; apiKey: string; model: string; fetchTimeoutMs: number }): Promise<XaiPricing> => {
    const direct = await fetch(`${baseUrl}/language-models/${encodeURIComponent(model)}`, {
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
    const list = await fetch(`${baseUrl}/language-models`, {
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

export { OpenAiHttpError };
