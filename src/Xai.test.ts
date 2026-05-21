import test from "node:test";
import assert from "node:assert/strict";
import Xai from "./Xai.ts";

const samplePricing = { prompt_pico_per_token: 12500, cached_pico_per_token: 2000, completion_pico_per_token: 25000 };

test("fromEnv: throws when XAI_API_KEY is unset", async () => {
    await assert.rejects(
        () => Xai.fromEnv({}, "grok-4.3"),
        /XAI_API_KEY must be set/,
    );
});

test("fromEnv: resolves pricing via /v1/language-models/{id} and context from prefix table", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
            id: "grok-4.3",
            prompt_text_token_price: 12500,
            cached_prompt_text_token_price: 2000,
            completion_text_token_price: 25000,
        }),
    })) as unknown as typeof fetch;

    const p = await Xai.fromEnv({ XAI_API_KEY: "sk-test" }, "grok-4.3");
    assert.equal(p.model, "grok-4.3");
    assert.equal(p.contextSize, 1_000_000);  // from per-family prefix table
    assert.deepEqual(p.pricing, samplePricing);
});

test("fromEnv: longest-prefix-wins on context lookup", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
            id: "grok-4.20-multi-agent-0309",
            prompt_text_token_price: 12500,
            cached_prompt_text_token_price: 2000,
            completion_text_token_price: 25000,
        }),
    })) as unknown as typeof fetch;

    // "grok-4.20-multi-agent" prefix (2M) wins over "grok-4.20" prefix (1M).
    const p = await Xai.fromEnv({ XAI_API_KEY: "sk-test" }, "grok-4.20-multi-agent-0309");
    assert.equal(p.contextSize, 2_000_000);
});

test("fromEnv: XAI_CONTEXT_SIZE env overrides the per-family table", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
            id: "grok-4.3",
            prompt_text_token_price: 12500,
            cached_prompt_text_token_price: 2000,
            completion_text_token_price: 25000,
        }),
    })) as unknown as typeof fetch;

    const p = await Xai.fromEnv({ XAI_API_KEY: "sk-test", XAI_CONTEXT_SIZE: "131072" }, "grok-4.3");
    assert.equal(p.contextSize, 131072);  // env override beats the 1M default
});

test("fromEnv: throws when alias matches no prefix AND XAI_CONTEXT_SIZE unset", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
            id: "grok-7-unknown",
            prompt_text_token_price: 12500,
            cached_prompt_text_token_price: 2000,
            completion_text_token_price: 25000,
        }),
    })) as unknown as typeof fetch;

    await assert.rejects(
        () => Xai.fromEnv({ XAI_API_KEY: "sk-test" }, "grok-7-unknown"),
        /no context-window known for "grok-7-unknown"/,
    );
});

test("fromEnv: falls back to list endpoint on 404 from per-id endpoint", async (t) => {
    let callCount = 0;
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => {
        callCount++;
        if (callCount === 1) return { ok: false, status: 404, text: async () => "not found" };
        return {
            ok: true,
            json: async () => ({ models: [
                { id: "grok-4.3", aliases: ["grok-4.3-latest"], prompt_text_token_price: 12500, cached_prompt_text_token_price: 2000, completion_text_token_price: 25000 },
            ] }),
        };
    }) as unknown as typeof fetch;

    const p = await Xai.fromEnv({ XAI_API_KEY: "sk-test" }, "grok-4.3-latest");
    assert.equal(callCount, 2, "should have fallen back to list endpoint");
    assert.deepEqual(p.pricing, samplePricing);
    assert.equal(p.contextSize, 1_000_000);  // matches "grok-4.3" prefix
});

test("contextSize, model, baseUrl exposed on instance", () => {
    const p = new Xai({
        baseUrl: "https://api.x.ai/v1",
        apiKey: "sk-test",
        model: "grok-4.3",
        contextSize: 1_000_000,
        fetchTimeoutMs: 600000,
        reasonBudget: 0,
        pricing: samplePricing,
    });
    assert.equal(p.contextSize, 1_000_000);
    assert.equal(p.model, "grok-4.3");
    assert.equal(p.baseUrl, "https://api.x.ai/v1");
});

test("costFor: three-rate math with cached subset of prompt", () => {
    const p = new Xai({
        baseUrl: "https://api.x.ai/v1",
        apiKey: "sk", model: "grok-4.3", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        pricing: { prompt_pico_per_token: 12500, cached_pico_per_token: 2000, completion_pico_per_token: 25000 },
    });
    // 1000 prompt (200 cached) + 100 completion
    // = (800 × 12500) + (200 × 2000) + (100 × 25000)
    // = 10_000_000 + 400_000 + 2_500_000
    // = 12_900_000 pico
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 200, total: 1100 }), 12_900_000);
});

test("costFor: cached=0 collapses to prompt+completion", () => {
    const p = new Xai({
        baseUrl: "https://api.x.ai/v1",
        apiKey: "sk", model: "grok-4.3", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        pricing: { prompt_pico_per_token: 12500, cached_pico_per_token: 2000, completion_pico_per_token: 25000 },
    });
    // 1000 × 12500 + 100 × 25000 = 12_500_000 + 2_500_000 = 15_000_000
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 0, total: 1100 }), 15_000_000);
});

test("countTokens: real cl100k_base via gpt-tokenizer", () => {
    const p = new Xai({
        baseUrl: "https://api.x.ai/v1",
        apiKey: "sk", model: "grok-4.3", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
        pricing: samplePricing,
    });
    assert.equal(p.countTokens(""), 0);
    // "hello world" is a known cl100k_base 2-token sequence ("hello"|" world").
    assert.equal(p.countTokens("hello world"), 2);
    // Single short word should be one token.
    assert.equal(p.countTokens("Paris"), 1);
    // Real tokenizer beats heuristic: a long English sentence tokenizes to
    // fewer tokens than chars/4 would suggest.
    const sentence = "The quick brown fox jumps over the lazy dog.";
    const heuristic = Math.ceil(sentence.length / 4);
    const real = p.countTokens(sentence);
    assert.ok(real > 0 && real < heuristic, `cl100k tokenized "${sentence}" to ${real}; heuristic was ${heuristic}`);
});
