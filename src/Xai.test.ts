import test, { mock } from "node:test";
import assert from "node:assert/strict";
import Xai from "./Xai.ts";

// Minimum env that satisfies all required guards in fromEnv. Tests that need
// to exercise one specific knob override its key on top of this.
const baseEnv = Object.freeze({
    XAI_API_KEY: "sk-test",
    PLURNK_PROVIDERS_FETCH_TIMEOUT: "600000",
    PLURNK_PROVIDERS_REASONING: "off", PLURNK_PROVIDERS_TEMPERATURE: "0.2", PLURNK_PROVIDERS_REPEAT_PENALTY: "1.15", PLURNK_PROVIDERS_FREQUENCY_PENALTY: "0.4", PLURNK_PROVIDERS_RETRY_DELAY: "1", PLURNK_PROVIDERS_PROBE_ATTEMPTS: "3", PLURNK_PROVIDERS_PROBE_DELAY: "1",
    PLURNK_PROVIDERS_RETRY_ATTEMPTS: "0",
});

// Mock the /language-models pricing probe. `entry` becomes the per-id response.
const mockPricing = (entry: unknown) => {
    const calls: string[] = [];
    mock.method(globalThis, "fetch", async (url: string) => {
        calls.push(String(url));
        return new Response(JSON.stringify(entry), { status: 200 });
    });
    return calls;
};

const pricingEntry = {
    id: "grok-4.3",
    prompt_text_token_price: 12500,
    cached_prompt_text_token_price: 2000,
    completion_text_token_price: 25000,
};

test.afterEach(() => mock.restoreAll());

// — fromEnv env guards —

test("fromEnv: throws when XAI_API_KEY is unset", async () => {
    await assert.rejects(() => Xai.fromEnv({}, "grok-4.3"), /XAI_API_KEY must be set/);
});

test("fromEnv: throws when PLURNK_PROVIDERS_FETCH_TIMEOUT is unset", async () => {
    await assert.rejects(
        () => Xai.fromEnv({ XAI_API_KEY: "sk-test", PLURNK_PROVIDERS_REASONING: "off" }, "grok-4.3"),
        /PLURNK_PROVIDERS_FETCH_TIMEOUT must be set/,
    );
});

test("fromEnv: throws when PLURNK_PROVIDERS_REASONING is not a valid mode", async () => {
    await assert.rejects(
        () => Xai.fromEnv({ ...baseEnv, PLURNK_PROVIDERS_REASONING: "8192" }, "grok-4.3"),
        /PLURNK_PROVIDERS_REASONING must be one of/,
    );
});

test("fromEnv: throws when PLURNK_PROVIDERS_CONTEXT_SIZE is non-numeric", async () => {
    await assert.rejects(
        () => Xai.fromEnv({ ...baseEnv, PLURNK_PROVIDERS_CONTEXT_SIZE: "huge" }, "grok-4.3"),
        /PLURNK_PROVIDERS_CONTEXT_SIZE must be a non-negative integer/,
    );
});

test("generate failure carries the provider:xai telemetry source (SPEC §12)", async () => {
    const { ProviderError } = await import("@plurnk/plurnk-providers");
    mock.method(globalThis, "fetch", async (url: string) => {
        if (String(url).includes("/language-models")) return new Response(JSON.stringify(pricingEntry), { status: 200 });
        return new Response("rate limited", { status: 429 });
    });
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    await assert.rejects(() => p.generate({ runId: "r", messages: [] }), (err: unknown) => {
        assert.ok(err instanceof ProviderError);
        assert.equal(err.kind, "rate_limit");
        assert.equal(err.toTelemetryEvent().source, "provider:xai");
        return true;
    });
});

// — context resolution —

test("fromEnv: resolves contextSize from the prefix table and probes pricing", async () => {
    const calls = mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    assert.equal(p.model, "grok-4.3");
    assert.equal(p.contextSize, 1_000_000);
    assert.equal(calls[0], "https://api.x.ai/v1/language-models/grok-4.3");
});

test("fromEnv: Grok Build (grok-build-0.1) resolves to 256k", async () => {
    mockPricing({ ...pricingEntry, id: "grok-build-0.1" });
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-build-0.1");
    assert.equal(p.contextSize, 256_000);
});

// #35: grok-build / grok-code-fast have NO reasoning channel — the provider must
// never emit reasoning_effort for them (xAI 400s), even under a REASONING intent.
const wireBodyFor = async (model: string, thinkingEnv: Record<string, string>) => {
    let body: Record<string, unknown> = {};
    mock.method(globalThis, "fetch", async (url: string, init?: RequestInit) => {
        if (String(url).includes("/language-models")) return new Response(JSON.stringify({ ...pricingEntry, id: model }), { status: 200 });
        if (init?.body !== undefined) body = JSON.parse(String(init.body));
        return new Response(new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode("data: [DONE]")); c.close(); } }), { status: 200 });
    });
    const p = await Xai.fromEnv({ ...baseEnv, ...thinkingEnv }, model);
    await p.generate({ runId: "r", messages: [] });
    mock.restoreAll();
    return body;
};

test("#35: grok-build emits NO reasoning_effort even with REASONING=on (coding model, no reasoning channel)", async () => {
    const body = await wireBodyFor("grok-build-0.1", { PLURNK_PROVIDERS_REASONING: "on", PLURNK_PROVIDERS_REASONING_BUDGET: "4096" });
    assert.equal("reasoning_effort" in body, false);
});

test("#35: grok-code-fast likewise sends no reasoning param", async () => {
    const body = await wireBodyFor("grok-code-fast-1", { PLURNK_PROVIDERS_REASONING: "adaptive" });
    assert.equal("reasoning_effort" in body, false);
});

test("#35: a reasoning grok (grok-4.3) STILL sends reasoning_effort — the fix is model-scoped", async () => {
    const body = await wireBodyFor("grok-4.3", { PLURNK_PROVIDERS_REASONING: "on", PLURNK_PROVIDERS_REASONING_BUDGET: "4096" });
    assert.equal(body.reasoning_effort, "high");
});

test("#36: data-capture knobs flow through the xai daughter (grok scraping alias)", async () => {
    const on = await wireBodyFor("grok-4.3", { PLURNK_PROVIDERS_REASONING: "off", PLURNK_PROVIDERS_LOGPROB: "3" });
    assert.equal(on.logprobs, true);
    assert.equal(on.top_logprobs, 3);
    const off = await wireBodyFor("grok-4.3", { PLURNK_PROVIDERS_REASONING: "off" });
    assert.equal("logprobs" in off, false); // off by default — serving turns unchanged
});

test("fromEnv: longest-prefix-wins on context lookup", async () => {
    mockPricing({ ...pricingEntry, id: "grok-4.20-multi-agent-0309" });
    // "grok-4.20-multi-agent" prefix (2M) wins over "grok-4.20" prefix (1M).
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.20-multi-agent-0309");
    assert.equal(p.contextSize, 2_000_000);
});

test("fromEnv: PLURNK_PROVIDERS_CONTEXT_SIZE env overrides the prefix table", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv, PLURNK_PROVIDERS_CONTEXT_SIZE: "131072" }, "grok-4.3");
    assert.equal(p.contextSize, 131072);
});

test("fromEnv: throws when alias matches no prefix AND PLURNK_PROVIDERS_CONTEXT_SIZE unset", async () => {
    mockPricing(pricingEntry); // pricing is fine; the throw is specifically the context one
    await assert.rejects(
        () => Xai.fromEnv({ ...baseEnv }, "grok-7-unknown"),
        /no context-window known for "grok-7-unknown"/,
    );
});

// — pricing probe —

test("fromEnv: falls back to list endpoint on 404 from per-id endpoint", async () => {
    let callCount = 0;
    mock.method(globalThis, "fetch", async () => {
        callCount++;
        if (callCount === 1) return new Response("not found", { status: 404 });
        return new Response(JSON.stringify({ models: [
            { id: "grok-4.3", aliases: ["grok-4.3-latest"], prompt_text_token_price: 12500, cached_prompt_text_token_price: 2000, completion_text_token_price: 25000 },
        ] }), { status: 200 });
    });

    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3-latest");
    assert.equal(callCount, 2, "should have fallen back to list endpoint");
    assert.equal(p.contextSize, 1_000_000); // matches "grok-4.3" prefix
});

// — Provider surface on the constructed instance —

// Raw price fields are usd_ticks (1e-10 USD); costFor returns pico (1e-12) — so
// every expected value is Σ(tokens × raw_price) × 100 (#38).
test("costFor: three-rate math with cached subset of prompt", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    // 1000 prompt (200 cached) + 100 completion
    // = [(800 × 12500) + (200 × 2000) + (100 × 25000)] × 100
    // = 12_900_000 ticks × 100 = 1_290_000_000 pico
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 200, reasoning: 0, total: 1100 }), 1_290_000_000);
});

test("costFor: cached=0 collapses to prompt+completion", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    // (1000 × 12500 + 100 × 25000) × 100 = 15_000_000 ticks × 100 = 1_500_000_000 pico
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 0, reasoning: 0, total: 1100 }), 1_500_000_000);
});

test("costFor: reasoning billed at completion rate while distinct cached rate still applies", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    // 1000 prompt (200 cached) + 100 completion + 50 reasoning
    // = [(800 × 12500) + (200 × 2000) + ((100 + 50) × 25000)] × 100
    // = 14_150_000 ticks × 100 = 1_415_000_000 pico
    assert.equal(
        p.costFor({ prompt: 1000, completion: 100, cached: 200, reasoning: 50, total: 1150 }),
        1_415_000_000,
    );
});

// #38 money-grade anchor: costFor(usage) must equal the backend's own
// cost_in_usd_ticks × 100, reconciled from a LIVE grok-code-fast-1 completion
// (usage {prompt:129, cached:64, completion:1, reasoning:143}, cost_in_usd_ticks:
// 3_658_000 at prices 10000/2000/20000). This is the whole point of the fix — the
// provider's cost is truthful pico-USD, not a 100×-undercharged tick count.
test("#38: costFor equals the live cost_in_usd_ticks × 100 (truthful pico, not undercharged ticks)", async () => {
    mockPricing({ ...pricingEntry, id: "grok-code-fast-1", prompt_text_token_price: 10000, cached_prompt_text_token_price: 2000, completion_text_token_price: 20000 });
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-code-fast-1");
    const COST_IN_USD_TICKS = 3_658_000; // authoritative, from the live wire
    assert.equal(
        p.costFor({ prompt: 129, cached: 64, completion: 1, reasoning: 143, total: 273 }),
        COST_IN_USD_TICKS * 100,
    );
});

// #39 long-context tier — the live grok-code-fast-1 entry: base 10000/2000/20000,
// long 2× (20000/4000/40000), threshold 200000 (prompt tokens).
const longEntry = {
    id: "grok-code-fast-1",
    prompt_text_token_price: 10000, cached_prompt_text_token_price: 2000, completion_text_token_price: 20000,
    prompt_text_token_price_long_context: 20000, cached_prompt_text_token_price_long_context: 4000, completion_text_token_price_long_context: 40000,
    long_context_threshold: 200000,
};

test("#39: prompt at/under the threshold bills at BASE rates", async () => {
    mockPricing(longEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-code-fast-1");
    // prompt 129 (< 200000) → base: same as #38's 3_658_000 × 100
    assert.equal(p.costFor({ prompt: 129, cached: 64, completion: 1, reasoning: 143, total: 273 }), 365_800_000);
});

test("#39: prompt over the threshold bills the WHOLE request at the 2× LONG tier — live cost_in_usd_ticks × 100", async () => {
    mockPricing(longEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-code-fast-1");
    // LIVE reconciliation: prompt 219131 (cached 64) + 1 completion + 203 reasoning,
    // cost_in_usd_ticks 4_389_756_000 (long tier). costFor = ticks × 100.
    const COST_IN_USD_TICKS_LONG = 4_389_756_000;
    assert.equal(
        p.costFor({ prompt: 219_131, cached: 64, completion: 1, reasoning: 203, total: 219_335 }),
        COST_IN_USD_TICKS_LONG * 100,
    );
    // The same usage at base rates would be exactly half — the undercharge #39 closes.
    assert.equal(COST_IN_USD_TICKS_LONG * 100, 2 * ((219_131 - 64) * 10000 + 64 * 2000 + 204 * 20000) * 100);
});

test("#39: an entry without the long tier bills at base regardless of prompt size (no half-configured tier)", async () => {
    mockPricing(pricingEntry); // grok-4.3 entry — no *_long_context fields
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    const huge = p.costFor({ prompt: 500_000, cached: 0, completion: 0, reasoning: 0, total: 500_000 });
    // 500000 × 12500 × 100 (base) — never a phantom long tier
    assert.equal(huge, 500_000 * 12_500 * 100);
});

