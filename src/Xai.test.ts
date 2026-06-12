import test, { mock } from "node:test";
import assert from "node:assert/strict";
import Xai from "./Xai.ts";

// Minimum env that satisfies all required guards in fromEnv. Tests that need
// to exercise one specific knob override its key on top of this.
const baseEnv = Object.freeze({
    XAI_API_KEY: "sk-test",
    PLURNK_FETCH_TIMEOUT: "600000",
    PLURNK_REASON: "0",
    PLURNK_PROVIDERS_THINKING: "0",
    PLURNK_PROVIDERS_REASONING: "1",
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

test("fromEnv: throws when PLURNK_FETCH_TIMEOUT is unset", async () => {
    await assert.rejects(
        () => Xai.fromEnv({ XAI_API_KEY: "sk-test", PLURNK_REASON: "0" }, "grok-4.3"),
        /PLURNK_FETCH_TIMEOUT must be set/,
    );
});

test("fromEnv: throws when PLURNK_REASON is non-numeric", async () => {
    await assert.rejects(
        () => Xai.fromEnv({ ...baseEnv, PLURNK_REASON: "lots" }, "grok-4.3"),
        /PLURNK_REASON must be a non-negative integer/,
    );
});

test("fromEnv: throws when PLURNK_PROVIDER_CONTEXT_SIZE is non-numeric", async () => {
    await assert.rejects(
        () => Xai.fromEnv({ ...baseEnv, PLURNK_PROVIDER_CONTEXT_SIZE: "huge" }, "grok-4.3"),
        /PLURNK_PROVIDER_CONTEXT_SIZE must be a non-negative integer/,
    );
});

test("generate failure carries the provider:xai telemetry source (SPEC §12)", async () => {
    const { ProviderError } = await import("@plurnk/plurnk-providers");
    mock.method(globalThis, "fetch", async (url: string) => {
        if (String(url).includes("/language-models")) return new Response(JSON.stringify(pricingEntry), { status: 200 });
        return new Response("rate limited", { status: 429 });
    });
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    await assert.rejects(() => p.generate({ messages: [] }), (err: unknown) => {
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

test("fromEnv: longest-prefix-wins on context lookup", async () => {
    mockPricing({ ...pricingEntry, id: "grok-4.20-multi-agent-0309" });
    // "grok-4.20-multi-agent" prefix (2M) wins over "grok-4.20" prefix (1M).
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.20-multi-agent-0309");
    assert.equal(p.contextSize, 2_000_000);
});

test("fromEnv: PLURNK_PROVIDER_CONTEXT_SIZE env overrides the prefix table", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv, PLURNK_PROVIDER_CONTEXT_SIZE: "131072" }, "grok-4.3");
    assert.equal(p.contextSize, 131072);
});

test("fromEnv: throws when alias matches no prefix AND PLURNK_PROVIDER_CONTEXT_SIZE unset", async () => {
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

test("costFor: three-rate math with cached subset of prompt", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    // 1000 prompt (200 cached) + 100 completion
    // = (800 × 12500) + (200 × 2000) + (100 × 25000)
    // = 10_000_000 + 400_000 + 2_500_000 = 12_900_000 pico
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 200, reasoning: 0, total: 1100 }), 12_900_000);
});

test("costFor: cached=0 collapses to prompt+completion", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    // 1000 × 12500 + 100 × 25000 = 12_500_000 + 2_500_000 = 15_000_000
    assert.equal(p.costFor({ prompt: 1000, completion: 100, cached: 0, reasoning: 0, total: 1100 }), 15_000_000);
});

test("costFor: reasoning billed at completion rate while distinct cached rate still applies", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    // 1000 prompt (200 cached) + 100 completion + 50 reasoning
    // = (800 × 12500) + (200 × 2000) + ((100 + 50) × 25000)
    // = 10_000_000 + 400_000 + 3_750_000 = 14_150_000 pico
    assert.equal(
        p.costFor({ prompt: 1000, completion: 100, cached: 200, reasoning: 50, total: 1150 }),
        14_150_000,
    );
});

test("countTokens: cl100k tokenizer (hello world = 2)", async () => {
    mockPricing(pricingEntry);
    const p = await Xai.fromEnv({ ...baseEnv }, "grok-4.3");
    assert.equal(p.countTokens(""), 0);
    assert.equal(p.countTokens("hello world"), 2);
    assert.equal(p.countTokens("Paris"), 1);
});
