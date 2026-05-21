# @plurnk/plurnk-providers-xai

xAI provider for [plurnk-service](https://github.com/plurnk/plurnk-service). Routes `xai/{model}` aliases through xAI's OpenAI-compatible chat-completions endpoint, with real per-token pricing pulled from `/v1/language-models` at construction time.

## install

```
npm install @plurnk/plurnk-providers-xai
```

Requires Node ≥ 25 (native TypeScript).

## use

```ts
import Xai from "@plurnk/plurnk-providers-xai";

const provider = await Xai.fromEnv(process.env, "grok-4.3");

const result = await provider.generate({
    messages: [
        { role: "system", content: "You are a plurnk agent." },
        { role: "user",   content: "What is the capital of France?" },
    ],
});
```

## env

| Variable | Required | Notes |
|---|---|---|
| `XAI_API_KEY` | yes | Bearer token from console.x.ai |
| `XAI_BASE_URL` | no | Override the API root. Default `https://api.x.ai/v1` |
| `XAI_CONTEXT_SIZE` | no | Override context window when the per-family default doesn't apply (rare; new model not yet in the table) |
| `PLURNK_REASON` | no | Universal reasoning budget (PROVIDERS.md §3.8); see translation table below |
| `PLURNK_PROVIDER_FETCH_TIMEOUT` | no | Universal fetch timeout in ms; default `600000` (10m) |

## pricing

Real, pulled from xAI's `/v1/language-models/{id}` at `fromEnv` time. xAI returns three distinct rates per model in pico-dollars per token:

- `prompt_text_token_price` — applied to the non-cached portion of `prompt_tokens`
- `cached_prompt_text_token_price` — applied to `cached_tokens` (typically lower; xAI's prompt cache discount)
- `completion_text_token_price` — applied to `completion_tokens`

`costFor` does the full three-rate math: `(prompt - cached) × prompt_rate + cached × cached_rate + completion × completion_rate`.

## context window

xAI does **not** expose context window via any documented API endpoint. The sibling ships a per-family alias-prefix table sourced from [docs.x.ai/developers/models](https://docs.x.ai/developers/models):

| Family prefix | Context |
|---|---|
| `grok-4.20-multi-agent` | 2,000,000 |
| `grok-4.1-fast` | 2,000,000 |
| `grok-4.20` | 1,000,000 |
| `grok-4.3` | 1,000,000 |
| `grok-code-fast` | 256,000 |

For aliases not matching any prefix, set `XAI_CONTEXT_SIZE` explicitly. The table updates with xAI's docs page; PRs welcome.

## reasoning

xAI's reasoning is a tiered `reasoning_effort` body param (`low | medium | high`), not a token budget. The universal `PLURNK_REASON` (a numeric token budget) translates as:

| PLURNK_REASON | reasoning_effort |
|---|---|
| `0` (default) | omit (no reasoning) |
| `1`–`1000` | `low` |
| `1001`–`4000` | `medium` |
| `4001`+ | `high` |

Some Grok models reject the param entirely (the non-reasoning variants); requests against those will 400 if `PLURNK_REASON > 0`. Pick a reasoning-capable alias (`grok-4.3`, `grok-4.20-0309-reasoning`) when reasoning is required.

## tokenization

Real `cl100k_base` via [`gpt-tokenizer`](https://www.npmjs.com/package/gpt-tokenizer) (sync, pure JS). xAI's docs document Grok as using cl100k_base; this sibling encodes `countTokens(text)` accordingly. No per-model dispatch — all current Grok variants share the same encoding.

## license

MIT.
