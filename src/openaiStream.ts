type StreamRequest = {
    url: string;
    headers: Record<string, string>;
    body: Record<string, unknown>;
    signal: AbortSignal;
};

export type StreamResponse = {
    model: string | null;
    content: string;
    reasoning_content: string;
    finish_reason: string | null;
    usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; cached_tokens?: number } | null;
    chunkMetadata: Record<string, unknown>;
};

export class OpenAiHttpError extends Error {
    readonly status: number;
    readonly body: string;
    readonly retryAfter: number | null;
    constructor(status: number, body: string, retryAfter: number | null) {
        super(`OpenAI ${status} - ${body}`);
        this.status = status;
        this.body = body;
        this.retryAfter = retryAfter;
    }
}

const parseRetryAfter = (header: string | null): number | null => {
    if (header === null) return null;
    const asInt = Number.parseInt(header, 10);
    if (Number.isFinite(asInt)) return asInt * 1000;
    const asDate = Date.parse(header);
    if (Number.isFinite(asDate)) return Math.max(0, asDate - Date.now());
    return null;
};

// SSE client for OpenAI-compatible /chat/completions. Adapted from rummy's
// proven implementation. Streaming keeps long completions alive through CDN
// proxies. Returns aggregated response; throws OpenAiHttpError on non-2xx.
export const chatCompletionStream = async ({ url, headers, body, signal }: StreamRequest): Promise<StreamResponse> => {
    const requestBody = { ...body, stream: true, stream_options: { include_usage: true } };

    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: JSON.stringify(requestBody),
        signal,
    });

    if (!response.ok) {
        const errorBody = await response.text();
        throw new OpenAiHttpError(response.status, errorBody, parseRetryAfter(response.headers.get("retry-after")));
    }

    if (response.body === null) throw new Error("OpenAI response body is null");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let content = "";
    let reasoning_content = "";
    let usage: StreamResponse["usage"] = null;
    let model: string | null = null;
    let finish_reason: string | null = null;
    const chunkMetadata: Record<string, unknown> = {};

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trimStart();
            if (payload === "[DONE]" || payload === "") continue;

            let chunk: Record<string, unknown>;
            try { chunk = JSON.parse(payload) as Record<string, unknown>; } catch { continue; }

            if (typeof chunk.model === "string") model = chunk.model;
            if (chunk.usage !== undefined && chunk.usage !== null) usage = chunk.usage as StreamResponse["usage"];

            for (const [k, v] of Object.entries(chunk)) {
                if (k === "choices" || k === "usage") continue;
                chunkMetadata[k] = v;
            }

            const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
            const choice = choices?.[0];
            if (choice === undefined) continue;
            if (typeof choice.finish_reason === "string") finish_reason = choice.finish_reason;

            const delta = choice.delta as Record<string, unknown> | undefined;
            if (delta === undefined) continue;
            if (typeof delta.content === "string") content += delta.content;
            // Reasoning surfaces under different field names per provider.
            if (typeof delta.reasoning_content === "string") reasoning_content += delta.reasoning_content;
            if (typeof delta.reasoning === "string") reasoning_content += delta.reasoning;
            if (typeof delta.thinking === "string") reasoning_content += delta.thinking;
        }
    }

    return { model, content, reasoning_content, finish_reason, usage, chunkMetadata };
};
