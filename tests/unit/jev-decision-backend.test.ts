import test from "node:test";
import assert from "node:assert/strict";

import {
  DecisionBackendRegistry,
  chooseWithDecisionBackend,
  type DecisionOption,
} from "../../open-sse/services/autoCombo/decisionBackend.ts";
import { createJevDecisionBackend } from "../../open-sse/services/autoCombo/jevDecisionBackend.ts";
import { TYPESAFE_MAX_CHOICE_OPTIONS, TYPESAFE_MAX_STATE_CHARS } from "../../open-sse/services/typesafe/systemOne.ts";

const candidates: DecisionOption[] = [
  { id: "glm/glm-5.2", description: "GLM 5.2" },
  { id: "openai/gpt-5.5", description: "GPT 5.5" },
];

function response(
  choice = "option_1",
  probabilities: Record<string, number> = { option_0: 0.12, option_1: 0.88 },
  confidence = 0.91,
): Response {
  return new Response(JSON.stringify({
    model: "jev-1.13.0",
    answers: { route: { type: "choice", choice, probabilities, confidence } },
    usage: { input_tokens: 120, output_tokens: 11 },
  }), { status: 200, headers: { "Content-Type": "application/json" } });
}

function makeRegistry(fetchImpl: typeof fetch, apiKey: string | null = "ts_test_key") {
  const registry = new DecisionBackendRegistry();
  registry.register(createJevDecisionBackend({ apiKey, fetchImpl }));
  return registry;
}

function choose(
  registry: DecisionBackendRegistry,
  overrides: Partial<Parameters<typeof chooseWithDecisionBackend>[0]> = {},
) {
  return chooseWithDecisionBackend({
    registry, backendId: "jev", candidates, state: "fix a TypeScript issue", ...overrides,
  });
}

test("maps safe TypeSafe criteria IDs to approved OmniRoute model IDs", async () => {
  let called = 0;
  const registry = makeRegistry(async (url, init) => {
    called++;
    assert.equal(String(url), "https://api.typesafe.ai/v1/systemone");
    assert.equal(init?.method, "POST");
    assert.equal((init?.headers as Record<string, string>).Authorization, "Bearer ts_test_key");
    const body = JSON.parse(String(init?.body));
    assert.equal(body.model, "jev-latest");
    assert.equal(body.state, "fix a TypeScript issue");
    assert.deepEqual(body.questions.route.criteria, {
      option_0: "GLM 5.2",
      option_1: "GPT 5.5",
    });
    assert.equal(body.questions.route.type, "choice");
    assert.match(body.questions.route.instructions, /appropriate/i);
    assert.equal(typeof (init?.signal as AbortSignal)?.aborted, "boolean");
    assert.equal(Object.keys(body).sort().join(","), "model,questions,state");
    return response();
  });
  const picked = await choose(registry);
  assert.equal(called, 1);
  assert.deepEqual(picked, {
    selectedId: "openai/gpt-5.5", source: "decision", backendId: "jev",
    reason: "selected", confidence: 0.91,
  });
});

test("maps choice and probability keys back to approved IDs", async () => {
  const backend = createJevDecisionBackend({ apiKey: "fake", fetchImpl: async () => response() });
  const ans = await backend.choose({
    state: "hello", options: candidates, signal: new AbortController().signal,
  });
  assert.equal(ans.choice, candidates[1].id);
  assert.deepEqual(ans.probabilities, {
    [candidates[0].id]: 0.12,
    [candidates[1].id]: 0.88,
  });
});

test("supports configured System One model", async () => {
  const registry = new DecisionBackendRegistry();
  registry.register(createJevDecisionBackend({
    apiKey: "fake", model: "jev-1.13.0",
    fetchImpl: async (_url, init) => {
      assert.equal(JSON.parse(String(init?.body)).model, "jev-1.13.0");
      return response();
    },
  }));
  assert.equal((await choose(registry)).selectedId, candidates[1].id);
});

test("missing TypeSafe API key falls back without an HTTP call", async () => {
  let called = false;
  const result = await choose(makeRegistry(async () => {
    called = true;
    return response();
  }, null));
  assert.equal(called, false);
  assert.equal(result.source, "fallback");
  assert.equal(result.reason, "backend_error");
  assert.equal(result.selectedId, candidates[0].id);
});

test("upstream HTTP error never reveals credentials or response bodies", async () => {
  const result = await choose(makeRegistry(async () => new Response(
    JSON.stringify({ error: "Bearer ts_test_key (do not log)" }),
    { status: 401 },
  )));
  assert.equal(result.source, "fallback");
  assert.equal(result.reason, "backend_error");
  assert.doesNotMatch(JSON.stringify(result), /ts_test_key|do not log|401/);
});

test("rejects a model not offered as a choice", async () => {
  const result = await choose(makeRegistry(async () => response("option_999")));
  assert.equal(result.reason, "backend_error");
  assert.equal(result.selectedId, candidates[0].id);
});

test("rejects probabilities for unknown choices", async () => {
  const result = await choose(makeRegistry(async () => response("option_1", {
    option_0: 0.1, option_1: 0.7, injected_option: 0.2,
  })));
  assert.equal(result.reason, "backend_error");
  assert.equal(result.selectedId, candidates[0].id);
});

test("rejects probabilities greater than one", async () => {
  const result = await choose(makeRegistry(async () => response("option_1", {
    option_0: 0.2, option_1: 1.3,
  })));
  assert.equal(result.reason, "backend_error");
});

test("rejects probabilities missing the winning choice", async () => {
  const result = await choose(makeRegistry(async () => response("option_1", {
    option_0: 1,
  })));
  assert.equal(result.reason, "backend_error");
});

test("uses the common confidence threshold", async () => {
  const result = await choose(makeRegistry(async () => response(
    "option_1", { option_0: 0.1, option_1: 0.9 }, 0.5,
  )));
  assert.equal(result.reason, "low_confidence");
  assert.equal(result.selectedId, candidates[0].id);
});

test("eligible pinned model bypasses JEV entirely", async () => {
  let called = false;
  const result = await choose(makeRegistry(async () => {
    called = true;
    return response();
  }), { pinnedId: candidates[1].id });
  assert.equal(called, false);
  assert.equal(result.source, "pin");
  assert.equal(result.selectedId, candidates[1].id);
});

test("an empty user message avoids sending anything to TypeSafe", async () => {
  let called = false;
  const result = await choose(makeRegistry(async () => {
    called = true;
    return response();
  }), { state: "  " });
  assert.equal(called, false);
  assert.equal(result.reason, "empty_state");
});

test("does not send more than the System One maximum of 255 choices", async () => {
  let called = false;
  const many = Array.from({ length: TYPESAFE_MAX_CHOICE_OPTIONS + 1 }, (_, i) => ({
    id: `model_${i}`, description: `Model ${i}`,
  }));
  const result = await choose(makeRegistry(async () => {
    called = true;
    return response();
  }), { candidates: many });
  assert.equal(called, false);
  assert.equal(result.source, "fallback");
  assert.equal(result.selectedId, many[0].id);
});

test("reuses TypeSafe state length limit", async () => {
  const state = "x".repeat(TYPESAFE_MAX_STATE_CHARS + 50);
  const picked = await choose(makeRegistry(async (_url, init) => {
    assert.equal(JSON.parse(String(init?.body)).state.length, TYPESAFE_MAX_STATE_CHARS);
    return response();
  }), { state });
  assert.equal(picked.source, "decision");
});

test("hanging TypeSafe fetch is cancelled on router timeout", async () => {
  let aborted = false;
  const registry = makeRegistry(async (_url, init) => new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => {
      aborted = true;
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  }));
  const result = await choose(registry, { timeoutMs: 10 });
  assert.equal(aborted, true);
  assert.equal(result.reason, "timeout");
});

test("caller cancellation propagates through JEV without fallback", async () => {
  const abort = new AbortController();
  const registry = makeRegistry(async (_url, init) => new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  }));
  const promise = choose(registry, { signal: abort.signal });
  abort.abort();
  await assert.rejects(promise, /cancelled/);
});

test("malformed TypeSafe payload is treated as an unavailable decision", async () => {
  const result = await choose(makeRegistry(async () => new Response("{broken", { status: 200 })));
  assert.equal(result.reason, "backend_error");
  assert.equal(result.selectedId, candidates[0].id);
});
