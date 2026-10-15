import test from "node:test";
import assert from "node:assert/strict";
import {
  DecisionBackendRegistry,
  chooseWithDecisionBackend,
  type DecisionBackend,
  type DecisionOption,
} from "../../open-sse/services/autoCombo/decisionBackend.ts";

const candidates: DecisionOption[] = [
  { id: "glm/glm-5.2", description: "GLM 5.2" },
  { id: "openai/gpt", description: "GPT" },
];

function registryWith(choose: DecisionBackend["choose"]): DecisionBackendRegistry {
  const registry = new DecisionBackendRegistry();
  registry.register({ id: "mock", choose });
  return registry;
}

test("selects only an approved model", async () => {
  const registry = registryWith(async (request) => {
    assert.equal(request.options.length, 2);
    assert.equal(request.state, "fix this bug");
    return {
      choice: "openai/gpt",
      confidence: 0.94,
      probabilities: { "glm/glm-5.2": 0.06, "openai/gpt": 0.94 },
    };
  });
  const result = await chooseWithDecisionBackend({
    backendId: "mock", registry, state: "  fix this bug  ", candidates,
  });
  assert.deepEqual(result, {
    selectedId: "openai/gpt",
    source: "decision", backendId: "mock", reason: "selected", confidence: 0.94,
  });
});

test("different backend IDs implement the same contract", async () => {
  const registry = new DecisionBackendRegistry();
  registry.register({ id: "jev", choose: async () => ({ choice: candidates[1].id, confidence: 0.8 }) });
  registry.register({ id: "microsoft", choose: async () => ({ choice: candidates[0].id, confidence: 0.9 }) });
  const opts = { registry, candidates, state: "explain this" };
  assert.equal((await chooseWithDecisionBackend({ ...opts, backendId: "jev" })).selectedId, candidates[1].id);
  assert.equal((await chooseWithDecisionBackend({ ...opts, backendId: "microsoft" })).selectedId, candidates[0].id);
  assert.throws(() => registry.register({ id: "jev", choose: async () => ({ choice: "a", confidence: 1 }) }));
});

test("missing backend falls back to first approved candidate", async () => {
  const result = await chooseWithDecisionBackend({
    backendId: "absent", registry: new DecisionBackendRegistry(), state: "hello", candidates,
  });
  assert.equal(result.reason, "no_backend");
  assert.equal(result.source, "fallback");
  assert.equal(result.selectedId, candidates[0].id);
});

test("valid session pin avoids external inference", async () => {
  let called = false;
  const result = await chooseWithDecisionBackend({
    backendId: "mock",
    registry: registryWith(async () => {
      called = true;
      return { choice: candidates[0].id, confidence: 1 };
    }),
    candidates, state: "followup", pinnedId: candidates[1].id,
  });
  assert.equal(called, false);
  assert.equal(result.reason, "pinned");
  assert.equal(result.selectedId, candidates[1].id);
});

test("never accepts a model outside the approved pool", async () => {
  const result = await chooseWithDecisionBackend({
    backendId: "mock", registry: registryWith(async () => ({ choice: "secret/model", confidence: 0.99 })),
    state: "hi", candidates,
  });
  assert.equal(result.reason, "invalid_answer");
  assert.equal(result.selectedId, candidates[0].id);
});

test("invalid confidence and probability keys fall back", async () => {
  for (const answer of [
    { choice: candidates[1].id, confidence: Number.NaN },
    { choice: candidates[1].id, confidence: 1.2 },
    { choice: candidates[1].id, confidence: 0.9, probabilities: { "not-in-pool": 0.9 } },
  ]) {
    const result = await chooseWithDecisionBackend({
      backendId: "mock", registry: registryWith(async () => answer), state: "hi", candidates,
    });
    assert.equal(result.reason, "invalid_answer");
  }
});

test("low confidence uses deterministic fallback", async () => {
  const result = await chooseWithDecisionBackend({
    backendId: "mock", registry: registryWith(async () => ({ choice: candidates[1].id, confidence: 0.64 })),
    state: "architecture", candidates, minConfidence: 0.65,
  });
  assert.equal(result.reason, "low_confidence");
  assert.equal(result.selectedId, candidates[0].id);
});

test("empty prompt avoids external call", async () => {
  let called = false;
  const result = await chooseWithDecisionBackend({
    backendId: "mock",
    registry: registryWith(async () => {
      called = true;
      return { choice: candidates[1].id, confidence: 1 };
    }),
    state: "   ", candidates,
  });
  assert.equal(called, false);
  assert.equal(result.reason, "empty_state");
});

test("backend failure does not leak upstream error details", async () => {
  const result = await chooseWithDecisionBackend({
    backendId: "mock",
    registry: registryWith(async () => {
      throw new Error("Bearer hidden-secret");
    }),
    candidates, state: "fix issue",
  });
  assert.equal(result.reason, "backend_error");
  assert.doesNotMatch(JSON.stringify(result), /hidden-secret|Bearer/);
});

test("hanging backend times out and falls back", async () => {
  let observedAbort = false;
  const result = await chooseWithDecisionBackend({
    backendId: "mock",
    registry: registryWith(async ({ signal }) => {
      await new Promise<never>((_resolve, reject) => {
        signal.addEventListener("abort", () => {
          observedAbort = true;
          reject(new Error("aborted"));
        }, { once: true });
      });
      return { choice: candidates[1].id, confidence: 1 };
    }),
    state: "heavy task", candidates, timeoutMs: 10,
  });
  assert.equal(observedAbort, true);
  assert.equal(result.reason, "timeout");
});

test("caller cancellation does not invoke fallback", async () => {
  const controller = new AbortController();
  const promise = chooseWithDecisionBackend({
    backendId: "mock",
    registry: registryWith(async ({ signal }) => await new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
    })),
    state: "heavy task", candidates, signal: controller.signal,
  });
  controller.abort();
  await assert.rejects(promise, /cancelled/);
});

test("invalid, duplicate and empty candidate pools fail closed", async () => {
  const args = { backendId: "mock", registry: new DecisionBackendRegistry(), state: "hi" };
  await assert.rejects(() => chooseWithDecisionBackend({ ...args, candidates: [] }), /No eligible/);
  await assert.rejects(() => chooseWithDecisionBackend({ ...args, candidates: [candidates[0], candidates[0]] }), /unique/);
  await assert.rejects(() => chooseWithDecisionBackend({ ...args, candidates: [{ id: "", description: "bad" }] }), /unique/);
});
