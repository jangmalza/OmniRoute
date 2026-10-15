import test from "node:test";
import assert from "node:assert/strict";
import {
  AutoDecisionStrategyImpl,
  extractDecisionUserText,
  parseAutoDecisionConfig,
} from "../../open-sse/services/autoCombo/decisionStrategy.ts";
import type { RouterStrategy, RoutingContext, RoutingDecision } from "../../open-sse/services/autoCombo/routerStrategy.ts";
import type { ProviderCandidate } from "../../open-sse/services/autoCombo/scoring.ts";
import type { DecisionBackend } from "../../open-sse/services/autoCombo/decisionBackend.ts";
import { createJevDecisionBackend } from "../../open-sse/services/autoCombo/jevDecisionBackend.ts";
import { parseAutoConfig } from "../../open-sse/services/combo/autoConfig.ts";

const pool = [
  { provider: "anthropic", model: "claude-x", connectionId: "slow", circuitBreakerState: "CLOSED" },
  { provider: "anthropic", model: "claude-x", connectionId: "fast", circuitBreakerState: "CLOSED" },
  { provider: "openai", model: "gpt-x", connectionId: "openai-1", circuitBreakerState: "CLOSED" },
] as ProviderCandidate[];
const ctx: RoutingContext = { taskType: "coding", decision: { backendId: "jev", state: "fix this bug" } };
const fallback: RouterStrategy = {
  name: "rules", description: "test scoring",
  select(candidates): RoutingDecision {
    const candidate = candidates[candidates.length - 1];
    return { provider: candidate.provider, model: candidate.model, connectionId: candidate.connectionId,
      strategy: "rules", reason: "fallback scorer", finalScore: 0.4, candidatesConsidered: candidates.length };
  },
};
function strategy(resolveBackend: () => Promise<DecisionBackend | null>) {
  return new AutoDecisionStrategyImpl({ fallback, resolveBackend });
}
function backend(choice: string, confidence = 0.95): DecisionBackend {
  return { id: "jev", choose: async () => ({ choice, confidence }) };
}

test("decision chooses a model while rules scorer chooses its connection", async () => {
  const result = await strategy(async () => backend("option_0")).selectAsync(pool, ctx);
  assert.equal(result.strategy, "decision");
  assert.equal(result.model, "claude-x");
  assert.equal(result.connectionId, "fast");
});
test("uses the real System One adapter and maps model candidates to safe choices", async () => {
  let calls = 0;
  const adapter = createJevDecisionBackend({ apiKey: "fake-for-test", fetchImpl: async (_url, init) => {
    calls++;
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body.questions.route.criteria, { option_0: "anthropic/claude-x", option_1: "openai/gpt-x" });
    assert.equal(body.state, "fix this bug");
    return new Response(JSON.stringify({
      answers: { route: { type: "choice", choice: "option_1",
        probabilities: { option_0: 0.2, option_1: 0.8 }, confidence: 0.92 } }
    }), { status: 200 });
  } });
  const result = await strategy(async () => adapter).selectAsync(pool, ctx);
  assert.equal(calls, 1);
  assert.equal(result.model, "gpt-x");
  assert.equal(result.strategy, "decision");
});
test("all open-breaker candidates are excluded from external decision", async () => {
  let calls = 0;
  const result = await strategy(async () => ({ id: "jev", choose: async () => {
    calls++;
    return { choice: "option_0", confidence: 1 };
  }})).selectAsync(pool.map(c => ({ ...c, circuitBreakerState: "OPEN" })) as ProviderCandidate[], ctx);
  assert.equal(calls, 0);
  assert.equal(result.strategy, "rules");
});
test("only one eligible model avoids an external decision call", async () => {
  let calls = 0;
  const result = await strategy(async () => ({ id: "jev", choose: async () => {
    calls++;
    return { choice: "option_0", confidence: 1 };
  }})).selectAsync(pool.slice(0,2), ctx);
  assert.equal(calls, 0);
  assert.equal(result.strategy, "rules");
});
test("untrusted choice and low confidence revert to original scorer", async () => {
  for (const adapter of [backend("outside_pool"), backend("option_0", 0.3)]) {
    const result = await strategy(async () => adapter).selectAsync(pool, ctx);
    assert.equal(result.strategy, "rules");
    assert.equal(result.model, "gpt-x");
  }
});
test("missing backend or backend failure never leaks credentials", async () => {
  for (const resolve of [
    async () => null,
    async () => { throw new Error("Bearer secret-key"); }
  ]) {
    const result = await strategy(resolve).selectAsync(pool, ctx);
    assert.equal(result.strategy, "rules");
    assert.doesNotMatch(result.reason, /secret-key/);
  }
});
test("external decision timeout falls back without hanging", async () => {
  const adapter = backend("option_0");
  adapter.choose = async () => new Promise(() => {});
  const result = await strategy(async () => adapter).selectAsync(pool, {
    ...ctx, decision: { ...ctx.decision!, timeoutMs: 10 }
  });
  assert.equal(result.strategy, "rules");
  assert.match(result.reason, /timeout/);
});
test("caller abort never picks a replacement model", async () => {
  const controller = new AbortController();
  const adapter = backend("option_0");
  adapter.choose = async ({signal}) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
  });
  const promise = strategy(async () => adapter).selectAsync(pool, {
    ...ctx, decision: { ...ctx.decision!, signal: controller.signal }
  });
  controller.abort();
  await assert.rejects(promise, /cancelled/);
});
test("input extraction excludes system and tool turns", () => {
  assert.equal(extractDecisionUserText({ messages: [
    { role: "system", content: "secret" }, { role: "user", content: "new task" },
    { role: "tool", content: "sensitive" }
  ]}), "new task");
  assert.equal(extractDecisionUserText({ input: [
    { role: "user", content: [{ type: "input_text", text: "responses task" }] },
    { role: "tool", content: "sensitive" }
  ]}), "responses task");
});
test("configured backend is selected only when explicitly activated", () => {
  const parsed = parseAutoConfig({ name: "example", models: [], strategy: "auto", autoConfig: {
    routerStrategy: "decision", decision: { backendId: "jev", minConfidence: 0.8 }
  } }, []);
  assert.equal(parsed.routingStrategy, "decision");
  assert.equal(parsed.decisionConfig.backendId, "jev");
  assert.equal(parsed.decisionConfig.minConfidence, 0.8);
  assert.equal(parseAutoDecisionConfig({ backendId: "untrusted" }).backendId, "unavailable");
  assert.equal(parseAutoConfig({ name:"default", models:[], strategy:"auto" }, []).routingStrategy,"rules");
});
