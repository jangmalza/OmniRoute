/**
 * TypeSafe JEV adapter for the vendor-neutral decision backend contract.
 *
 * Reuses the existing System One client. This adapter only recommends one of
 * the caller's already-eligible candidates: it never authorizes or dispatches
 * a model. API credentials are held in the closure, not stored in decisions.
 */
import {
  evaluateSystemOneChoice,
  TYPESAFE_MAX_CHOICE_OPTIONS,
} from "../typesafe/systemOne.ts";
import type { DecisionAnswer, DecisionBackend, DecisionRequest } from "./decisionBackend.ts";

export interface JevDecisionBackendConfig {
  /** Configure via OmniRoute's existing encrypted TypeSafe credential store. */
  apiKey: string | null | undefined;
  /** Optional TypeSafe model override; defaults to jev-latest. */
  model?: string;
  /** Optional upstream deadline; the router also enforces its own deadline. */
  timeoutMs?: number;
  /** Inject a transport for deterministic tests; omitted in production. */
  fetchImpl?: typeof fetch;
}

/** Create an independent backend instance, without globally registering it. */
export function createJevDecisionBackend(config: JevDecisionBackendConfig): DecisionBackend {
  return {
    id: "jev",
    async choose(request: DecisionRequest): Promise<DecisionAnswer> {
      if (request.signal.aborted) throw new Error("JEV decision cancelled");
      if (request.options.length === 0 || request.options.length > TYPESAFE_MAX_CHOICE_OPTIONS) {
        throw new RangeError("Unsupported number of JEV choices");
      }

      // Use unambiguous, safe criterion IDs instead of provider/model IDs, which
      // may contain punctuation unsupported by decision backends. The mapping is
      // local to this request, so no external result can add an eligible model.
      const idsByCriterion = new Map<string, string>();
      const criteria: Record<string, string> = Object.create(null);
      for (const [index, option] of request.options.entries()) {
        const criterion = `option_${index}`;
        idsByCriterion.set(criterion, option.id);
        criteria[criterion] = option.description.trim() || option.id;
      }

      const answer = await evaluateSystemOneChoice({
        apiKey: config.apiKey,
        model: config.model,
        timeoutMs: config.timeoutMs,
        fetchImpl: config.fetchImpl,
        signal: request.signal,
        state: request.state,
        criteria,
      });

      // The caller's common selection wrapper catches this and falls back to
      // the first *eligible* model; never propagate vendor error text or keys.
      if (!answer.ok) throw new Error("JEV decision unavailable");

      const choice = idsByCriterion.get(answer.choice);
      if (!choice) throw new Error("JEV returned an unknown choice");

      const probabilities: Array<[string, number]> = [];
      for (const [criterion, probability] of Object.entries(answer.probabilities)) {
        const id = idsByCriterion.get(criterion);
        if (!id || !Number.isFinite(probability) || probability < 0 || probability > 1) {
          throw new Error("JEV returned invalid probabilities");
        }
        probabilities.push([id, probability]);
      }
      if (!Object.hasOwn(answer.probabilities, answer.choice)) {
        throw new Error("JEV probabilities omitted the winning choice");
      }

      return {
        choice,
        confidence: answer.confidence,
        probabilities: Object.fromEntries(probabilities),
      };
    },
  };
}
