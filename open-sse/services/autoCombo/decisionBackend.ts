/**
 * Decision backend contract for an opt-in semantic routing layer.
 *
 * This module does not dispatch to an upstream model. Its caller must apply
 * credentials, API-key scopes, capabilities, quota, budget, and circuit-breaker
 * checks BEFORE presenting candidates here. Nothing in a decision response may
 * add a new candidate or relax those pre-existing hard gates.
 *
 * Deliberately vendor-neutral: JEV / OpenAI Decisions / Microsoft Decision-1
 * adapters can implement the same Choice contract without changing routing.
 */

export interface DecisionOption {
  /** Stable ID already approved by the gateway (e.g. a model/connection tuple). */
  id: string;
  /** Short non-sensitive label. Never include an API key or raw system prompt. */
  description: string;
}

export interface DecisionRequest {
  /** A minimal, privacy-reviewed excerpt of the user's latest task. */
  state: string;
  options: readonly DecisionOption[];
  signal: AbortSignal;
}

export interface DecisionAnswer {
  /** An option ID, NOT an arbitrary model ID suggested by the provider. */
  choice: string;
  /** Adapter-normalized confidence on [0, 1], not a guarantee of success. */
  confidence: number;
  /** Optional adapter-normalized choice probabilities. */
  probabilities?: Readonly<Record<string, number>>;
}

export interface DecisionBackend {
  readonly id: string;
  choose(request: DecisionRequest): Promise<DecisionAnswer>;
}

export type DecisionReason =
  | "selected"
  | "pinned"
  | "no_backend"
  | "empty_state"
  | "low_confidence"
  | "invalid_answer"
  | "timeout"
  | "backend_error";

export interface DecisionSelection {
  selectedId: string;
  source: "decision" | "fallback" | "pin";
  backendId: string | null;
  reason: DecisionReason;
  confidence: number | null;
}

export interface ChooseWithDecisionOptions {
  backendId: string;
  registry: DecisionBackendRegistry;
  /** Pre-filtered by the existing OmniRoute safety and eligibility gates. */
  candidates: readonly DecisionOption[];
  /** Apply redaction before passing this string to any external backend. */
  state: string;
  /** If eligible, reuse the same session/model without another decision call. */
  pinnedId?: string;
  /** Configurable per backend; scores aren't necessarily calibrated alike. */
  minConfidence?: number;
  /** Hard routing-decision deadline, not the upstream LLM timeout. */
  timeoutMs?: number;
  signal?: AbortSignal;
}

function isValidId(value: string): boolean {
  return /^[a-zA-Z0-9][a-zA-Z0-9_.:/-]*$/.test(value);
}

export class DecisionBackendRegistry {
  private readonly backends = new Map<string, DecisionBackend>();

  register(backend: DecisionBackend): void {
    if (!isValidId(backend.id)) throw new TypeError("Invalid decision backend ID");
    if (this.backends.has(backend.id)) throw new Error("Decision backend already registered");
    this.backends.set(backend.id, backend);
  }

  get(id: string): DecisionBackend | null {
    return this.backends.get(id) ?? null;
  }
}

function isValidAnswer(answer: unknown, allowed: ReadonlySet<string>): answer is DecisionAnswer {
  if (!answer || typeof answer !== "object" || Array.isArray(answer)) return false;
  const row = answer as Partial<DecisionAnswer>;
  if (typeof row.choice !== "string" || !allowed.has(row.choice)) return false;
  if (typeof row.confidence !== "number" || !Number.isFinite(row.confidence)) return false;
  if (row.confidence < 0 || row.confidence > 1) return false;

  if (row.probabilities !== undefined) {
    if (!row.probabilities || typeof row.probabilities !== "object" || Array.isArray(row.probabilities)) {
      return false;
    }
    for (const [id, probability] of Object.entries(row.probabilities)) {
      if (!allowed.has(id) || !Number.isFinite(probability) || probability < 0 || probability > 1) {
        return false;
      }
    }
    if (!(row.choice in row.probabilities)) return false;
  }
  return true;
}

/**
 * Ask exactly one registered decision backend; keep all fallback behavior local.
 *
 * The caller owns model execution. Neither the raw provider response nor thrown
 * errors are surfaced to the caller's logs/traces by this function.
 */
export async function chooseWithDecisionBackend(
  options: ChooseWithDecisionOptions
): Promise<DecisionSelection> {
  const { candidates, backendId, registry, state, pinnedId, signal } = options;
  if (candidates.length === 0) throw new RangeError("No eligible routing candidates");
  const ids = new Set<string>();
  for (const candidate of candidates) {
    if (!isValidId(candidate.id) || ids.has(candidate.id)) {
      throw new TypeError("Routing candidate IDs must be unique and nonempty");
    }
    ids.add(candidate.id);
  }
  const minConfidence = options.minConfidence ?? 0.65;
  const timeoutMs = options.timeoutMs ?? 2_000;
  if (!Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1) {
    throw new RangeError("minConfidence must be in [0,1]");
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > 30_000) {
    throw new RangeError("timeoutMs must be >0 and <=30000");
  }
  if (signal?.aborted) throw new Error("Decision request cancelled");

  const backend = registry.get(backendId);
  const fallback = (reason: DecisionReason): DecisionSelection => ({
    selectedId: candidates[0].id,
    source: "fallback",
    backendId: backend?.id ?? null,
    reason,
    confidence: null,
  });

  if (pinnedId && ids.has(pinnedId)) {
    return {
      selectedId: pinnedId,
      source: "pin",
      backendId: null,
      reason: "pinned",
      confidence: null,
    };
  }
  if (!backend) return fallback("no_backend");
  if (typeof state !== "string" || !state.trim()) return fallback("empty_state");

  const controller = new AbortController();
  let deadline: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  let onCallerAbort: (() => void) | undefined;
  const timeoutPromise = new Promise<never>((_, reject) => {
    deadline = setTimeout(() => {
      timedOut = true;
      controller.abort();
      reject(new Error("decision_timeout"));
    }, timeoutMs);
  });
  const cancelledPromise = new Promise<never>((_, reject) => {
    if (!signal) return;
    onCallerAbort = () => {
      controller.abort();
      reject(new Error("decision_cancelled"));
    };
    signal.addEventListener("abort", onCallerAbort, { once: true });
  });

  try {
    const answer = await Promise.race([
      backend.choose({
        state: state.trim(),
        options: candidates,
        signal: controller.signal,
      }),
      timeoutPromise,
      cancelledPromise,
    ]);
    if (!isValidAnswer(answer, ids)) return fallback("invalid_answer");
    if (answer.confidence < minConfidence) return fallback("low_confidence");
    return {
      selectedId: answer.choice,
      source: "decision",
      backendId: backend.id,
      reason: "selected",
      confidence: answer.confidence,
    };
  } catch {
    if (signal?.aborted) throw new Error("Decision request cancelled");
    return fallback(timedOut ? "timeout" : "backend_error");
  } finally {
    if (deadline) clearTimeout(deadline);
    if (onCallerAbort) signal?.removeEventListener("abort", onCallerAbort);
  }
}
