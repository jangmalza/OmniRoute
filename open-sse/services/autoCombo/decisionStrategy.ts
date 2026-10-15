/**
 * Opt-in, vendor-neutral semantic strategy for Auto-Combo.
 *
 * The backend chooses a model from an approved candidate pool; OmniRoute's
 * existing rules strategy still chooses the provider connection for that model.
 * An unconfigured/unreliable decision never bypasses operational scoring.
 */
import type { RouterStrategy, RoutingContext, RoutingDecision } from "./routerStrategy.ts";
import type { ProviderCandidate } from "./scoring.ts";
import {
  DecisionBackendRegistry,
  chooseWithDecisionBackend,
  type DecisionBackend,
  type DecisionOption,
} from "./decisionBackend.ts";
import { createJevDecisionBackend } from "./jevDecisionBackend.ts";
import { TYPESAFE_PROVIDER_ID, TYPESAFE_MAX_STATE_CHARS } from "../typesafe/systemOne.ts";

export interface AutoDecisionConfig {
  backendId: string;
  minConfidence?: number;
  timeoutMs?: number;
  /** Respects the caller's existing connection allowlist when loading credentials. */
  allowedConnections?: string[] | null;
  /** Latest user text only. Do not pass full transcripts or system instructions. */
  state?: string;
  signal?: AbortSignal | null;
}

export type DecisionBackendResolver = (
  backendId: string,
  config: AutoDecisionConfig
) => Promise<DecisionBackend | null>;

/** Never accept arbitrary provider URLs or secrets from persisted combo config. */
export function parseAutoDecisionConfig(raw: unknown): Pick<AutoDecisionConfig, "backendId" | "minConfidence" | "timeoutMs"> {
  const obj = raw && typeof raw === "object" && !Array.isArray(raw)
    ? raw as Record<string, unknown> : {};
  const backendId = obj.backendId === undefined || obj.backendId === "jev" ? "jev" : "unavailable";
  const minConfidence = typeof obj.minConfidence === "number" &&
    Number.isFinite(obj.minConfidence) && obj.minConfidence >= 0 && obj.minConfidence <= 1
    ? obj.minConfidence : undefined;
  const timeoutMs = typeof obj.timeoutMs === "number" &&
    Number.isFinite(obj.timeoutMs) && obj.timeoutMs > 0 && obj.timeoutMs <= 30_000
    ? obj.timeoutMs : undefined;
  return { backendId, minConfidence, timeoutMs };
}

function readUserContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => {
    if (!part || typeof part !== "object" || Array.isArray(part)) return "";
    const row = part as { type?: unknown; text?: unknown };
    return (row.type === "text" || row.type === "input_text") && typeof row.text === "string"
      ? row.text : "";
  }).filter(Boolean).join("\n");
}

/** Extract only the last user turn across Chat Completions and Responses APIs. */
export function extractDecisionUserText(body: Record<string, unknown>): string {
  const messages = Array.isArray(body.messages) ? body.messages : null;
  if (messages) {
    for (let index = messages.length - 1; index >= 0; index--) {
      const msg = messages[index];
      if (msg && typeof msg === "object" && msg.role === "user") {
        return readUserContent(msg.content).trim().slice(0, TYPESAFE_MAX_STATE_CHARS);
      }
    }
  }
  if (typeof body.input === "string") return body.input.trim().slice(0, TYPESAFE_MAX_STATE_CHARS);
  if (Array.isArray(body.input)) {
    for (let index = body.input.length - 1; index >= 0; index--) {
      const msg = body.input[index];
      if (typeof msg === "string") return msg.trim().slice(0, TYPESAFE_MAX_STATE_CHARS);
      if (msg && typeof msg === "object" && msg.role === "user") {
        return readUserContent(msg.content ?? msg.text).trim().slice(0, TYPESAFE_MAX_STATE_CHARS);
      }
    }
  }
  return typeof body.prompt === "string" ? body.prompt.trim().slice(0, TYPESAFE_MAX_STATE_CHARS) : "";
}

async function resolveStoredDecisionBackend(
  backendId: string,
  config: AutoDecisionConfig
): Promise<DecisionBackend | null> {
  if (backendId !== "jev") return null;
  // Lazy import avoids loading the DB/auth implementation unless explicitly
  // selected. The TypeSafe credential is never read from public combo JSON.
  const { getProviderCredentials } = await import("../../../src/sse/services/auth.ts");
  const credentials = await getProviderCredentials(
    TYPESAFE_PROVIDER_ID, null, config.allowedConnections ?? null
  );
  const apiKey = credentials && typeof (credentials as { apiKey?: unknown }).apiKey === "string"
    ? (credentials as { apiKey: string }).apiKey : null;
  return createJevDecisionBackend({ apiKey });
}

export class AutoDecisionStrategyImpl implements RouterStrategy {
  readonly name = "decision";
  readonly description = "An opt-in decision backend selects a model; the existing rules scorer selects its connection";
  private readonly fallback: RouterStrategy;
  private readonly resolveBackend: DecisionBackendResolver;

  constructor(options: { fallback: RouterStrategy; resolveBackend?: DecisionBackendResolver }) {
    this.fallback = options.fallback;
    this.resolveBackend = options.resolveBackend ?? resolveStoredDecisionBackend;
  }

  select(pool: ProviderCandidate[], context: RoutingContext): RoutingDecision {
    return this.fallbackDecision(pool, context, "sync_call");
  }

  async selectAsync(pool: ProviderCandidate[], context: RoutingContext): Promise<RoutingDecision> {
    const candidates = pool.filter((candidate) => candidate.circuitBreakerState !== "OPEN");
    if (candidates.length === 0) {
      // Leave no hard-blocked target in a decision request. The existing rules
      // caller remains responsible for preserving its legacy no-candidate behavior.
      return this.fallbackDecision(pool, context, "no_healthy_candidates");
    }
    const settings = context.decision;
    if (!settings?.backendId || settings.backendId === "unavailable") {
      return this.fallbackDecision(candidates, context, "no_backend");
    }
    if (settings.signal?.aborted) throw new Error("Decision request cancelled");
    const state = settings.state?.trim() || "";
    if (!state) return this.fallbackDecision(candidates, context, "empty_state");

    const distinct = new Map<string, { provider: string; model: string; id: string }>();
    for (const candidate of candidates) {
      const identity = JSON.stringify([candidate.provider, candidate.model]);
      if (!distinct.has(identity)) {
        distinct.set(identity, {
          provider: candidate.provider,
          model: candidate.model,
          id: `option_${distinct.size}`,
        });
      }
    }
    if (distinct.size <= 1 || distinct.size > 255) {
      return this.fallbackDecision(candidates, context, "invalid_pool_size");
    }

    const choices = [...distinct.values()];
    const options: DecisionOption[] = choices.map((c) => ({
      id: c.id, description: `${c.provider}/${c.model}`,
    }));

    try {
      const backend = await this.resolveBackend(settings.backendId, settings);
      if (settings.signal?.aborted) throw new Error("Decision request cancelled");
      if (!backend || backend.id !== settings.backendId) {
        return this.fallbackDecision(candidates, context, "backend_unavailable");
      }
      const registry = new DecisionBackendRegistry();
      registry.register(backend);
      const selection = await chooseWithDecisionBackend({
        backendId: settings.backendId,
        registry,
        state,
        candidates: options,
        minConfidence: settings.minConfidence,
        timeoutMs: settings.timeoutMs,
        signal: settings.signal ?? undefined,
      });
      if (selection.source !== "decision") {
        return this.fallbackDecision(candidates, context, selection.reason);
      }
      const chosen = choices.find((c) => c.id === selection.selectedId);
      if (!chosen) return this.fallbackDecision(candidates, context, "invalid_answer");
      const eligibleConnections = candidates.filter(
        (candidate) => candidate.provider === chosen.provider && candidate.model === chosen.model
      );
      const decision = this.fallback.select(eligibleConnections, context);
      return {
        ...decision,
        strategy: this.name,
        reason: `DecisionRouter backend=${backend.id} confidence=${selection.confidence?.toFixed(2) ?? "n/a"}; ${decision.reason}`,
        candidatesConsidered: candidates.length,
      };
    } catch {
      if (settings.signal?.aborted) throw new Error("Decision request cancelled");
      return this.fallbackDecision(candidates, context, "backend_error");
    }
  }

  private fallbackDecision(
    pool: ProviderCandidate[],
    context: RoutingContext,
    reason: string
  ): RoutingDecision {
    const decision = this.fallback.select(pool, context);
    return {
      ...decision,
      reason: `DecisionRouter fallback=${reason}; ${decision.reason}`,
    };
  }
}
