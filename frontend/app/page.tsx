"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { ModelInfo } from "@/lib/types";

type CheckState = "checking" | "online" | "offline" | "blocked" | "unknown";

interface RuntimeCheck {
  state: CheckState;
  detail: string;
  checkedAt: string | null;
}

interface HealthPayload {
  status?: string;
  service?: string;
  [key: string]: unknown;
}

interface GraphNode {
  id: string;
  label: string;
  sublabel: string;
  state: CheckState | "source" | "design";
  x: number;
  y: number;
}

interface GraphEdge {
  from: string;
  to: string;
  label: string;
  state: CheckState | "source" | "design";
}

const ROUTE_CONTRACTS = [
  {
    route: "/health",
    state: "runtime checked",
    purpose: "gateway liveness",
  },
  {
    route: "/v1/models",
    state: "runtime checked",
    purpose: "model registry discovery",
  },
  {
    route: "/v1/chat/completions",
    state: "source-known",
    purpose: "OpenAI-compatible model calls",
  },
  {
    route: "/v1/rag/search",
    state: "source-known, auth-required",
    purpose: "retrieved snippets from documents and memory",
  },
  {
    route: "/v1/substrate/{name}/context",
    state: "source-known, auth + shard required",
    purpose: "budgeted context packet with citations; no answer generation",
  },
  {
    route: "/v1/substrate/{name}/factuality",
    state: "source-known, auth + shard required",
    purpose: "claim-level evidence receipts",
  },
  {
    route: "/v1/files",
    state: "source-known",
    purpose: "multimodal upload path for images and files",
  },
];

const CONTEXT_PACKET = [
  ["query", "The user question or task being transported."],
  ["scope", "User/team/shard boundary for which knowledge is allowed to move."],
  ["budget", "Maximum chars/items and context shape for target model constraints."],
  ["evidence", "Cited passages, tables, image refs, graph items, or simulator receipts."],
  ["citations", "Source ids and provenance attached to every context item."],
  ["claims", "Atomic statements that downstream models can accept, reject, or verify."],
  ["graph_delta", "Nodes and edges added or strengthened by this exchange."],
  ["policy", "Auth, sharing, abstention, confidence, and review requirements."],
];

function emptyCheck(detail: string): RuntimeCheck {
  return { state: "checking", detail, checkedAt: null };
}

function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function stateClasses(state: CheckState | "source" | "design") {
  if (state === "online") return "border-emerald-300 bg-emerald-50 text-emerald-800";
  if (state === "offline") return "border-rose-300 bg-rose-50 text-rose-800";
  if (state === "blocked") return "border-orange-300 bg-orange-50 text-orange-800";
  if (state === "checking") return "border-amber-300 bg-amber-50 text-amber-800";
  if (state === "source") return "border-sky-300 bg-sky-50 text-sky-800";
  if (state === "design") return "border-violet-300 bg-violet-50 text-violet-800";
  return "border-slate-300 bg-slate-50 text-slate-700";
}

function dotColor(state: CheckState | "source" | "design") {
  if (state === "online") return "#059669";
  if (state === "offline") return "#e11d48";
  if (state === "blocked") return "#ea580c";
  if (state === "checking") return "#d97706";
  if (state === "source") return "#0284c7";
  if (state === "design") return "#7c3aed";
  return "#64748b";
}

function nodeFill(state: CheckState | "source" | "design") {
  if (state === "online") return "#d1fae5";
  if (state === "offline") return "#ffe4e6";
  if (state === "blocked") return "#ffedd5";
  if (state === "checking") return "#fef3c7";
  if (state === "source") return "#e0f2fe";
  if (state === "design") return "#ede9fe";
  return "#f1f5f9";
}

function modelKind(model: ModelInfo) {
  if (model.kind) return model.kind;
  if (model.id === "echo") return "stub";
  return "untyped";
}

function localModelStatus(models: ModelInfo[], registry: RuntimeCheck): RuntimeCheck {
  if (registry.state === "checking") return emptyCheck("waiting for /v1/models");
  if (registry.state === "blocked") {
    return { state: "unknown", detail: "model registry requires auth, so local model registration is not verified", checkedAt: registry.checkedAt };
  }
  if (registry.state === "offline") {
    return { state: "unknown", detail: "gateway is unreachable, so local model registration is not verified", checkedAt: registry.checkedAt };
  }
  const localIds = new Set(["local", "local-poe", "local-fast"]);
  const registered = models.filter((model) => localIds.has(model.id));
  if (registered.length > 0) {
    return {
      state: "online",
      detail: `${registered.map((model) => model.id).join(", ")} registered in the gateway`,
      checkedAt: registry.checkedAt,
    };
  }
  return {
    state: "offline",
    detail: "no local/local-poe/local-fast engine registered by /v1/models",
    checkedAt: registry.checkedAt,
  };
}

class RuntimeHttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "RuntimeHttpError";
    this.status = status;
  }
}

async function fetchJson<T>(url: string, signal: AbortSignal, token?: string | null): Promise<T> {
  const res = await fetch(url, {
    signal,
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status message */
    }
    throw new RuntimeHttpError(res.status, message);
  }
  return (await res.json()) as T;
}

export default function HomePage() {
  const { token, ready } = useAuth();
  const [health, setHealth] = useState<RuntimeCheck>(() => emptyCheck("checking /health"));
  const [registry, setRegistry] = useState<RuntimeCheck>(() => emptyCheck("checking /v1/models"));
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [packetQuery, setPacketQuery] = useState("What evidence should move between a local model, a frontier model, and the knowledge substrate?");

  const refreshRuntime = useCallback(async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 2800);
    const checkedAt = nowLabel();

    setHealth(emptyCheck("checking /health"));
    setRegistry(emptyCheck("checking /v1/models"));

    try {
      const payload = await fetchJson<HealthPayload>(`${API_BASE}/health`, controller.signal);
      setHealth({
        state: payload.status === "ok" ? "online" : "unknown",
        detail: `${payload.service ?? "gateway"} returned status=${payload.status ?? "unknown"}`,
        checkedAt,
      });
    } catch (error) {
      setHealth({
        state: "offline",
        detail: error instanceof Error ? error.message : "request failed",
        checkedAt,
      });
    }

    try {
      const payload = await fetchJson<{ data?: ModelInfo[] }>(`${API_BASE}/v1/models`, controller.signal, token);
      const nextModels = payload.data ?? [];
      setModels(nextModels);
      setSelectedModel((current) => current || nextModels[0]?.id || "");
      setRegistry({
        state: "online",
        detail: `${nextModels.length} model${nextModels.length === 1 ? "" : "s"} returned by the live registry`,
        checkedAt,
      });
    } catch (error) {
      setModels([]);
      if (error instanceof RuntimeHttpError && (error.status === 401 || error.status === 403)) {
        setRegistry({
          state: "blocked",
          detail: `${error.message}; sign in to load the live model registry`,
          checkedAt,
        });
        return;
      }
      setRegistry({
        state: "offline",
        detail: error instanceof Error ? error.message : "request failed",
        checkedAt,
      });
    } finally {
      window.clearTimeout(timeout);
    }
  }, [token]);

  useEffect(() => {
    if (!ready) return;
    void refreshRuntime();
  }, [ready, refreshRuntime]);

  const localModel = useMemo(() => localModelStatus(models, registry), [models, registry]);
  const selected = models.find((model) => model.id === selectedModel);

  const graph = useMemo(() => {
    const modelNodes = models.slice(0, 5).map<GraphNode>((model, index) => ({
      id: `model-${model.id}`,
      label: model.id,
      sublabel: modelKind(model),
      state: model.id === "echo" ? "source" : "online",
      x: 560,
      y: 115 + index * 54,
    }));

    const nodes: GraphNode[] = [
      { id: "frontend", label: "Frontend", sublabel: "this page", state: "online", x: 95, y: 74 },
      { id: "gateway", label: "Gateway", sublabel: "/health", state: health.state, x: 290, y: 74 },
      { id: "registry", label: "Registry", sublabel: "/v1/models", state: registry.state, x: 290, y: 204 },
      { id: "substrate", label: "Knowledge substrate", sublabel: "/v1/substrate/*", state: "source", x: 95, y: 252 },
      { id: "packet", label: "Context packet", sublabel: "transport contract", state: "design", x: 300, y: 350 },
      { id: "review", label: "Receipts + policy", sublabel: "citations, scope, abstain", state: "design", x: 560, y: 350 },
      ...modelNodes,
    ];

    if (models.length === 0) {
      nodes.push({ id: "no-models", label: "No live models shown", sublabel: "registry unavailable or empty", state: registry.state, x: 560, y: 142 });
    }

    const firstModelId = modelNodes[0]?.id ?? "no-models";
    const edges: GraphEdge[] = [
      { from: "frontend", to: "gateway", label: "checks", state: health.state },
      { from: "gateway", to: "registry", label: "lists", state: registry.state },
      { from: "registry", to: firstModelId, label: "serves", state: registry.state },
      { from: "substrate", to: "packet", label: "assemble", state: "source" },
      { from: "packet", to: firstModelId, label: "transport", state: "design" },
      { from: firstModelId, to: "review", label: "claim receipts", state: "design" },
      { from: "review", to: "substrate", label: "graph delta", state: "design" },
    ];
    return { nodes, edges };
  }, [health.state, models, registry.state]);

  return (
    <main className="min-h-screen bg-[#f5f7fb] text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 px-5 py-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="rounded-sm border border-slate-300 bg-slate-50 px-2 py-1 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-600">Mixle</span>
              <span className={`rounded-sm border px-2 py-1 font-mono text-[11px] uppercase tracking-[0.1em] ${stateClasses(health.state)}`}>
                gateway {health.state}
              </span>
              <span className={`rounded-sm border px-2 py-1 font-mono text-[11px] uppercase tracking-[0.1em] ${stateClasses(registry.state)}`}>
                registry {registry.state}
              </span>
            </div>
            <h1 className="max-w-4xl text-3xl font-semibold tracking-normal text-slate-950 md:text-4xl">Runtime And Knowledge Transport Console</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              This page shows live checks when the gateway responds. Source-known routes and context-packet diagrams are labeled separately; they are not model output.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => void refreshRuntime()}
              className="rounded-sm bg-slate-950 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              Refresh runtime
            </button>
            <Link href="/chat" className="rounded-sm border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:border-slate-500">
              Open chat
            </Link>
            <Link href="/documents" className="rounded-sm border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:border-slate-500">
              Documents
            </Link>
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-7xl gap-4 px-5 py-5 md:grid-cols-2 xl:grid-cols-4">
        <StatusCard title="Frontend" state="online" detail="Next.js page is rendering in the browser." checkedAt="now" />
        <StatusCard title="Gateway" state={health.state} detail={`${API_BASE} — ${health.detail}`} checkedAt={health.checkedAt} />
        <StatusCard title="Model Registry" state={registry.state} detail={registry.detail} checkedAt={registry.checkedAt} />
        <StatusCard title="Local Small Model" state={localModel.state} detail={localModel.detail} checkedAt={localModel.checkedAt} />
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(520px,1.35fr)]">
        <div className="space-y-5">
          <Panel title="Live Model Registry" meta={registry.state === "online" ? "runtime response" : "not verified"}>
            {models.length > 0 ? (
              <div className="grid gap-3">
                {models.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => setSelectedModel(model.id)}
                    className={`rounded-sm border p-3 text-left transition ${
                      selectedModel === model.id ? "border-slate-950 bg-white shadow-sm" : "border-slate-200 bg-slate-50 hover:border-slate-400"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="font-mono text-sm font-semibold text-slate-950">{model.id}</div>
                      <span className="rounded-sm border border-slate-300 bg-white px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] text-slate-600">
                        {modelKind(model)}
                      </span>
                    </div>
                    <div className="mt-2 text-xs leading-5 text-slate-600">
                      {model.capabilities?.length ? model.capabilities.join(", ") : "No capabilities advertised by registry response."}
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <EmptyBlock
                title={registry.state === "blocked" ? "Sign in required" : "No live registry data"}
                body={
                  registry.state === "blocked"
                    ? "The gateway is online, but /v1/models requires a valid API key before it will reveal model records."
                    : "The page could not load /v1/models, or the registry returned an empty list."
                }
              />
            )}
          </Panel>

          <Panel title="Selected Model" meta={selected ? "runtime response" : "none"}>
            {selected ? (
              <dl className="grid gap-3 text-sm">
                <KeyValue label="id" value={selected.id} />
                <KeyValue label="kind" value={modelKind(selected)} />
                <KeyValue label="owned_by" value={selected.owned_by ?? "not reported"} />
                <KeyValue label="capabilities" value={selected.capabilities?.length ? selected.capabilities.join(", ") : "not reported"} />
              </dl>
            ) : (
              <EmptyBlock title="No model selected" body="A selected model appears only after /v1/models returns at least one registry item." />
            )}
          </Panel>
        </div>

        <Panel title="Knowledge Graph" meta="runtime + source-known + design states">
          <KnowledgeGraph nodes={graph.nodes} edges={graph.edges} />
          <div className="mt-4 grid gap-2 text-xs text-slate-600 sm:grid-cols-3">
            <Legend label="runtime verified" state="online" />
            <Legend label="source-known route" state="source" />
            <Legend label="design contract" state="design" />
          </div>
        </Panel>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-8 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <Panel title="Context Transport" meta="contract draft, not a model call">
          <label className="block">
            <span className="mb-2 block font-mono text-[11px] uppercase tracking-[0.12em] text-slate-500">Transport question</span>
            <textarea
              value={packetQuery}
              onChange={(event) => setPacketQuery(event.target.value)}
              className="h-24 w-full resize-none rounded-sm border border-slate-300 bg-white p-3 text-sm leading-6 outline-none focus:border-slate-950"
            />
          </label>

          <div className="mt-4 rounded-sm border border-violet-200 bg-violet-50 p-3 text-sm text-violet-950">
            Local draft only. This box does not call a model or retrieve evidence. It sketches the packet we need to pass between models once the substrate endpoint is wired into this page.
          </div>

          <div className="mt-4 overflow-hidden rounded-sm border border-slate-200 bg-white">
            {CONTEXT_PACKET.map(([field, description]) => (
              <div key={field} className="grid grid-cols-[130px_1fr] gap-3 border-b border-slate-100 px-3 py-2 last:border-b-0">
                <div className="font-mono text-xs text-slate-950">{field}</div>
                <div className="text-xs leading-5 text-slate-600">{description}</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Available Context Routes" meta="from current gateway source">
          <div className="grid gap-3">
            {ROUTE_CONTRACTS.map((contract) => (
              <div key={contract.route} className="rounded-sm border border-slate-200 bg-white p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <code className="font-mono text-sm text-slate-950">{contract.route}</code>
                  <span className={`rounded-sm border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] ${
                    contract.state.includes("runtime") ? stateClasses("online") : stateClasses("source")
                  }`}>
                    {contract.state}
                  </span>
                </div>
                <div className="mt-2 text-sm leading-6 text-slate-600">{contract.purpose}</div>
              </div>
            ))}
          </div>
        </Panel>
      </section>
    </main>
  );
}

function StatusCard({
  title,
  state,
  detail,
  checkedAt,
}: {
  title: string;
  state: CheckState;
  detail: string;
  checkedAt: string | null;
}) {
  return (
    <div className="rounded-sm border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-500">{title}</div>
        <span className={`rounded-sm border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] ${stateClasses(state)}`}>{state}</span>
      </div>
      <div className="mt-3 min-h-12 text-sm leading-6 text-slate-700">{detail}</div>
      <div className="mt-3 font-mono text-[11px] text-slate-400">{checkedAt ? `checked ${checkedAt}` : "not checked"}</div>
    </div>
  );
}

function Panel({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-sm border border-slate-200 bg-white shadow-sm">
      <div className="flex min-h-12 items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
        <h2 className="font-mono text-[12px] uppercase tracking-[0.12em] text-slate-700">{title}</h2>
        {meta ? <span className="text-right font-mono text-[10px] uppercase tracking-[0.1em] text-slate-400">{meta}</span> : null}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

function EmptyBlock({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-sm border border-dashed border-slate-300 bg-slate-50 p-4">
      <div className="text-sm font-medium text-slate-900">{title}</div>
      <div className="mt-2 text-sm leading-6 text-slate-600">{body}</div>
    </div>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[112px_1fr] gap-3 border-b border-slate-100 pb-2 last:border-b-0">
      <dt className="font-mono text-xs text-slate-500">{label}</dt>
      <dd className="min-w-0 break-words text-sm text-slate-900">{value}</dd>
    </div>
  );
}

function Legend({ label, state }: { label: string; state: CheckState | "source" | "design" }) {
  return (
    <div className="flex items-center gap-2">
      <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: dotColor(state) }} />
      <span>{label}</span>
    </div>
  );
}

function KnowledgeGraph({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const byId = new Map(nodes.map((node) => [node.id, node]));

  return (
    <div className="overflow-hidden rounded-sm border border-slate-200 bg-[#fbfcff]">
      <svg viewBox="0 0 680 430" role="img" aria-label="Knowledge graph showing runtime checks, context transport, and model registry" className="h-[430px] w-full">
        <defs>
          <marker id="arrow" markerHeight="8" markerWidth="8" orient="auto" refX="7" refY="4">
            <path d="M0,0 L8,4 L0,8 Z" fill="#64748b" />
          </marker>
        </defs>

        {edges.map((edge) => {
          const from = byId.get(edge.from);
          const to = byId.get(edge.to);
          if (!from || !to) return null;
          const midX = (from.x + to.x) / 2;
          const midY = (from.y + to.y) / 2;
          return (
            <g key={`${edge.from}-${edge.to}`}>
              <line
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={dotColor(edge.state)}
                strokeDasharray={edge.state === "design" ? "5 5" : edge.state === "source" ? "2 5" : undefined}
                strokeWidth="1.5"
                markerEnd="url(#arrow)"
              />
              <rect x={midX - 42} y={midY - 10} width="84" height="20" rx="2" fill="#ffffff" stroke="#e2e8f0" />
              <text x={midX} y={midY + 4} textAnchor="middle" className="fill-slate-500 text-[9px] uppercase tracking-normal">
                {edge.label}
              </text>
            </g>
          );
        })}

        {nodes.map((node) => (
          <g key={node.id}>
            <rect x={node.x - 72} y={node.y - 23} width="144" height="46" rx="3" fill={nodeFill(node.state)} stroke={dotColor(node.state)} strokeWidth="1.5" />
            <circle cx={node.x - 57} cy={node.y - 8} r="4" fill={dotColor(node.state)} />
            <text x={node.x - 46} y={node.y - 4} className="fill-slate-950 text-[11px] font-semibold tracking-normal">
              {node.label}
            </text>
            <text x={node.x - 57} y={node.y + 13} className="fill-slate-500 text-[9px] tracking-normal">
              {node.sublabel}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
