"use client";

import { useState } from "react";

const KV_CACHE_URL =
  process.env.NEXT_PUBLIC_KV_CACHE_URL || "";
const SPEC_DECODING_URL =
  process.env.NEXT_PUBLIC_SPEC_DECODING_URL || "";

// ── Types ──────────────────────────────────────────────────
interface KvModeResult {
  total_tokens: number;
  elapsed_s: number;
  tokens_per_sec: number;
  peak_memory_mb: number;
  generated_text: string;
  step_times_ms: number[];
}

interface KvResult {
  benchmark: string;
  with_kv_cache: KvModeResult;
  without_kv_cache: KvModeResult;
  speedup: number;
}

interface TokenDetail {
  token: string;
  source: "draft" | "target";
}

interface SpecResult {
  benchmark: string;
  prompt: string;
  target_model: string;
  draft_model: string;
  gamma: number;
  autoregressive: {
    total_tokens: number;
    elapsed_s: number;
    tokens_per_sec: number;
    generated_text: string;
  };
  speculative: {
    total_tokens: number;
    elapsed_s: number;
    tokens_per_sec: number;
    generated_text: string;
    acceptance_rate: number;
    draft_accepted: number;
    target_resampled: number;
    token_details: TokenDetail[];
  };
  speedup: number;
}

// ── Styles (inline) ────────────────────────────────────────
const styles = {
  page: {
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    maxWidth: 1100,
    margin: "0 auto",
    padding: "2rem 1.5rem",
    color: "#1a1a2e",
    background: "#f8f9fc",
    minHeight: "100vh",
  } as React.CSSProperties,
  h1: {
    fontSize: "1.75rem",
    fontWeight: 700,
    marginBottom: "0.25rem",
  } as React.CSSProperties,
  subtitle: {
    color: "#6b7280",
    marginBottom: "2rem",
    fontSize: "0.95rem",
  } as React.CSSProperties,
  tabs: {
    display: "flex",
    gap: "0.5rem",
    marginBottom: "1.5rem",
  } as React.CSSProperties,
  tab: (active: boolean) =>
    ({
      padding: "0.6rem 1.25rem",
      border: "none",
      borderRadius: "8px",
      cursor: "pointer",
      fontWeight: 600,
      fontSize: "0.9rem",
      background: active ? "#4f46e5" : "#e5e7eb",
      color: active ? "#fff" : "#374151",
      transition: "all 0.15s",
    }) as React.CSSProperties,
  card: {
    background: "#fff",
    borderRadius: "12px",
    padding: "1.5rem",
    marginBottom: "1.25rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
    border: "1px solid #e5e7eb",
  } as React.CSSProperties,
  label: {
    display: "block",
    fontWeight: 600,
    fontSize: "0.85rem",
    color: "#374151",
    marginBottom: "0.35rem",
  } as React.CSSProperties,
  input: {
    width: "100%",
    padding: "0.5rem 0.75rem",
    border: "1px solid #d1d5db",
    borderRadius: "8px",
    fontSize: "0.9rem",
    marginBottom: "0.75rem",
    boxSizing: "border-box",
  } as React.CSSProperties,
  row: {
    display: "flex",
    gap: "1rem",
    flexWrap: "wrap",
  } as React.CSSProperties,
  col: {
    flex: 1,
    minWidth: 180,
  } as React.CSSProperties,
  btn: (loading: boolean) =>
    ({
      padding: "0.65rem 1.5rem",
      border: "none",
      borderRadius: "8px",
      background: loading ? "#9ca3af" : "#4f46e5",
      color: "#fff",
      fontWeight: 700,
      fontSize: "0.9rem",
      cursor: loading ? "not-allowed" : "pointer",
      marginTop: "0.5rem",
    }) as React.CSSProperties,
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "0.875rem",
  } as React.CSSProperties,
  th: {
    textAlign: "left",
    padding: "0.6rem 0.75rem",
    borderBottom: "2px solid #e5e7eb",
    fontWeight: 600,
    color: "#6b7280",
    fontSize: "0.8rem",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  } as React.CSSProperties,
  td: {
    padding: "0.6rem 0.75rem",
    borderBottom: "1px solid #f3f4f6",
  } as React.CSSProperties,
  metric: {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.4rem",
    padding: "0.5rem 1rem",
    borderRadius: "8px",
    background: "#f0fdf4",
    border: "1px solid #bbf7d0",
    fontWeight: 700,
    fontSize: "1.1rem",
    color: "#166534",
    marginRight: "0.75rem",
    marginBottom: "0.5rem",
  } as React.CSSProperties,
  metricLabel: {
    fontSize: "0.75rem",
    fontWeight: 500,
    color: "#6b7280",
    textTransform: "uppercase",
  } as React.CSSProperties,
  bar: (pct: number, color: string) =>
    ({
      height: 24,
      width: `${pct}%`,
      background: color,
      borderRadius: "4px",
      transition: "width 0.5s ease",
      display: "flex",
      alignItems: "center",
      paddingLeft: 6,
      color: "#fff",
      fontSize: "0.75rem",
      fontWeight: 600,
      minWidth: pct > 5 ? undefined : 28,
    }) as React.CSSProperties,
  tokenDraft: {
    background: "#dcfce7",
    color: "#166534",
    padding: "1px 3px",
    borderRadius: "3px",
    fontFamily: "monospace",
    fontSize: "0.85rem",
  } as React.CSSProperties,
  tokenTarget: {
    background: "#fee2e2",
    color: "#991b1b",
    padding: "1px 3px",
    borderRadius: "3px",
    fontFamily: "monospace",
    fontSize: "0.85rem",
  } as React.CSSProperties,
  error: {
    color: "#dc2626",
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "8px",
    padding: "0.75rem 1rem",
    marginTop: "0.75rem",
    fontSize: "0.85rem",
  } as React.CSSProperties,
  genText: {
    background: "#f9fafb",
    borderRadius: "8px",
    padding: "0.75rem 1rem",
    fontFamily: "monospace",
    fontSize: "0.825rem",
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    maxHeight: 200,
    overflowY: "auto",
    border: "1px solid #e5e7eb",
  } as React.CSSProperties,
} as const;

// ── KV Cache Panel ─────────────────────────────────────────
function KvCachePanel() {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [maxTokens, setMaxTokens] = useState(256);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<KvResult | null>(null);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(KV_CACHE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div style={styles.card}>
        <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>Configuration</h3>
        <label style={styles.label}>Prompt</label>
        <input
          style={styles.input}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div style={styles.row}>
          <div style={styles.col}>
            <label style={styles.label}>Max New Tokens</label>
            <input
              style={styles.input}
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />
          </div>
        </div>
        <button style={styles.btn(loading)} onClick={run} disabled={loading}>
          {loading ? "Running benchmark..." : "Run KV Cache Benchmark"}
        </button>
        {error && <div style={styles.error}>{error}</div>}
      </div>

      {result && (
        <>
          {/* Headline metrics */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Summary</h3>
            <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
              <div style={styles.metric}>
                <div style={{ fontSize: "2rem", fontWeight: 700, color: result.speedup >= 1 ? "#166534" : "#991b1b" }}>
                  {result.speedup}x
                </div>
                <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>Speedup</div>
              </div>
              <div style={styles.metric}>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#4f46e5" }}>
                  {result.with_kv_cache.tokens_per_sec}
                </div>
                <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>tok/s (with cache)</div>
              </div>
              <div style={styles.metric}>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#9ca3af" }}>
                  {result.without_kv_cache.tokens_per_sec}
                </div>
                <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>tok/s (without cache)</div>
              </div>
              <div style={styles.metric}>
                <div style={{ fontSize: "1.2rem", fontWeight: 600 }}>
                  {result.with_kv_cache.peak_memory_mb} / {result.without_kv_cache.peak_memory_mb} MB
                </div>
                <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>Memory (with / without)</div>
              </div>
            </div>
          </div>

          {/* Per-step latency chart */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Per-Step Latency (ms)</h3>
            <p style={{ fontSize: "0.8rem", color: "#6b7280", margin: "0 0 1rem 0" }}>
              With KV cache each step takes constant time (only processes the new token).
              Without cache, latency grows linearly as the model reprocesses all tokens each step.
            </p>
            {(() => {
              const withSteps = result.with_kv_cache.step_times_ms;
              const withoutSteps = result.without_kv_cache.step_times_ms;
              const maxLen = Math.max(withSteps.length, withoutSteps.length);
              const maxMs = Math.max(...withSteps, ...withoutSteps, 1);
              const chartH = 200;
              const chartW = Math.max(600, maxLen * 4);
              const step = chartW / Math.max(maxLen - 1, 1);

              function polyline(data: number[], color: string) {
                const points = data
                  .map((v, i) => `${i * step},${chartH - (v / maxMs) * chartH}`)
                  .join(" ");
                return (
                  <polyline
                    key={color}
                    points={points}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                  />
                );
              }

              return (
                <div style={{ overflowX: "auto" }}>
                  <svg
                    viewBox={`-40 -10 ${chartW + 60} ${chartH + 40}`}
                    width="100%"
                    style={{ maxHeight: 280 }}
                  >
                    {/* Y axis labels */}
                    <text x={-5} y={5} textAnchor="end" fontSize="10" fill="#6b7280">
                      {maxMs.toFixed(1)}
                    </text>
                    <text x={-5} y={chartH} textAnchor="end" fontSize="10" fill="#6b7280">
                      0
                    </text>
                    <text x={-5} y={chartH / 2} textAnchor="end" fontSize="10" fill="#6b7280">
                      {(maxMs / 2).toFixed(1)}
                    </text>
                    {/* Grid lines */}
                    <line x1={0} y1={0} x2={chartW} y2={0} stroke="#e5e7eb" strokeDasharray="4" />
                    <line x1={0} y1={chartH / 2} x2={chartW} y2={chartH / 2} stroke="#e5e7eb" strokeDasharray="4" />
                    <line x1={0} y1={chartH} x2={chartW} y2={chartH} stroke="#e5e7eb" />
                    {/* Data */}
                    {polyline(withoutSteps, "#ef4444")}
                    {polyline(withSteps, "#4f46e5")}
                    {/* X axis label */}
                    <text x={chartW / 2} y={chartH + 30} textAnchor="middle" fontSize="11" fill="#6b7280">
                      Generation Step
                    </text>
                  </svg>
                  <div style={{ display: "flex", gap: "1.5rem", fontSize: "0.8rem", marginTop: "0.25rem" }}>
                    <span><span style={{ color: "#4f46e5" }}>■</span> With KV Cache</span>
                    <span><span style={{ color: "#ef4444" }}>■</span> Without KV Cache</span>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Comparison table */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Detailed Comparison</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Metric</th>
                    <th style={styles.th}>With KV Cache</th>
                    <th style={styles.th}>Without KV Cache</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={styles.td}>Tokens/sec</td>
                    <td style={styles.td}><strong>{result.with_kv_cache.tokens_per_sec}</strong></td>
                    <td style={styles.td}>{result.without_kv_cache.tokens_per_sec}</td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Total time (s)</td>
                    <td style={styles.td}>{result.with_kv_cache.elapsed_s}</td>
                    <td style={styles.td}>{result.without_kv_cache.elapsed_s}</td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Tokens generated</td>
                    <td style={styles.td}>{result.with_kv_cache.total_tokens}</td>
                    <td style={styles.td}>{result.without_kv_cache.total_tokens}</td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Peak memory (MB)</td>
                    <td style={styles.td}>{result.with_kv_cache.peak_memory_mb}</td>
                    <td style={styles.td}>{result.without_kv_cache.peak_memory_mb}</td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Avg step latency (ms)</td>
                    <td style={styles.td}>
                      {(result.with_kv_cache.step_times_ms.reduce((a: number, b: number) => a + b, 0) / result.with_kv_cache.step_times_ms.length).toFixed(2)}
                    </td>
                    <td style={styles.td}>
                      {(result.without_kv_cache.step_times_ms.reduce((a: number, b: number) => a + b, 0) / result.without_kv_cache.step_times_ms.length).toFixed(2)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Sample output */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Generated Text</h3>
            <div style={styles.genText}>
              {result.with_kv_cache.generated_text}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Speculative Decoding Panel ─────────────────────────────
function SpecPanel() {
  const [prompt, setPrompt] = useState(
    "Explain the theory of relativity in simple terms."
  );
  const [maxTokens, setMaxTokens] = useState(128);
  const [gamma, setGamma] = useState(4);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SpecResult | null>(null);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(SPEC_DECODING_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          gamma,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div style={styles.card}>
        <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>Configuration</h3>
        <label style={styles.label}>Prompt</label>
        <input
          style={styles.input}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div style={styles.row}>
          <div style={styles.col}>
            <label style={styles.label}>Max New Tokens</label>
            <input
              style={styles.input}
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />
          </div>
          <div style={styles.col}>
            <label style={styles.label}>Gamma (draft tokens per step)</label>
            <input
              style={styles.input}
              type="number"
              value={gamma}
              onChange={(e) => setGamma(Number(e.target.value))}
            />
          </div>
        </div>
        <div
          style={{
            fontSize: "0.8rem",
            color: "#6b7280",
            marginBottom: "0.5rem",
          }}
        >
          Target: Qwen 2.5 7B · Draft: Qwen 2.5 0.5B
        </div>
        <button style={styles.btn(loading)} onClick={run} disabled={loading}>
          {loading
            ? "Running benchmark..."
            : "Run Speculative Decoding Benchmark"}
        </button>
        {error && <div style={styles.error}>{error}</div>}
      </div>

      {result && (
        <>
          {/* Headline metrics */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Performance</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              <div style={styles.metric}>
                <span>{result.speedup}x</span>
                <span style={styles.metricLabel}>Speedup</span>
              </div>
              <div style={styles.metric}>
                <span>{(result.speculative.acceptance_rate * 100).toFixed(1)}%</span>
                <span style={styles.metricLabel}>Acceptance Rate</span>
              </div>
              <div style={styles.metric}>
                <span>{result.speculative.tokens_per_sec}</span>
                <span style={styles.metricLabel}>Spec tok/s</span>
              </div>
              <div style={styles.metric}>
                <span>{result.autoregressive.tokens_per_sec}</span>
                <span style={styles.metricLabel}>Baseline tok/s</span>
              </div>
            </div>
          </div>

          {/* Side-by-side comparison */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Comparison</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Metric</th>
                    <th style={styles.th}>Autoregressive</th>
                    <th style={styles.th}>Speculative</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={styles.td}>Tokens/sec</td>
                    <td style={styles.td}>
                      {result.autoregressive.tokens_per_sec}
                    </td>
                    <td style={styles.td}>
                      <strong>{result.speculative.tokens_per_sec}</strong>
                    </td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Total tokens</td>
                    <td style={styles.td}>
                      {result.autoregressive.total_tokens}
                    </td>
                    <td style={styles.td}>
                      {result.speculative.total_tokens}
                    </td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Elapsed (s)</td>
                    <td style={styles.td}>
                      {result.autoregressive.elapsed_s}
                    </td>
                    <td style={styles.td}>{result.speculative.elapsed_s}</td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Draft accepted</td>
                    <td style={styles.td}>—</td>
                    <td style={styles.td}>
                      {result.speculative.draft_accepted}
                    </td>
                  </tr>
                  <tr>
                    <td style={styles.td}>Target resampled</td>
                    <td style={styles.td}>—</td>
                    <td style={styles.td}>
                      {result.speculative.target_resampled}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Acceptance rate bar */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Token Attribution</h3>
            <div style={{ marginBottom: "0.75rem" }}>
              <div
                style={{
                  display: "flex",
                  height: 32,
                  borderRadius: 6,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${result.speculative.acceptance_rate * 100}%`,
                    background: "#22c55e",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#fff",
                    fontWeight: 700,
                    fontSize: "0.8rem",
                  }}
                >
                  Draft:{" "}
                  {(result.speculative.acceptance_rate * 100).toFixed(1)}%
                </div>
                <div
                  style={{
                    width: `${(1 - result.speculative.acceptance_rate) * 100}%`,
                    background: "#ef4444",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#fff",
                    fontWeight: 700,
                    fontSize: "0.8rem",
                  }}
                >
                  Target:{" "}
                  {((1 - result.speculative.acceptance_rate) * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Token-by-token visualization */}
            <div
              style={{
                fontSize: "0.8rem",
                fontWeight: 600,
                color: "#374151",
                marginBottom: "0.5rem",
              }}
            >
              Token-by-token output{" "}
              <span style={styles.tokenDraft}>draft (accepted)</span>{" "}
              <span style={styles.tokenTarget}>target (resampled)</span>
            </div>
            <div style={styles.genText}>
              {result.speculative.token_details.map((t, i) => (
                <span
                  key={i}
                  style={
                    t.source === "draft"
                      ? styles.tokenDraft
                      : styles.tokenTarget
                  }
                >
                  {t.token}
                </span>
              ))}
            </div>
          </div>

          {/* Autoregressive output for comparison */}
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Autoregressive Output (baseline)</h3>
            <div style={styles.genText}>
              {result.autoregressive.generated_text}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────
export default function Home() {
  const [tab, setTab] = useState<"kv" | "spec">("kv");

  return (
    <div style={styles.page}>
      <h1 style={styles.h1}>Inference Benchmarks</h1>
      <p style={styles.subtitle}>
        Compare KV cache performance and speculative decoding on GPU
      </p>

      <div style={styles.tabs}>
        <button style={styles.tab(tab === "kv")} onClick={() => setTab("kv")}>
          KV Cache
        </button>
        <button
          style={styles.tab(tab === "spec")}
          onClick={() => setTab("spec")}
        >
          Speculative Decoding
        </button>
      </div>

      {tab === "kv" ? <KvCachePanel /> : <SpecPanel />}
    </div>
  );
}
