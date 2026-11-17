// The form's settings, and the request they make. The form covers the common settings; the
// request's JSON can be edited for any of cutup's others.

import type { Method } from "./bindings/Method";
import type { Metric } from "./bindings/Metric";
import type { Request } from "./bindings/Request";

export type MethodName = Method["name"];

export const METRICS: { metric: Metric; label: string }[] = [
  { metric: "impressions", label: "Impressions" },
  { metric: "clicks", label: "Clicks" },
  { metric: "link_clicks", label: "Link clicks" },
  { metric: "views", label: "Views" },
];

export const METHODS: { name: MethodName; label: string }[] = [
  { name: "equal_weight", label: "Equal weight" },
  { name: "inverse_volatility", label: "Inverse volatility" },
  { name: "inverse_variance", label: "Inverse variance" },
  { name: "min_variance", label: "Minimum variance" },
  { name: "mean_variance", label: "Mean-variance" },
  { name: "max_sharpe", label: "Maximum Sharpe ratio" },
  { name: "max_diversification", label: "Maximum diversification" },
  { name: "risk_parity", label: "Risk parity" },
  { name: "hrp", label: "Hierarchical risk parity" },
  { name: "herc", label: "Hierarchical equal risk contribution" },
  { name: "min_cvar", label: "Minimum CVaR" },
  { name: "min_mad", label: "Minimum mean absolute deviation" },
  { name: "min_cdar", label: "Minimum CDaR (drawdowns)" },
];

/** The methods that take weight bounds, so a cap on each platform's share. */
export const BOUNDED: MethodName[] = [
  "min_variance",
  "mean_variance",
  "min_cvar",
  "min_mad",
  "min_cdar",
];

/** The methods that take a confidence, the share of days or drawdowns their tail leaves out. */
export const CONFIDENT: MethodName[] = ["min_cvar", "min_cdar"];

export const DEFAULT_PLATFORMS = ["Meta", "Google", "TikTok"];

/** The lowest cap on each platform's share the form offers for `count` platforms: the
 * smallest 5% step from which the shares can still add up to the whole, and never under 20%. */
export function lowestCap(count: number): number {
  return Math.max(0.2, Math.ceil(20 / Math.max(1, count) - 1e-9) / 20);
}

export type Form = {
  platforms: string[];
  from: string;
  to: string;
  metrics: Metric[];
  method: MethodName;
  /** The most any platform can take, from 0 to 1, or null for no cap. */
  maxWeight: number | null;
  riskAversion: number;
  /** Points along the efficient frontier, for mean-variance, or null for none. */
  frontier: number | null;
  /** The confidence of min CVaR's and min CDaR's tails, from 0 to 1. */
  confidence: number;
  /** The return min MAD's shortfall is measured below, or null for the deviation either side
   * of the mean. Returns are excesses over the average yield, so 0 is the average. */
  downside: number | null;
  rolling: { window: number; step: number } | null;
};

export const DEFAULT_FORM: Form = {
  platforms: DEFAULT_PLATFORMS,
  from: "",
  to: "",
  metrics: METRICS.map(({ metric }) => metric),
  method: "min_variance",
  maxWeight: null,
  riskAversion: 1,
  frontier: null,
  confidence: 0.95,
  downside: null,
  rolling: null,
};

/** The request a form makes. Settings a method doesn't take are left out, so the request
 * holds only what it uses. */
export function toRequest(form: Form): Request {
  const method = { name: form.method } as Record<string, unknown>;
  if (BOUNDED.includes(form.method) && form.maxWeight !== null) {
    method.bounds = { upper: form.maxWeight };
  }
  if (form.method === "mean_variance") {
    method.risk_aversion = form.riskAversion;
  }
  if (CONFIDENT.includes(form.method)) {
    method.confidence = form.confidence;
  }
  if (form.method === "min_mad" && form.downside !== null) {
    method.downside = form.downside;
  }
  const request: Request = {
    platforms: form.platforms,
    metrics: form.metrics,
    method: method as Method,
  };
  if (form.from) request.from = form.from;
  if (form.to) request.to = form.to;
  if (form.method === "mean_variance" && form.frontier !== null) {
    request.frontier = form.frontier;
  }
  if (form.rolling) request.rolling = { ...form.rolling };
  return request;
}

/** A request's JSON as the editor shows it. */
export function requestText(request: Request): string {
  return JSON.stringify(request, null, 2);
}
