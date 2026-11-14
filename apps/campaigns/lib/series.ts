// A response turned into what the charts, tables and exports show.

import type { MetricResult } from "./bindings/MetricResult";
import type { Request } from "./bindings/Request";
import type { Response } from "./bindings/Response";
import type { EngineName } from "./engine";

/** Each platform's weight, in the response's platform order, 0 for one left out. */
export function weightsOf(
  response: Response,
  result: MetricResult,
): number[] | null {
  const weights = result.weights;
  return weights
    ? response.platforms.map((platform) => weights[platform] ?? 0)
    : null;
}

/** One rolling window: the last day it covers, and each platform's weight in the response's
 * platform order, or why the method couldn't allocate over it. */
export type WindowPoint = {
  day: string;
  first: string;
  weights: number[] | null;
  error?: unknown;
};

/** A result's rolling windows, by the last day each covers. */
export function windowsOf(
  response: Response,
  result: MetricResult,
): WindowPoint[] {
  return (result.rolling ?? []).map((window) => {
    const weights = window.weights;
    return {
      day: response.days[window.end - 1] ?? "",
      first: response.days[window.start] ?? "",
      weights: weights
        ? response.platforms.map((platform) => weights[platform] ?? 0)
        : null,
      ...(window.error === undefined ? {} : { error: window.error }),
    };
  });
}

/** An error from either engine or cutup, in words: its kind, then any details. */
export function describeError(error: unknown): string {
  if (typeof error === "string") return humanize(error);
  if (error && typeof error === "object") {
    const [kind, details] = Object.entries(error)[0] ?? ["error", null];
    if (details && typeof details === "object") {
      const parts = Object.entries(details).map(([key, value]) =>
        key === "message" && typeof value === "string"
          ? value
          : key === "error"
            ? describeError(value)
            : `${key.replaceAll("_", " ")} ${JSON.stringify(value)}`,
      );
      return `${humanize(kind)}: ${parts.join(", ")}`;
    }
    return humanize(kind);
  }
  return String(error);
}

function humanize(name: string): string {
  const words = name.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Everything a run gave, for the JSON export. */
export type Run = {
  engine: EngineName;
  request: Request | string;
  status: number;
  milliseconds: { total: number; engine: number | null };
  response: Response | null;
  error: unknown;
};

export function runJson(run: Run): string {
  return JSON.stringify(run, null, 2);
}

/** Every allocation in a response as CSV, one row per metric and period: the whole range, then
 * each rolling window, with each platform's weight. An allocation that failed has empty
 * weights. */
export function weightsCsv(response: Response): string {
  const header = [
    "metric",
    "window",
    "first_day",
    "last_day",
    ...response.platforms,
  ];
  const rows = [header];
  const cells = (weights: number[] | null) =>
    weights ? weights.map(String) : response.platforms.map(() => "");
  for (const result of response.results) {
    rows.push([
      result.metric,
      "all",
      response.days[0] ?? "",
      response.days[response.days.length - 1] ?? "",
      ...cells(weightsOf(response, result)),
    ]);
    windowsOf(response, result).forEach((window, i) => {
      rows.push([
        result.metric,
        String(i),
        window.first,
        window.day,
        ...cells(window.weights),
      ]);
    });
  }
  return rows.map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}

function csvCell(cell: string): string {
  return /[",\n]/.test(cell) ? `"${cell.replaceAll('"', '""')}"` : cell;
}
