// Numbers and names as the page shows them.

import type { Metric } from "./bindings/Metric";
import { DEFAULT_PLATFORMS, METRICS } from "./request";

export function metricLabel(metric: Metric): string {
  return METRICS.find((m) => m.metric === metric)?.label ?? metric;
}

/** A share, such as a weight: 0.4567 is 45.7%. A solver's -1e-17 is 0.0%, not -0.0%. */
export function percent(value: number, digits = 1): string {
  const text = (value * 100).toFixed(digits);
  return `${Number(text) === 0 ? text.replace("-", "") : text}%`;
}

/** A plain number, to `digits` significant figures but no finer than 1e-10, with thousands
 * separated. A solver's rounding, such as an expected return of -5e-16 for one that's 0, is 0. */
export function number(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return "–";
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: digits,
    maximumFractionDigits: 10,
    roundingPriority: "lessPrecision",
    signDisplay: "negative",
  }).format(value);
}

/** Milliseconds: under 10 to a tenth, otherwise whole. */
export function milliseconds(value: number): string {
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

/** An axis tick: in full under 10,000, compact from there. */
export function tick(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: Math.abs(value) >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(value);
}

/** The sample's platforms in the order they take colors: the default three first, since the
 * first three colors are the ones that stay apart in every pair, then the rest by name. */
export function platformOrder(all: string[]): string[] {
  const defaults = DEFAULT_PLATFORMS.filter((name) => all.includes(name));
  const rest = all.filter((name) => !defaults.includes(name)).sort();
  return [...defaults, ...rest];
}

/** The platform's color, by its place among the sample's six platforms, so it keeps its color
 * whichever platforms a request picks. */
export function platformColor(platform: string, all: string[]): string {
  const slot = platformOrder(all).indexOf(platform);
  return slot >= 0 && slot < 6 ? `var(--series-${slot + 1})` : "var(--muted)";
}
