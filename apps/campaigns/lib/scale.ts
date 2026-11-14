// Axis scales and their ticks.

/** A round step that splits [low, high] into about `count` parts. */
function niceStep(low: number, high: number, count: number): number {
  const rough = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  return ([1, 2, 2.5, 5, 10].find((s) => s * power >= rough) ?? 10) * power;
}

/** [low, high] widened to round values, with round ticks from end to end. A single value gets
 * room around it. */
export function nice(
  low: number,
  high: number,
  count = 5,
): { domain: [number, number]; ticks: number[] } {
  if (!(high > low)) {
    const pad = Math.abs(low) * 0.1 || 1;
    return nice(low - pad, high + pad, count);
  }
  const step = niceStep(low, high, count);
  const start = Math.floor(low / step);
  const end = Math.ceil(high / step);
  const ticks: number[] = [];
  for (let i = start; i <= end; i++)
    ticks.push(Number((i * step).toPrecision(12)));
  return { domain: [ticks[0]!, ticks[ticks.length - 1]!], ticks };
}

/** A linear map from [d0, d1] onto [r0, r1]. A domain of one value maps to the range's middle. */
export function linear(
  d0: number,
  d1: number,
  r0: number,
  r1: number,
): (value: number) => number {
  if (d1 === d0) return () => (r0 + r1) / 2;
  return (value) => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

/** The smallest and largest of the finite values, or null if there are none. */
export function extent(
  values: Iterable<number | null | undefined>,
): [number, number] | null {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (value === null || value === undefined || !Number.isFinite(value))
      continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  return low <= high ? [low, high] : null;
}

/** The index of the value in sorted `xs` nearest to `x`. */
export function nearest(xs: number[], x: number): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) {
    if (Math.abs(xs[i]! - x) < Math.abs(xs[best]! - x)) best = i;
  }
  return best;
}
