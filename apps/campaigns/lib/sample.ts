// The sample's weekly yields, for the chart of what each platform delivered over time. Read on
// the server when the page builds; the engines read the same CSV for themselves.

import type { Metric } from "./bindings/Metric";

/** Each platform's yield of each metric, per unit of spend, by week, from Monday. A week a
 * platform didn't spend in is null. */
export type WeeklyYields = {
  weeks: string[];
  platforms: string[];
  yields: Record<Metric, Record<string, (number | null)[]>>;
};

const METRICS: Metric[] = ["impressions", "clicks", "link_clicks", "views"];

/** The Monday on or before an ISO date, as an ISO date. */
export function weekOf(date: string): string {
  const day = new Date(`${date}T00:00:00Z`);
  const sinceMonday = (day.getUTCDay() + 6) % 7;
  day.setUTCDate(day.getUTCDate() - sinceMonday);
  return day.toISOString().slice(0, 10);
}

/** Weekly yields from the sample's CSV, `date,platform,spend,impressions,clicks,link_clicks,
 * views`: each week's total of a metric over its total spend, so a big day weighs more than a
 * small one. */
export function weeklyYields(csv: string): WeeklyYields {
  const totals = new Map<string, Map<string, number[]>>();
  const platforms = new Set<string>();
  for (const line of csv.trim().split("\n").slice(1)) {
    const [date, platform, ...values] = line.split(",");
    if (!date || !platform) continue;
    platforms.add(platform);
    const week = weekOf(date);
    const byPlatform = totals.get(week) ?? new Map<string, number[]>();
    totals.set(week, byPlatform);
    const sums = byPlatform.get(platform) ?? [0, 0, 0, 0, 0];
    values.forEach((value, i) => (sums[i] = (sums[i] ?? 0) + Number(value)));
    byPlatform.set(platform, sums);
  }
  const weeks = [...totals.keys()].sort();
  const names = [...platforms].sort();
  const yields = Object.fromEntries(
    METRICS.map((metric, m) => [
      metric,
      Object.fromEntries(
        names.map((platform) => [
          platform,
          weeks.map((week) => {
            const sums = totals.get(week)?.get(platform);
            const spend = sums?.[0] ?? 0;
            return sums && spend > 0 ? (sums[m + 1] ?? 0) / spend : null;
          }),
        ]),
      ),
    ]),
  ) as WeeklyYields["yields"];
  return { weeks, platforms: names, yields };
}
