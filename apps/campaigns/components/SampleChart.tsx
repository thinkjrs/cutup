"use client";

// What each platform delivered per unit of spend, week by week: the yields the engine's
// returns come from, before they're divided by their average.

import { useState } from "react";
import type { Metric } from "@/lib/bindings/Metric";
import {
  metricLabel,
  number,
  platformColor,
  platformOrder,
} from "@/lib/format";
import { METRICS } from "@/lib/request";
import type { WeeklyYields } from "@/lib/sample";
import { LineChart } from "./charts/LineChart";
import { ChartCard, DataTable, Legend } from "./charts/parts";

const month = new Intl.DateTimeFormat("en-US", {
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

export function SampleChart({ sample }: { sample: WeeklyYields }) {
  const [metric, setMetric] = useState<Metric>("clicks");
  const { weeks } = sample;
  const platforms = platformOrder(sample.platforms);
  const yields = sample.yields[metric];
  const colors = platforms.map((name) => platformColor(name, platforms));
  // A tick on the first week of every other month.
  const ticks = weeks
    .map((week, i) => ({ week, i }))
    .filter(
      ({ week, i }) =>
        i === 0 || week.slice(0, 7) !== weeks[i - 1]!.slice(0, 7),
    )
    .filter((_, n) => n % 2 === 0)
    .map(({ i }) => i);

  return (
    <ChartCard
      title={`${metricLabel(metric)} per unit of spend, by week`}
      note={
        <span className="flex flex-wrap items-center gap-1">
          {METRICS.map(({ metric: option, label }) => (
            <button
              key={option}
              type="button"
              aria-pressed={metric === option}
              onClick={() => setMetric(option)}
              className={`rounded-full border px-2 py-0.5 ${
                metric === option
                  ? "border-ink bg-ink text-surface"
                  : "border-border text-ink-2 hover:bg-grid/60"
              }`}
            >
              {label}
            </button>
          ))}
        </span>
      }
      legend={
        <Legend
          items={platforms.map((name, i) => ({ name, color: colors[i]! }))}
          mark="line"
        />
      }
      chart={
        <LineChart
          shared
          xLabel="Week"
          yLabel={`${metricLabel(metric)} per unit`}
          formatX={(i) =>
            month.format(new Date(`${weeks[i] ?? weeks[0]}T00:00:00Z`))
          }
          formatY={(v) => number(v, 3)}
          formatPoint={(i) => `Week of ${weeks[i] ?? ""}`}
          xTicks={ticks}
          series={platforms.map((name, p) => ({
            name,
            color: colors[p]!,
            points: (yields[name] ?? []).map((y, i) => ({ x: i, y })),
          }))}
        />
      }
      table={
        <DataTable
          head={["Week of", ...platforms]}
          rows={weeks.map((week, i) => [
            week,
            ...platforms.map((name) => number(yields[name]?.[i] ?? null, 4)),
          ])}
        />
      }
    />
  );
}
