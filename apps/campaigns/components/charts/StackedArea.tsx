"use client";

// Shares of a whole over time, stacked to 100%: here each rolling window's weights, by the
// last day it covers. Bands are split by a 2px gap in the surface color. A window the method
// couldn't allocate over leaves a gap. A crosshair snaps to the nearest window and the tooltip
// lists every platform's share there.

import { useState } from "react";
import { percent } from "@/lib/format";
import { linear, nearest } from "@/lib/scale";
import { hasShorts, TooltipRows, useWidth } from "./parts";

const MARGIN = { top: 6, right: 12, bottom: 24, left: 40 };

export function StackedArea({
  labels,
  values,
  names,
  colors,
  height = 180,
}: {
  /** Each point's label, such as a window's last day. */
  labels: string[];
  /** Each point's shares, in `names` order, or null where there are none. */
  values: (number[] | null)[];
  names: string[];
  colors: string[];
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ px: number; py: number } | null>(null);
  const plotWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  if (values.some((value) => value && hasShorts(value))) {
    return (
      <div ref={ref} className="text-xs text-ink-2">
        Short positions don&apos;t stack to a whole: see the table.
      </div>
    );
  }
  if (width === 0 || values.length === 0)
    return <div ref={ref} style={{ height }} />;

  const x = linear(0, Math.max(1, values.length - 1), 0, plotWidth);
  const y = linear(0, 1, plotHeight, 0);

  // Runs of windows with weights, each drawn as its own stack.
  const runs: number[][] = [];
  values.forEach((value, i) => {
    if (!value) return;
    const run = runs[runs.length - 1];
    if (run && run[run.length - 1] === i - 1) run.push(i);
    else runs.push([i]);
  });

  const band = (run: number[], k: number) => {
    const below = (i: number) =>
      (values[i] ?? []).slice(0, k).reduce((a, b) => a + b, 0);
    const above = (i: number) => below(i) + (values[i]?.[k] ?? 0);
    // A run of one window is drawn a few pixels wide, so it shows.
    const xs = (i: number) =>
      run.length === 1 ? [x(i) - 2, x(i) + 2] : [x(i)];
    const top = run.flatMap((i) =>
      xs(i).map((px) => `${px.toFixed(1)},${y(above(i)).toFixed(1)}`),
    );
    const bottom = [...run]
      .reverse()
      .flatMap((i) =>
        [...xs(i)]
          .reverse()
          .map((px) => `${px.toFixed(1)},${y(below(i)).toFixed(1)}`),
      );
    return `M${top.join("L")}L${bottom.join("L")}Z`;
  };

  const ticks = [
    0,
    Math.floor((labels.length - 1) / 2),
    labels.length - 1,
  ].filter((value, i, all) => all.indexOf(value) === i);
  const at = hover
    ? nearest(
        values.map((_, i) => x(i)),
        hover.px,
      )
    : null;

  return (
    <div ref={ref} className="relative">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Weights over ${values.length} rolling windows`}
        onPointerMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          setHover({
            px: event.clientX - box.left - MARGIN.left,
            py: event.clientY - box.top - MARGIN.top,
          });
        }}
        onPointerLeave={() => setHover(null)}
        className="touch-none select-none"
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {[0, 0.5, 1].map((value) => (
            <text
              key={value}
              x={-8}
              y={y(value)}
              dy="0.32em"
              textAnchor="end"
              className="tabular fill-muted text-[11px]"
            >
              {percent(value, 0)}
            </text>
          ))}
          {runs.flatMap((run, r) =>
            names.map((name, k) => (
              <path
                key={`${r}-${name}`}
                d={band(run, k)}
                fill={colors[k]}
                stroke="var(--surface)"
                strokeWidth={2}
                strokeLinejoin="round"
              />
            )),
          )}
          <line
            x1={0}
            x2={plotWidth}
            y1={plotHeight}
            y2={plotHeight}
            stroke="var(--axis)"
          />
          {ticks.map((i) => (
            <text
              key={i}
              x={x(i)}
              y={plotHeight + 17}
              textAnchor={
                i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle"
              }
              className="tabular fill-muted text-[11px]"
            >
              {labels[i]}
            </text>
          ))}
          {at !== null ? (
            <line
              x1={x(at)}
              x2={x(at)}
              y1={0}
              y2={plotHeight}
              stroke="var(--ink)"
              strokeOpacity={0.5}
            />
          ) : null}
        </g>
      </svg>
      {at !== null && hover ? (
        <div
          className="absolute z-10"
          style={{
            left: Math.min(MARGIN.left + x(at) + 12, Math.max(0, width - 170)),
            top: Math.max(0, hover.py - 10),
          }}
        >
          <TooltipRows
            title={`Window ending ${labels[at]}`}
            rows={
              values[at]
                ? names.map((name, k) => ({
                    name,
                    color: colors[k],
                    value: percent(values[at]![k] ?? 0),
                  }))
                : [{ name: "the method couldn't allocate", value: "–" }]
            }
          />
        </div>
      ) : null}
    </div>
  );
}
