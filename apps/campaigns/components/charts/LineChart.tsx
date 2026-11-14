"use client";

// Lines on one axis. Over time, the series share their x values: a crosshair snaps to the
// nearest and the tooltip lists every series there. Otherwise, as for a frontier, each series
// has its own points, and the tooltip gives the point nearest the pointer. With two to four
// series, each is labeled at its end unless the labels would collide; the legend always is.

import { useState } from "react";
import { extent, linear, nearest, nice } from "@/lib/scale";
import { tick } from "@/lib/format";
import { TooltipRows, useWidth } from "./parts";

export type Point = { x: number; y: number | null };
export type Series = { name: string; color: string; points: Point[] };
export type Marker = { x: number; y: number; color: string; label: string };

const MARGIN = { top: 10, right: 16, bottom: 26, left: 48 };
const LABEL_ROOM = 84;

export function LineChart({
  series,
  shared,
  height = 240,
  xLabel,
  yLabel,
  formatX,
  formatY,
  formatPoint = formatX,
  xTicks,
  markers = [],
}: {
  series: Series[];
  /** Whether the series share their x values, as over time. */
  shared: boolean;
  height?: number;
  xLabel: string;
  yLabel: string;
  formatX: (x: number) => string;
  formatY: (y: number) => string;
  /** An x value as a tooltip names it, if not as the axis does. */
  formatPoint?: (x: number) => string;
  /** The x values to tick, when they're not evenly spread numbers, such as dates. */
  xTicks?: number[];
  markers?: Marker[];
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ px: number; py: number } | null>(null);

  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = [
    ...series.flatMap((s) => s.points.map((p) => p.y)),
    ...markers.map((m) => m.y),
  ];
  const xRange = extent([...xs, ...markers.map((m) => m.x)]);
  const yRange = extent(ys);
  // A single series is named by the chart's title, so only two to four get end labels.
  const labeled = series.length >= 2 && series.length <= 4;
  const right = MARGIN.right + (labeled ? LABEL_ROOM : 0);
  const plotWidth = Math.max(0, width - MARGIN.left - right);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;

  if (!xRange || !yRange || width === 0) {
    return <div ref={ref} style={{ height }} />;
  }

  const yAxis = nice(yRange[0], yRange[1]);
  const xAxis = shared
    ? { domain: xRange, ticks: xTicks ?? [] }
    : nice(xRange[0], xRange[1]);
  const x = linear(xAxis.domain[0], xAxis.domain[1], 0, plotWidth);
  const y = linear(yAxis.domain[0], yAxis.domain[1], plotHeight, 0);

  const path = (points: Point[]) => {
    let d = "";
    let pen = false;
    for (const p of points) {
      if (p.y === null) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  // The end labels, dropped together if any two would overlap.
  const ends = series
    .map((s) => {
      const last = [...s.points].reverse().find((p) => p.y !== null);
      return last ? { name: s.name, at: y(last.y!), x: x(last.x) } : null;
    })
    .filter((end) => end !== null);
  const sorted = [...ends].sort((a, b) => a.at - b.at);
  const fits = sorted.every(
    (end, i) => i === 0 || end.at - sorted[i - 1]!.at >= 13,
  );

  // What the pointer is over.
  let readout: {
    left: number;
    top: number;
    title: string;
    rows: Parameters<typeof TooltipRows>[0]["rows"];
  } | null = null;
  let crosshair: number | null = null;
  if (hover) {
    if (shared) {
      const grid = series[0]?.points.map((p) => p.x) ?? [];
      const i = nearest(
        grid.map((value) => x(value)),
        hover.px,
      );
      const at = grid[i];
      if (at !== undefined) {
        crosshair = x(at);
        readout = {
          left: crosshair,
          top: hover.py,
          title: formatPoint(at),
          rows: series.map((s) => {
            const value = s.points[i]?.y ?? null;
            return {
              name: s.name,
              color: s.color,
              value: value === null ? "–" : formatY(value),
            };
          }),
        };
      }
    } else {
      let best: { s: Series | Marker; p: Point; distance: number } | null =
        null;
      const candidates: [Series | Marker, Point][] = [
        ...series.flatMap((s) =>
          s.points.map((p) => [s, p] as [Series, Point]),
        ),
        ...markers.map((m) => [m, { x: m.x, y: m.y }] as [Marker, Point]),
      ];
      for (const [s, p] of candidates) {
        if (p.y === null) continue;
        const distance = Math.hypot(x(p.x) - hover.px, y(p.y) - hover.py);
        if (!best || distance < best.distance) best = { s, p, distance };
      }
      if (best && best.distance <= 24 && best.p.y !== null) {
        const name = "name" in best.s ? best.s.name : best.s.label;
        readout = {
          left: x(best.p.x),
          top: y(best.p.y),
          title: name,
          rows: [
            { name: xLabel, value: formatPoint(best.p.x) },
            { name: yLabel, value: formatY(best.p.y) },
          ],
        };
      }
    }
  }

  return (
    <div ref={ref} className="relative">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${yLabel} by ${xLabel}: ${series.map((s) => s.name).join(", ")}`}
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
          {yAxis.ticks.map((value) => (
            <g key={value}>
              <line
                x1={0}
                x2={plotWidth}
                y1={y(value)}
                y2={y(value)}
                stroke="var(--grid)"
              />
              <text
                x={-8}
                y={y(value)}
                dy="0.32em"
                textAnchor="end"
                className="tabular fill-muted text-[11px]"
              >
                {tick(value)}
              </text>
            </g>
          ))}
          <line
            x1={0}
            x2={plotWidth}
            y1={plotHeight}
            y2={plotHeight}
            stroke="var(--axis)"
          />
          {xAxis.ticks.map((value) => (
            <text
              key={value}
              x={x(value)}
              y={plotHeight + 18}
              textAnchor="middle"
              className="tabular fill-muted text-[11px]"
            >
              {formatX(value)}
            </text>
          ))}
          {crosshair !== null ? (
            <line
              x1={crosshair}
              x2={crosshair}
              y1={0}
              y2={plotHeight}
              stroke="var(--axis)"
            />
          ) : null}
          {series.map((s) => (
            <path
              key={s.name}
              d={path(s.points)}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {markers.map((m) => (
            <circle
              key={m.label}
              cx={x(m.x)}
              cy={y(m.y)}
              r={5}
              fill={m.color}
              stroke="var(--surface)"
              strokeWidth={2}
            />
          ))}
          {labeled && fits
            ? ends.map((end) => (
                <text
                  key={end.name}
                  x={end.x + 8}
                  y={end.at}
                  dy="0.32em"
                  className="fill-ink-2 text-[11px]"
                >
                  {end.name}
                </text>
              ))
            : null}
        </g>
      </svg>
      {readout ? (
        <div
          className="absolute z-10"
          style={{
            left: Math.min(
              MARGIN.left + readout.left + 12,
              Math.max(0, width - 180),
            ),
            top: Math.max(0, MARGIN.top + readout.top - 20),
          }}
        >
          <TooltipRows title={readout.title} rows={readout.rows} />
        </div>
      ) : null}
    </div>
  );
}
