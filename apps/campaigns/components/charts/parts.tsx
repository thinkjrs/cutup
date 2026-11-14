"use client";

// What every chart shares: its card, with a chart or table view; the legend; the tooltip; and
// the width it has to draw in.

import { useEffect, useRef, useState, type ReactNode } from "react";

/** An element's width, kept current as it resizes. */
export function useWidth<T extends HTMLElement>(): [
  React.RefObject<T | null>,
  number,
] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/** A chart with its title, legend, and a switch to a table of the same values. */
export function ChartCard({
  title,
  note,
  legend,
  chart,
  table,
}: {
  title: string;
  note?: ReactNode;
  legend?: ReactNode;
  chart: ReactNode;
  table: ReactNode;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <section className="rounded-lg border border-border bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <div
          className="flex text-xs"
          role="group"
          aria-label={`${title}: view`}
        >
          {(["chart", "table"] as const).map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={view === option}
              onClick={() => setView(option)}
              className={`rounded px-2 py-0.5 capitalize ${
                view === option
                  ? "bg-grid font-medium text-ink"
                  : "text-ink-2 hover:bg-grid/60"
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      </div>
      {note ? <p className="mt-1 text-xs text-ink-2">{note}</p> : null}
      {legend ? <div className="mt-3">{legend}</div> : null}
      <div className="mt-3">{view === "chart" ? chart : table}</div>
    </section>
  );
}

/** The platforms' keys, a swatch for areas and bars or a short line for lines. */
export function Legend({
  items,
  mark,
}: {
  items: { name: string; color: string }[];
  mark: "swatch" | "line";
}) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map(({ name, color }) => (
        <li key={name} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className={
              mark === "swatch"
                ? "size-2.5 rounded-sm"
                : "h-0.5 w-3 rounded-full"
            }
            style={{ background: color }}
          />
          {name}
        </li>
      ))}
    </ul>
  );
}

/** A tooltip's rows: the value first and strong, then the name, keyed by a short line. */
export function TooltipRows({
  title,
  rows,
}: {
  title?: string;
  rows: { name: string; color?: string; value: string }[];
}) {
  return (
    <div className="pointer-events-none rounded-md border border-border bg-surface px-2.5 py-2 text-xs whitespace-nowrap shadow-md">
      {title ? <div className="mb-1 text-ink-2">{title}</div> : null}
      <table>
        <tbody>
          {rows.map(({ name, color, value }) => (
            <tr key={name}>
              <td className="pr-1.5">
                {color ? (
                  <span
                    aria-hidden
                    className="block h-0.5 w-3 rounded-full"
                    style={{ background: color }}
                  />
                ) : null}
              </td>
              <td className="tabular pr-2 text-right font-semibold text-ink">
                {value}
              </td>
              <td className="text-ink-2">{name}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A plain table of a chart's values. */
export function DataTable({
  head,
  rows,
}: {
  head: string[];
  rows: (string | number)[][];
}) {
  return (
    <div className="max-h-80 overflow-auto">
      <table className="tabular w-full text-xs">
        <thead className="sticky top-0 bg-surface">
          <tr>
            {head.map((cell, i) => (
              <th
                key={i}
                className={`border-b border-grid px-2 py-1 font-medium text-ink-2 ${
                  i === 0 ? "text-left" : "text-right"
                }`}
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, i) => (
                <td
                  key={i}
                  className={`border-b border-grid/60 px-2 py-1 ${
                    i === 0 ? "text-left text-ink-2" : "text-right"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
