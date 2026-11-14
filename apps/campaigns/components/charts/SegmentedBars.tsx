"use client";

// One tall segmented bar per row, each a whole split into its parts: here each metric's budget
// split across the platforms. A part's share is printed inside when it fits; hover and focus
// give every part's name and share, and the table gives them all.

import { useState } from "react";
import { percent } from "@/lib/format";
import { hasShorts, TooltipRows, useWidth } from "./parts";

export type Bar = { label: string; parts: number[] | null; error?: string };

/** Roughly how wide a share's label is, at 12px, with its padding. */
const LABEL_WIDTH = 7 * "00.0%".length + 12;
/** The row labels' column and the gap after it: 7rem and 0.75rem. */
const LABEL_COLUMN = 112 + 12;

export function SegmentedBars({
  bars,
  names,
  colors,
}: {
  bars: Bar[];
  names: string[];
  colors: string[];
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{
    bar: number;
    part: number;
    x: number;
  } | null>(null);
  return (
    <div ref={ref} className="relative space-y-3">
      {bars.map((bar, b) => (
        <div
          key={bar.label}
          className="grid grid-cols-[7rem_1fr] items-center gap-3"
        >
          <div className="text-xs text-ink-2">{bar.label}</div>
          {bar.parts && hasShorts(bar.parts) ? (
            <div className="text-xs text-ink-2">
              Short positions don&apos;t fit a bar of shares: see the table.
            </div>
          ) : bar.parts ? (
            <div
              className="flex h-7 gap-0.5"
              role="img"
              aria-label={`${bar.label}: ${bar.parts
                .map((share, i) => `${names[i]} ${percent(share)}`)
                .join(", ")}`}
            >
              {bar.parts.map((share, p) =>
                share <= 0 ? null : (
                  <div
                    key={names[p]}
                    tabIndex={0}
                    aria-label={`${names[p]}: ${percent(share)}`}
                    className="relative flex min-w-px items-center rounded px-1.5 outline-offset-2 hover:brightness-110 focus-visible:outline-2 focus-visible:outline-ink"
                    style={{
                      flexBasis: `${share * 100}%`,
                      background: colors[p],
                    }}
                    onPointerMove={(event) =>
                      setHover({
                        bar: b,
                        part: p,
                        x:
                          event.clientX -
                          (ref.current?.getBoundingClientRect().left ?? 0),
                      })
                    }
                    onPointerLeave={() => setHover(null)}
                    onFocus={(event) =>
                      setHover({
                        bar: b,
                        part: p,
                        x:
                          event.currentTarget.getBoundingClientRect().left -
                          (ref.current?.getBoundingClientRect().left ?? 0),
                      })
                    }
                    onBlur={() => setHover(null)}
                  >
                    {share * (width - LABEL_COLUMN) >= LABEL_WIDTH ? (
                      <span
                        className="tabular text-xs font-medium"
                        style={{
                          color: colors[p]?.replace(
                            "--series-",
                            "--on-series-",
                          ),
                        }}
                      >
                        {percent(share)}
                      </span>
                    ) : null}
                  </div>
                ),
              )}
            </div>
          ) : (
            <div className="text-xs text-critical">
              {bar.error ?? "No allocation"}
            </div>
          )}
          {hover?.bar === b && bar.parts && !hasShorts(bar.parts) ? (
            <div
              className="absolute z-10"
              style={{
                left: Math.min(hover.x, Math.max(0, width - 160)),
                top: b * 40 + 32,
              }}
            >
              <TooltipRows
                title={bar.label}
                rows={[
                  {
                    name: names[hover.part] ?? "",
                    color: colors[hover.part],
                    value: percent(bar.parts[hover.part] ?? 0),
                  },
                ]}
              />
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}
