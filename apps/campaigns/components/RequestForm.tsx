"use client";

// The request: platforms, dates, metrics and a method with its common settings, and the JSON
// they make, which can be edited for any of cutup's other settings.

import type { PlatformSummary } from "@/lib/bindings/PlatformSummary";
import { metricLabel, platformColor } from "@/lib/format";
import {
  BOUNDED,
  lowestCap,
  METHODS,
  METRICS,
  type Form,
  type MethodName,
} from "@/lib/request";

const field = "rounded-md border border-border bg-surface px-2 py-1 text-sm";

export function RequestForm({
  platforms,
  form,
  onForm,
  text,
  edited,
  onText,
  onReset,
  onRun,
  running,
}: {
  platforms: PlatformSummary[];
  form: Form;
  onForm: (form: Form) => void;
  text: string;
  edited: boolean;
  onText: (text: string) => void;
  onReset: () => void;
  onRun: () => void;
  running: boolean;
}) {
  const names = platforms.map((p) => p.platform);
  const first = platforms.map((p) => p.first).sort()[0];
  const last = platforms
    .map((p) => p.last)
    .sort()
    .at(-1);
  const toggle = <T,>(list: T[], item: T, order: T[]) =>
    list.includes(item)
      ? list.filter((other) => other !== item)
      : order.filter((other) => other === item || list.includes(other));

  return (
    <form
      className="space-y-5"
      onSubmit={(event) => {
        event.preventDefault();
        onRun();
      }}
    >
      <fieldset disabled={edited} className="space-y-5 disabled:opacity-50">
        <div>
          <div className="text-xs font-medium text-ink-2">Platforms</div>
          <div className="mt-1.5 grid grid-cols-2 gap-1">
            {platforms.map(({ platform, days_with_spend }) => (
              <label key={platform} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={form.platforms.includes(platform)}
                  onChange={() => {
                    const next = toggle(form.platforms, platform, names);
                    // A cap the new count can't meet is raised to the lowest it can.
                    const cap = form.maxWeight;
                    const lowest = lowestCap(next.length);
                    onForm({
                      ...form,
                      platforms: next,
                      maxWeight:
                        cap === null || cap >= lowest
                          ? cap
                          : lowest >= 1
                            ? null
                            : lowest,
                    });
                  }}
                />
                <span
                  aria-hidden
                  className="size-2.5 rounded-sm"
                  style={{ background: platformColor(platform, names) }}
                />
                {platform}
                <span className="tabular text-xs text-muted">
                  {days_with_spend}d
                </span>
              </label>
            ))}
          </div>
          <p className="mt-1 text-xs text-muted">
            Only days on which every chosen platform spent count.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {(["from", "to"] as const).map((end) => (
            <label
              key={end}
              className="text-xs font-medium text-ink-2 capitalize"
            >
              {end}
              <input
                type="date"
                min={first}
                max={last}
                value={form[end]}
                onChange={(event) =>
                  onForm({ ...form, [end]: event.target.value })
                }
                className={`${field} mt-1 block w-full`}
              />
            </label>
          ))}
        </div>

        <div>
          <div className="text-xs font-medium text-ink-2">Metrics</div>
          <div className="mt-1.5 grid grid-cols-2 gap-1">
            {METRICS.map(({ metric }) => (
              <label key={metric} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={form.metrics.includes(metric)}
                  onChange={() =>
                    onForm({
                      ...form,
                      metrics: toggle(
                        form.metrics,
                        metric,
                        METRICS.map((m) => m.metric),
                      ),
                    })
                  }
                />
                {metricLabel(metric)}
              </label>
            ))}
          </div>
        </div>

        <label className="block text-xs font-medium text-ink-2">
          Method
          <select
            value={form.method}
            onChange={(event) =>
              onForm({ ...form, method: event.target.value as MethodName })
            }
            className={`${field} mt-1 block w-full`}
          >
            {METHODS.map(({ name, label }) => (
              <option key={name} value={name}>
                {label}
              </option>
            ))}
          </select>
        </label>

        {BOUNDED.includes(form.method) ? (
          <label className="block text-xs font-medium text-ink-2">
            Most any platform takes
            <div className="mt-1 flex items-center gap-2">
              <input
                type="range"
                min={lowestCap(form.platforms.length)}
                max={1}
                step={0.05}
                value={form.maxWeight ?? 1}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  onForm({ ...form, maxWeight: value >= 1 ? null : value });
                }}
                className="flex-1"
              />
              <span className="tabular w-12 text-right text-sm text-ink">
                {form.maxWeight === null
                  ? "no cap"
                  : `${Math.round(form.maxWeight * 100)}%`}
              </span>
            </div>
          </label>
        ) : null}

        {form.method === "mean_variance" ? (
          <div className="grid grid-cols-2 gap-2">
            <label className="text-xs font-medium text-ink-2">
              Risk aversion
              <input
                type="number"
                min={0}
                step={0.5}
                value={form.riskAversion}
                onChange={(event) =>
                  onForm({ ...form, riskAversion: Number(event.target.value) })
                }
                className={`${field} tabular mt-1 block w-full`}
              />
            </label>
            <label className="text-xs font-medium text-ink-2">
              Frontier points
              <input
                type="number"
                min={2}
                max={100}
                step={1}
                placeholder="none"
                value={form.frontier ?? ""}
                onChange={(event) =>
                  onForm({
                    ...form,
                    frontier:
                      event.target.value === ""
                        ? null
                        : Number(event.target.value),
                  })
                }
                className={`${field} tabular mt-1 block w-full`}
              />
            </label>
          </div>
        ) : null}

        <div>
          <label className="flex items-center gap-2 text-xs font-medium text-ink-2">
            <input
              type="checkbox"
              checked={form.rolling !== null}
              onChange={() =>
                onForm({
                  ...form,
                  rolling: form.rolling ? null : { window: 60, step: 7 },
                })
              }
            />
            Rolling windows
          </label>
          {form.rolling ? (
            <div className="mt-2 grid grid-cols-2 gap-2">
              {(["window", "step"] as const).map((key) => (
                <label key={key} className="text-xs text-ink-2">
                  {key === "window" ? "Days each" : "Days apart"}
                  <input
                    type="number"
                    min={1}
                    value={form.rolling![key]}
                    onChange={(event) =>
                      onForm({
                        ...form,
                        rolling: {
                          ...form.rolling!,
                          [key]: Number(event.target.value),
                        },
                      })
                    }
                    className={`${field} tabular mt-1 block w-full`}
                  />
                </label>
              ))}
            </div>
          ) : null}
        </div>
      </fieldset>

      <div>
        <div className="flex items-baseline justify-between">
          <label
            htmlFor="request-json"
            className="text-xs font-medium text-ink-2"
          >
            Request
          </label>
          {edited ? (
            <button
              type="button"
              onClick={onReset}
              className="text-xs text-ink-2 underline"
            >
              Back to the form
            </button>
          ) : (
            <span className="text-xs text-muted">
              Edit for any of cutup&apos;s settings
            </span>
          )}
        </div>
        <textarea
          id="request-json"
          value={text}
          onChange={(event) => onText(event.target.value)}
          spellCheck={false}
          rows={10}
          className={`${field} mt-1 block w-full font-mono text-xs`}
        />
      </div>

      <button
        type="submit"
        disabled={running}
        className="w-full rounded-md bg-ink px-3 py-2 text-sm font-medium text-surface disabled:opacity-60"
      >
        {running ? "Allocating…" : "Allocate"}
      </button>
    </form>
  );
}
