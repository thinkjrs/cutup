"use client";

// A run's results: which engine answered and how fast, each metric's weights side by side, their
// statistics, the frontier and the weights over time when asked for, and the exports.

import type { Response } from "@/lib/bindings/Response";
import type { Outcome, Timed } from "@/lib/engine";
import {
  metricLabel,
  milliseconds,
  number,
  percent,
  platformColor,
} from "@/lib/format";
import {
  describeError,
  runJson,
  weightsCsv,
  weightsOf,
  windowsOf,
  type Run,
} from "@/lib/series";
import { LineChart } from "./charts/LineChart";
import { ChartCard, DataTable, Legend } from "./charts/parts";
import { SegmentedBars } from "./charts/SegmentedBars";
import { StackedArea } from "./charts/StackedArea";

export type Result = { run: Run; timed: Timed; outcome: Outcome };

function download(name: string, type: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  // Revoked once the download has read it: some browsers cancel it if it's revoked at once.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function Results({ result, all }: { result: Result; all: string[] }) {
  const { run, timed, outcome } = result;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface px-4 py-3 text-sm">
        <dl className="tabular flex flex-wrap gap-x-6 gap-y-1">
          <div>
            <dt className="text-xs text-ink-2">Engine</dt>
            <dd className="font-medium">
              {run.engine === "server"
                ? "Server, native Rust"
                : "Browser, wasm"}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-ink-2">Status</dt>
            <dd className="font-medium">{outcome.status}</dd>
          </div>
          <div>
            <dt className="text-xs text-ink-2">Round trip</dt>
            <dd className="font-medium">{milliseconds(timed.total)}</dd>
          </div>
          <div>
            <dt className="text-xs text-ink-2">Engine time</dt>
            <dd className="font-medium">
              {timed.compute === null ? "–" : milliseconds(timed.compute)}
            </dd>
          </div>
          {outcome.ok ? (
            <div>
              <dt className="text-xs text-ink-2">Days</dt>
              <dd className="font-medium">
                {outcome.response.days.length}, {outcome.response.days[0]} to{" "}
                {outcome.response.days.at(-1)}
              </dd>
            </div>
          ) : null}
        </dl>
        <div className="flex gap-2 text-xs">
          <button
            type="button"
            className="rounded-md border border-border px-2 py-1 hover:bg-grid/60"
            onClick={() =>
              download("allocation.json", "application/json", runJson(run))
            }
          >
            Download JSON
          </button>
          {outcome.ok ? (
            <button
              type="button"
              className="rounded-md border border-border px-2 py-1 hover:bg-grid/60"
              onClick={() =>
                download(
                  "weights.csv",
                  "text/csv",
                  weightsCsv(outcome.response),
                )
              }
            >
              Download CSV
            </button>
          ) : null}
        </div>
      </div>

      {outcome.ok ? (
        <Allocations response={outcome.response} all={all} />
      ) : (
        <div
          role="alert"
          className="rounded-lg border border-border bg-surface px-4 py-3 text-sm"
        >
          <span className="font-medium text-critical">⚠ Not allocated:</span>{" "}
          {describeError(outcome.error)}
        </div>
      )}
    </div>
  );
}

function Allocations({ response, all }: { response: Response; all: string[] }) {
  const names = response.platforms;
  const colors = names.map((name) => platformColor(name, all));
  const legend = (mark: "swatch" | "line") => (
    <Legend
      items={names.map((name, i) => ({ name, color: colors[i]! }))}
      mark={mark}
    />
  );
  const results = response.results;
  const frontiers = results.filter((r) => r.frontier || r.frontier_error);
  const rolling = results.filter((r) => r.rolling);

  return (
    <>
      <ChartCard
        title="Weights by metric"
        note="Each metric's split of the budget across the platforms."
        legend={legend("swatch")}
        chart={
          <SegmentedBars
            names={names}
            colors={colors}
            bars={results.map((r) => ({
              label: metricLabel(r.metric),
              parts: weightsOf(response, r),
              error: r.error === undefined ? undefined : describeError(r.error),
            }))}
          />
        }
        table={
          <DataTable
            head={["Metric", ...names]}
            rows={results.map((r) => {
              const weights = weightsOf(response, r);
              return [
                metricLabel(r.metric),
                ...names.map((_, i) =>
                  weights ? percent(weights[i] ?? 0) : "–",
                ),
              ];
            })}
          />
        }
      />

      <section className="rounded-lg border border-border bg-surface p-4">
        <h3 className="text-sm font-semibold">Statistics</h3>
        <p className="mt-1 text-xs text-ink-2">
          Each day&apos;s return is how far the metric per unit of spend is
          above or below its average over the platforms and days: an expected
          return of 0.05 is 5% more than the average, and −0.05 is 5% less.
        </p>
        <div className="mt-3">
          <DataTable
            head={[
              "Metric",
              "Expected return",
              "Volatility",
              "Sharpe ratio",
              "Diversification ratio",
              ...names.map((name) => `${name} risk`),
            ]}
            rows={results.map((r) => [
              metricLabel(r.metric),
              number(r.stats?.expected_return, 4),
              number(r.stats?.volatility, 3),
              number(r.stats?.sharpe_ratio, 3),
              number(r.stats?.diversification_ratio, 3),
              ...names.map((_, i) => {
                const share = r.stats?.risk_contributions?.[i];
                return share === undefined ? "–" : percent(share);
              }),
            ])}
          />
        </div>
        <h4 className="mt-5 text-xs font-semibold">Losses and drawdowns</h4>
        <p className="mt-1 text-xs text-ink-2">
          CVaR is the average of the worst 5% of days, as a shortfall from the
          average. MAD is the mean absolute deviation from the portfolio&apos;s
          own mean, and the shortfall the mean of each day&apos;s shortfall from
          the average, the days above it counting as 0. A drawdown is how far
          the running sum of the returns is below its highest so far: a run of
          days below the average. CDaR is the average of the worst 5% of them.
        </p>
        <div className="mt-3">
          <DataTable
            head={[
              "Metric",
              "CVaR 95%",
              "MAD",
              "Shortfall",
              "CDaR 95%",
              "Max drawdown",
              "Average drawdown",
            ]}
            rows={results.map((r) => [
              metricLabel(r.metric),
              number(r.stats?.cvar, 3),
              number(r.stats?.mad, 3),
              number(r.stats?.downside_mad, 3),
              number(r.stats?.cdar, 3),
              number(r.stats?.max_drawdown, 3),
              number(r.stats?.average_drawdown, 3),
            ])}
          />
        </div>
      </section>

      {frontiers.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2">
          {frontiers.map((r) => (
            <ChartCard
              key={r.metric}
              title={`Efficient frontier: ${metricLabel(r.metric).toLowerCase()}`}
              note={
                r.frontier_error
                  ? `No frontier: ${describeError(r.frontier_error)}`
                  : "Expected return against volatility. The dot is this allocation."
              }
              chart={
                r.frontier ? (
                  <LineChart
                    shared={false}
                    height={220}
                    xLabel="Volatility"
                    yLabel="Expected return"
                    formatX={(v) => number(v, 3)}
                    formatY={(v) => number(v, 4)}
                    series={[
                      {
                        name: "Frontier",
                        color: "var(--ink-2)",
                        points: r.frontier.map((p) => ({
                          x: p.volatility,
                          y: p.expected_return,
                        })),
                      },
                    ]}
                    markers={
                      r.stats
                        ? [
                            {
                              x: r.stats.volatility,
                              y: r.stats.expected_return,
                              color: "var(--ink)",
                              label: "This allocation",
                            },
                          ]
                        : []
                    }
                  />
                ) : null
              }
              table={
                <DataTable
                  head={["Volatility", "Expected return", ...names]}
                  rows={(r.frontier ?? []).map((p) => [
                    number(p.volatility, 4),
                    number(p.expected_return, 4),
                    ...names.map((name) => percent(p.weights[name] ?? 0)),
                  ])}
                />
              }
            />
          ))}
        </div>
      ) : null}

      {rolling.length > 0 ? (
        <section className="space-y-3">
          <div>
            <h3 className="text-sm font-semibold">Weights over time</h3>
            <p className="mt-1 text-xs text-ink-2">
              Each rolling window&apos;s allocation, by the last day it covers.
            </p>
            <div className="mt-2">{legend("swatch")}</div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {rolling.map((r) => {
              const windows = windowsOf(response, r);
              return (
                <ChartCard
                  key={r.metric}
                  title={metricLabel(r.metric)}
                  chart={
                    <StackedArea
                      labels={windows.map((w) => w.day)}
                      values={windows.map((w) => w.weights)}
                      names={names}
                      colors={colors}
                    />
                  }
                  table={
                    <DataTable
                      head={["Last day", ...names]}
                      rows={windows.map((w) => [
                        w.day,
                        ...names.map((_, i) =>
                          w.weights
                            ? percent(w.weights[i] ?? 0)
                            : describeError(w.error),
                        ),
                      ])}
                    />
                  }
                />
              );
            })}
          </div>
        </section>
      ) : null}
    </>
  );
}
