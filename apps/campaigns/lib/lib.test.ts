// The frontend's own logic, over the engine's committed fixture replies where it reads one, so
// it's tested against what the engines really send.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Response } from "./bindings/Response";
import { engineTiming, readPlatforms, readReply } from "./engine";
import { number, percent, platformColor, platformOrder, tick } from "./format";
import { DEFAULT_FORM, lowestCap, toRequest } from "./request";
import { weekOf, weeklyYields } from "./sample";
import { extent, linear, nearest, nice } from "./scale";
import {
  describeError,
  runJson,
  weightsCsv,
  weightsOf,
  windowsOf,
} from "./series";

const replies: Record<string, { status: number; body: unknown }> = JSON.parse(
  readFileSync(new URL("../fixtures/replies.json", import.meta.url), "utf8"),
);

function response(name: string): Response {
  const reply = replies[name]!;
  const outcome = readReply({
    status: reply.status,
    body: JSON.stringify(reply.body),
  });
  if (!outcome.ok) throw new Error(`${name} isn't a response`);
  return outcome.response;
}

describe("engine", () => {
  it("reads the engine's time from server-timing", () => {
    expect(engineTiming("engine;dur=12.345")).toBe(12.345);
    expect(engineTiming("cdn-cache;desc=MISS, engine; dur=0.5")).toBe(0.5);
    expect(engineTiming("engine")).toBeNull();
    expect(engineTiming("engine;dur=abc")).toBeNull();
    expect(engineTiming("other;dur=3")).toBeNull();
    expect(engineTiming(null)).toBeNull();
  });

  it("reads replies as responses or errors", () => {
    expect(response("min_variance").platforms).toEqual([
      "Meta",
      "Google",
      "TikTok",
    ]);
    const reply = replies.error_unknown_platform!;
    expect(
      readReply({ status: reply.status, body: JSON.stringify(reply.body) }),
    ).toEqual({
      ok: false,
      status: 422,
      error: { unknown_platform: { platform: "Myspace" } },
    });
  });

  it("reads a reply that isn't the engine's JSON as an error with its status", () => {
    expect(
      readReply({
        status: 504,
        body: "An error occurred with your deployment",
      }),
    ).toEqual({
      ok: false,
      status: 504,
      error: {
        request: {
          message:
            "status 504 without the engine's JSON: An error occurred with your deployment",
        },
      },
    });
    expect(readReply({ status: 500, body: "" })).toEqual({
      ok: false,
      status: 500,
      error: { request: { message: "status 500 without the engine's JSON" } },
    });
    expect(readReply({ status: 200, body: "<html>" }).ok).toBe(false);
  });

  it("reads the platforms only from a 200", () => {
    const body =
      '[{"platform":"Meta","first":"2025-09-01","last":"2026-09-30","days_with_spend":395}]';
    expect(readPlatforms({ status: 200, body })[0]?.platform).toBe("Meta");
    expect(() => readPlatforms({ status: 405, body: "{}" })).toThrow("405");
  });
});

describe("request", () => {
  it("offers only caps the platforms can meet", () => {
    expect(lowestCap(2)).toBe(0.5);
    expect(lowestCap(3)).toBe(0.35);
    expect(lowestCap(4)).toBe(0.25);
    expect(lowestCap(5)).toBe(0.2);
    expect(lowestCap(6)).toBe(0.2);
    expect(lowestCap(1)).toBe(1);
    expect(lowestCap(0)).toBe(1);
  });

  it("leaves out what the method doesn't take", () => {
    expect(toRequest(DEFAULT_FORM)).toEqual({
      platforms: ["Meta", "Google", "TikTok"],
      metrics: ["impressions", "clicks", "link_clicks", "views"],
      method: { name: "min_variance" },
    });
    const equal = toRequest({
      ...DEFAULT_FORM,
      method: "equal_weight",
      maxWeight: 0.5,
      frontier: 10,
    });
    expect(equal.method).toEqual({ name: "equal_weight" });
    expect(equal.frontier).toBeUndefined();
  });

  it("carries the risk methods' confidence and threshold", () => {
    const form = { ...DEFAULT_FORM, confidence: 0.8, downside: -0.1 };
    expect(toRequest({ ...form, method: "min_cdar" }).method).toEqual({
      name: "min_cdar",
      confidence: 0.8,
    });
    expect(toRequest({ ...form, method: "min_cvar" }).method).toEqual({
      name: "min_cvar",
      confidence: 0.8,
    });
    expect(
      toRequest({ ...form, method: "min_mad", maxWeight: 0.5 }).method,
    ).toEqual({ name: "min_mad", bounds: { upper: 0.5 }, downside: -0.1 });
    // With no threshold, the deviation either side of the mean.
    expect(
      toRequest({ ...form, method: "min_mad", downside: null }).method,
    ).toEqual({ name: "min_mad" });
    // Neither goes to a method that doesn't take it.
    expect(toRequest({ ...form, method: "hrp" }).method).toEqual({
      name: "hrp",
    });
  });

  it("carries the settings a method takes", () => {
    expect(
      toRequest({
        ...DEFAULT_FORM,
        method: "mean_variance",
        maxWeight: 0.6,
        riskAversion: 4,
        frontier: 20,
        from: "2026-01-01",
        to: "2026-06-30",
        rolling: { window: 60, step: 7 },
      }),
    ).toEqual({
      platforms: ["Meta", "Google", "TikTok"],
      metrics: ["impressions", "clicks", "link_clicks", "views"],
      method: {
        name: "mean_variance",
        bounds: { upper: 0.6 },
        risk_aversion: 4,
      },
      from: "2026-01-01",
      to: "2026-06-30",
      frontier: 20,
      rolling: { window: 60, step: 7 },
    });
  });
});

describe("series", () => {
  it("orders weights by the response's platforms", () => {
    const min = response("min_variance");
    const result = min.results[0]!;
    const weights = weightsOf(min, result)!;
    expect(weights).toEqual(min.platforms.map((p) => result.weights![p]));
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it("dates each rolling window by the days it covers", () => {
    const rolling = response("min_cvar_rolling");
    const windows = windowsOf(rolling, rolling.results[0]!);
    expect(windows).toHaveLength(7);
    expect(windows[0]).toMatchObject({
      first: rolling.days[35],
      day: rolling.days[124],
    });
    expect(windows.at(-1)!.day).toBe(rolling.days.at(-1));
    expect(windows[1]!.weights).toEqual([0, 0, 1]);
  });

  it("writes every allocation as CSV", () => {
    const rolling = response("min_cvar_rolling");
    const lines = weightsCsv(rolling).trimEnd().split("\n");
    expect(lines[0]).toBe(
      "metric,window,first_day,last_day,Meta,Google,TikTok",
    );
    expect(lines).toHaveLength(1 + 1 + 7);
    expect(lines[1]).toMatch(/^clicks,all,2025-09-01,2026-09-30,/);
    expect(lines[3]).toBe(
      `clicks,1,${rolling.days[80]},${rolling.days[169]},0,0,1`,
    );
  });

  it("puts errors in words", () => {
    expect(describeError("no_platforms")).toBe("No platforms");
    expect(describeError({ unknown_platform: { platform: "Myspace" } })).toBe(
      'Unknown platform: platform "Myspace"',
    );
    expect(
      describeError({ request: { message: "expected value at line 1" } }),
    ).toBe("Request: expected value at line 1");
    expect(describeError({ rolling: { error: "no_assets" } })).toBe(
      "Rolling: No assets",
    );
  });

  it("exports a run as JSON", () => {
    const run = {
      engine: "browser" as const,
      request: "{",
      status: 400,
      milliseconds: { total: 3, engine: 1 },
      response: null,
      error: { request: { message: "EOF" } },
    };
    expect(JSON.parse(runJson(run))).toEqual(run);
  });
});

describe("sample", () => {
  it("weeks start on Monday", () => {
    expect(weekOf("2025-09-01")).toBe("2025-09-01");
    expect(weekOf("2025-09-07")).toBe("2025-09-01");
    expect(weekOf("2025-09-08")).toBe("2025-09-08");
    expect(weekOf("2026-03-01")).toBe("2026-02-23");
  });

  it("weights each week's yield by its spend", () => {
    const csv = [
      "date,platform,spend,impressions,clicks,link_clicks,views",
      "2025-09-01,Meta,100,1000,10,5,50",
      "2025-09-02,Meta,300,1000,10,5,50",
      "2025-09-01,Google,0,0,0,0,0",
      "2025-09-08,Google,50,500,5,5,0",
    ].join("\n");
    const sample = weeklyYields(csv);
    expect(sample.weeks).toEqual(["2025-09-01", "2025-09-08"]);
    expect(sample.platforms).toEqual(["Google", "Meta"]);
    expect(sample.yields.impressions.Meta).toEqual([5, null]);
    expect(sample.yields.clicks.Google).toEqual([null, 0.1]);
    expect(sample.yields.views.Google).toEqual([null, 0]);
  });

  it("reads the bundled sample", () => {
    const csv = readFileSync(
      new URL("../data/platforms_daily.csv", import.meta.url),
      "utf8",
    );
    const sample = weeklyYields(csv);
    expect(sample.platforms).toEqual([
      "Google",
      "Meta",
      "Reddit",
      "Snap",
      "Spotify",
      "TikTok",
    ]);
    expect(sample.weeks[0]).toBe("2025-09-01");
    expect(sample.weeks.at(-1)).toBe("2026-09-28");
  });
});

describe("scale and format", () => {
  it("rounds axes out to round ticks", () => {
    expect(nice(0.13, 0.87)).toEqual({
      domain: [0, 1],
      ticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
    });
    expect(nice(2, 2).domain[0]).toBeLessThan(2);
    expect(extent([null, 3, -1, Number.NaN, undefined])).toEqual([-1, 3]);
    expect(extent([null])).toBeNull();
    expect(linear(0, 10, 0, 100)(2.5)).toBe(25);
    expect(linear(1, 1, 0, 100)(1)).toBe(50);
    expect(nearest([0, 10, 20], 14)).toBe(1);
  });

  it("formats numbers and colors platforms by their place", () => {
    expect(percent(0.4567)).toBe("45.7%");
    expect(percent(-1e-17)).toBe("0.0%");
    expect(percent(-0.25)).toBe("-25.0%");
    expect(number(1234.5678, 5)).toBe("1,234.6");
    expect(number(null)).toBe("–");
    // A solver's rounding of 0, such as equal weight's expected return, is 0.
    expect(number(-5.2e-16, 4)).toBe("0");
    expect(number(0.000123456, 4)).toBe("0.0001235");
    expect(number(-0.4243918, 4)).toBe("-0.4244");
    expect(tick(2500)).toBe("2,500");
    expect(tick(25_000)).toBe("25K");
    const all = ["Google", "Meta", "Reddit", "Snap", "Spotify", "TikTok"];
    expect(platformOrder(all)).toEqual([
      "Meta",
      "Google",
      "TikTok",
      "Reddit",
      "Snap",
      "Spotify",
    ]);
    expect(platformColor("Meta", all)).toBe("var(--series-1)");
    expect(platformColor("TikTok", all)).toBe("var(--series-3)");
    expect(platformColor("Spotify", all)).toBe("var(--series-6)");
    expect(platformColor("Myspace", all)).toBe("var(--muted)");
  });
});
