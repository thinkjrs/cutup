"use client";

// The app: a request on the left, sent to the engine of your choice, and its results on the
// right. The same request text goes to either engine, and the results say which answered and
// how long it took.

import { useEffect, useMemo, useState } from "react";
import type { PlatformSummary } from "@/lib/bindings/PlatformSummary";
import type { Request } from "@/lib/bindings/Request";
import {
  browserEngine,
  readPlatforms,
  readReply,
  serverEngine,
  type Engine,
  type EngineName,
} from "@/lib/engine";
import { platformOrder } from "@/lib/format";
import { DEFAULT_FORM, requestText, toRequest, type Form } from "@/lib/request";
import { RequestForm } from "./RequestForm";
import { Results, type Result } from "./Results";

const ENGINES: { name: EngineName; label: string; note: string }[] = [
  { name: "server", label: "Server", note: "Native Rust on Vercel" },
  { name: "browser", label: "Browser", note: "The same Rust as wasm, here" },
];

export function Allocator() {
  const engines = useMemo<Record<EngineName, Engine>>(
    () => ({ server: serverEngine(), browser: browserEngine() }),
    [],
  );
  const [engine, setEngine] = useState<EngineName>("server");
  const [notice, setNotice] = useState<string | null>(null);
  const [platforms, setPlatforms] = useState<PlatformSummary[] | null>(null);
  const [form, setForm] = useState<Form>(DEFAULT_FORM);
  const [text, setText] = useState(() => requestText(toRequest(DEFAULT_FORM)));
  const [edited, setEdited] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  // The platforms, from the server engine, or from the browser's if the server isn't there,
  // as under `next dev` without `vercel dev`.
  useEffect(() => {
    let live = true;
    engines.server
      .platforms()
      .then(({ reply }) => readPlatforms(reply))
      .catch(async () => {
        const { reply } = await engines.browser.platforms();
        if (live) {
          setEngine("browser");
          setNotice(
            "The server engine didn't answer, so the browser engine is on.",
          );
        }
        return readPlatforms(reply);
      })
      .then((summaries) => {
        // Listed in the order they take colors, the default three first.
        const order = platformOrder(summaries.map((s) => s.platform));
        const sorted = [...summaries].sort(
          (a, b) => order.indexOf(a.platform) - order.indexOf(b.platform),
        );
        if (live) setPlatforms(sorted);
      })
      .catch((error) => live && setFailure(`No platforms: ${error}`));
    return () => {
      live = false;
    };
  }, [engines]);

  const changeForm = (next: Form) => {
    setForm(next);
    setText(requestText(toRequest(next)));
  };

  async function run() {
    setRunning(true);
    setFailure(null);
    let request: Request | string = text;
    try {
      request = JSON.parse(text) as Request;
    } catch {
      // Sent as it is: the engine answers 400, with serde's reason.
    }
    try {
      const timed = await engines[engine].allocate(text);
      const outcome = readReply(timed.reply);
      setResult({
        timed,
        outcome,
        run: {
          engine,
          request,
          status: outcome.status,
          milliseconds: { total: timed.total, engine: timed.compute },
          response: outcome.ok ? outcome.response : null,
          error: outcome.ok ? null : outcome.error,
        },
      });
    } catch (error) {
      setFailure(`The ${engine} engine failed: ${error}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[20rem_1fr]">
      <aside className="space-y-5">
        <div>
          <div className="text-xs font-medium text-ink-2">Engine</div>
          <div
            role="radiogroup"
            aria-label="Engine"
            className="mt-1.5 grid grid-cols-2 gap-1 rounded-lg border border-border bg-surface p-1"
          >
            {ENGINES.map(({ name, label, note }) => (
              <button
                key={name}
                type="button"
                role="radio"
                aria-checked={engine === name}
                onClick={() => {
                  engines[name].warm();
                  setEngine(name);
                  setNotice(null);
                }}
                className={`rounded-md px-2 py-1.5 text-left ${
                  engine === name ? "bg-ink text-surface" : "hover:bg-grid/60"
                }`}
              >
                <div className="text-sm font-medium">{label}</div>
                <div
                  className={`text-xs ${engine === name ? "opacity-80" : "text-ink-2"}`}
                >
                  {note}
                </div>
              </button>
            ))}
          </div>
          {notice ? (
            <p className="mt-1.5 text-xs text-ink-2">{notice}</p>
          ) : null}
        </div>

        {platforms ? (
          <RequestForm
            platforms={platforms}
            form={form}
            onForm={changeForm}
            text={text}
            edited={edited}
            onText={(next) => {
              setText(next);
              setEdited(true);
            }}
            onReset={() => {
              setEdited(false);
              setText(requestText(toRequest(form)));
            }}
            onRun={run}
            running={running}
          />
        ) : failure ? null : (
          <p className="text-sm text-ink-2">Loading the platforms…</p>
        )}
      </aside>

      <div className={`min-w-0 space-y-4 ${running ? "opacity-60" : ""}`}>
        {failure ? (
          <div
            role="alert"
            className="rounded-lg border border-border bg-surface px-4 py-3 text-sm"
          >
            <span className="font-medium text-critical">⚠ Failed:</span>{" "}
            {failure}
          </div>
        ) : null}
        {result ? (
          <Results
            result={result}
            all={platforms?.map((p) => p.platform) ?? []}
          />
        ) : (
          <div className="rounded-lg border border-dashed border-border px-4 py-10 text-center text-sm text-ink-2">
            Choose platforms, metrics and a method, then allocate.
          </div>
        )}
      </div>
    </div>
  );
}
