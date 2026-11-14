// The two engines behind one interface: the native functions on Vercel's servers, and the
// same Rust compiled to wasm in a worker in the browser. Both take the request's JSON text and
// give the engine's reply as it is, so they can only differ in where the work runs.

import type { ErrorBody } from "./bindings/ErrorBody";
import type { PlatformSummary } from "./bindings/PlatformSummary";
import type { Response } from "./bindings/Response";

export type EngineName = "server" | "browser";

/** A reply from either engine: an HTTP status and the JSON body. */
export type Reply = { status: number; body: string };

/** A reply, with how long it took in milliseconds: the whole round trip, and the engine's own
 * share of it when the engine says. */
export type Timed = { reply: Reply; total: number; compute: number | null };

export interface Engine {
  name: EngineName;
  allocate(request: string): Promise<Timed>;
  platforms(): Promise<Timed>;
  /** Gets ready to answer, so a first call's time is the call's, not the engine's start. */
  warm(): void;
}

/** A reply read: a response, or the engine's error with its status. */
export type Outcome =
  | { ok: true; status: number; response: Response }
  | { ok: false; status: number; error: ErrorBody["error"] };

/** A reply read as a response or an error. A body that isn't the engine's JSON, such as a
 * platform's own page for a crash or a timeout, is an error with its status and its start. */
export function readReply(reply: Reply): Outcome {
  let body: unknown;
  try {
    body = JSON.parse(reply.body);
  } catch {
    body = undefined;
  }
  if (reply.status === 200 && body && typeof body === "object") {
    return { ok: true, status: reply.status, response: body as Response };
  }
  const error = (body as Partial<ErrorBody> | undefined)?.error;
  if (reply.status !== 200 && error !== undefined) {
    return { ok: false, status: reply.status, error };
  }
  const start = reply.body.trim().slice(0, 120);
  return {
    ok: false,
    status: reply.status,
    error: {
      request: {
        message: `status ${reply.status} without the engine's JSON${start ? `: ${start}` : ""}`,
      },
    },
  };
}

export function readPlatforms(reply: Reply): PlatformSummary[] {
  if (reply.status !== 200) {
    throw new Error(`the platforms came back with status ${reply.status}`);
  }
  return JSON.parse(reply.body) as PlatformSummary[];
}

/** The engine's time from a `server-timing` header, `engine;dur=1.234`, if it gives one. */
export function engineTiming(header: string | null): number | null {
  for (const metric of header?.split(",") ?? []) {
    const [name, ...params] = metric.split(";").map((part) => part.trim());
    if (name !== "engine") continue;
    for (const param of params) {
      const [key, value] = param.split("=");
      const duration = Number(value);
      if (key === "dur" && value !== undefined && Number.isFinite(duration)) {
        return duration;
      }
    }
  }
  return null;
}

/** The native functions, at /api. */
export function serverEngine(base = ""): Engine {
  async function send(path: string, init?: RequestInit): Promise<Timed> {
    const start = performance.now();
    const response = await fetch(`${base}${path}`, init);
    const body = await response.text();
    return {
      reply: { status: response.status, body },
      total: performance.now() - start,
      compute: engineTiming(response.headers.get("server-timing")),
    };
  }
  return {
    name: "server",
    allocate: (request) =>
      send("/api/allocate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: request,
      }),
    platforms: () => send("/api/platforms"),
    warm: () => {},
  };
}

/** A message to the worker, and its answer. */
export type WorkerCall = {
  id: number;
  kind: "allocate" | "platforms";
  request?: string;
};
export type WorkerAnswer =
  { id: number; reply: Reply; compute: number } | { id: number; error: string };

/** The wasm build in a worker, started when it's warmed or on the first call, so a visitor who
 * never picks the browser engine never downloads it. A worker that fails, such as on a panic,
 * which leaves the wasm unusable, is replaced on the next call. */
export function browserEngine(url = "/engine-worker.js"): Engine {
  let worker: Worker | null = null;
  let next = 0;
  const pending = new Map<
    number,
    { resolve: (answer: WorkerAnswer) => void; reject: (error: Error) => void }
  >();

  function start(): Worker {
    const started = new Worker(url, { type: "module" });
    started.onmessage = ({ data }: MessageEvent<WorkerAnswer>) => {
      pending.get(data.id)?.resolve(data);
      pending.delete(data.id);
    };
    started.onerror = (event) => {
      event.preventDefault();
      fail(new Error(event.message || "the browser engine failed"));
    };
    return started;
  }

  function fail(error: Error) {
    worker?.terminate();
    worker = null;
    for (const { reject } of pending.values()) reject(error);
    pending.clear();
  }

  async function call(
    kind: WorkerCall["kind"],
    request?: string,
  ): Promise<Timed> {
    worker ??= start();
    const id = next++;
    const begin = performance.now();
    const answer = await new Promise<WorkerAnswer>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker!.postMessage({ id, kind, request } satisfies WorkerCall);
    });
    if ("error" in answer) {
      fail(new Error(answer.error));
      throw new Error(answer.error);
    }
    return {
      reply: answer.reply,
      total: performance.now() - begin,
      compute: answer.compute,
    };
  }

  return {
    name: "browser",
    allocate: (request) => call("allocate", request),
    platforms: () => call("platforms"),
    warm: () => {
      worker ??= start();
    },
  };
}
