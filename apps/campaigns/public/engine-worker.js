// The browser engine: the campaign engine compiled to wasm, from scripts/build-wasm.sh, run off
// the page's thread. Answers lib/engine.ts's calls, `{id, kind, request}`, with the engine's
// reply and how long the engine took, or with an error.

import init, { allocate, platforms } from "./wasm/campaign_wasm.js";

const ready = init();

self.onmessage = async ({ data: { id, kind, request } }) => {
  try {
    await ready;
    const start = performance.now();
    let reply;
    if (kind === "allocate") {
      const answer = allocate(request);
      reply = { status: answer.status, body: answer.body };
      answer.free();
    } else {
      reply = { status: 200, body: platforms() };
    }
    self.postMessage({ id, reply, compute: performance.now() - start });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
