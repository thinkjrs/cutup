import { readFile } from "node:fs/promises";
import path from "node:path";
import { Allocator } from "@/components/Allocator";
import { SampleChart } from "@/components/SampleChart";
import { weeklyYields } from "@/lib/sample";

// The sample's weekly yields are read once, when the page builds.
export default async function Page() {
  const csv = await readFile(
    path.join(process.cwd(), "data/platforms_daily.csv"),
    "utf8",
  );
  const sample = weeklyYields(csv);

  return (
    <main className="mx-auto max-w-7xl space-y-10 px-4 py-8 sm:px-6">
      <header className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold">Campaign allocation</h1>
        <p className="text-sm text-ink-2">
          Split an ad budget across platforms with the portfolio methods of{" "}
          <a className="underline" href="https://github.com/thinkjrs/cutup">
            cutup
          </a>
          . Each platform is an asset, and its return on a day is what a unit of
          spend bought there, for each of four metrics. The same Rust runs on
          the server or, compiled to WebAssembly, in your browser.
        </p>
        <p className="text-sm text-ink-2">
          <span className="font-medium text-ink">One caveat:</span> these
          methods take a platform&apos;s results to grow in proportion to its
          spend, so they don&apos;t model saturation. Past some budget, each
          extra unit of spend on a platform buys less than the weights assume.
        </p>
      </header>

      <Allocator />

      <section className="space-y-3">
        <div className="max-w-3xl">
          <h2 className="text-lg font-semibold">The sample</h2>
          <p className="mt-1 text-sm text-ink-2">
            Six platforms&apos; daily spend and results from {sample.weeks[0]}{" "}
            to the end of September 2026. Every volume is scaled by the same
            constant, so the yields are real and the totals aren&apos;t.
          </p>
        </div>
        <SampleChart sample={sample} />
      </section>
    </main>
  );
}
