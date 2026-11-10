//! Times `allocate` for the methods that run a solver or a clustering step, at 10, 100 and
//! 500 assets.
//!
//! ```bash
//! cargo bench
//! cargo bench -- MinCvar/100
//! ```
//!
//! The solvers are private, so each one is measured through a method that uses it:
//! `MinVariance` for the active-set QP solver, `MinCvar` for the LP, `Hrp` and `Herc` for the
//! clustering, and `RiskParity` for its Newton iteration. Each run covers the whole of
//! `allocate`, including the sample covariance.
//!
//! `cargo test --all-targets` runs this file too, once per benchmark and unoptimized, so
//! there it checks only the smallest size.
//!
//! criterion is a native-only dev-dependency, so on wasm this is an empty program.

#[cfg(not(target_arch = "wasm32"))]
criterion::criterion_main!(native::benches);

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::hint::black_box;

    use criterion::{BenchmarkId, Criterion, criterion_group};
    use cutup::{Allocation, Herc, Hrp, MinCvar, MinVariance, Returns, RiskParity};

    /// The sizes each method is timed at, in assets, with how many samples criterion takes at
    /// each. Its default is 100. Sizes where a solve takes over a tenth of a second get its minimum
    /// of 10, so the full run stays within minutes: at 100 samples, a 1-second solve takes over a
    /// minute and a half to measure.
    const SIZES: [(usize, usize); 3] = [(10, 100), (100, 100), (500, 100)];

    /// The sizes for `MinVariance` and `MinCvar`, where a solve at 500 assets takes seconds:
    /// about 18 for `MinVariance` and 2 for `MinCvar`.
    const SLOW_SIZES: [(usize, usize); 3] = [(10, 100), (100, 100), (500, 10)];

    /// Periods per asset. Twice as many periods as assets keeps the sample covariance well
    /// conditioned at every size.
    const PERIODS_PER_ASSET: usize = 2;

    /// Common factors driving the returns, so assets are correlated as real ones are.
    const FACTORS: usize = 3;

    /// SplitMix64: a small, fixed-seed generator, so every run sees the same returns without a
    /// dependency on `rand`.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// A uniform draw in (0, 1).
        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        /// A standard normal draw, by Box-Muller.
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// Daily-scale returns for `assets` assets from a factor model: each asset has its own
    /// exposures to the common factors, a positive drift, and its own noise.
    fn returns(assets: usize) -> Returns {
        let periods = PERIODS_PER_ASSET * assets;
        let mut rng = SplitMix64(0x5eed ^ assets as u64);

        let exposures: Vec<[f64; FACTORS]> = (0..assets)
            .map(|_| std::array::from_fn(|_| 0.5 + rng.uniform()))
            .collect();
        let drifts: Vec<f64> = (0..assets)
            .map(|_| 0.0002 + 0.0006 * rng.uniform())
            .collect();
        let noise: Vec<f64> = (0..assets).map(|_| 0.005 + 0.015 * rng.uniform()).collect();

        let rows: Vec<Vec<f64>> = (0..periods)
            .map(|_| {
                let factors: [f64; FACTORS] = std::array::from_fn(|_| 0.006 * rng.normal());
                (0..assets)
                    .map(|i| {
                        let common: f64 =
                            exposures[i].iter().zip(&factors).map(|(b, f)| b * f).sum();
                        drifts[i] + common + noise[i] * rng.normal()
                    })
                    .collect()
            })
            .collect();

        Returns::new(&rows).expect("generated returns are finite and rectangular")
    }

    /// Whether this is a real benchmark run. `cargo bench` passes `--bench`; `cargo test` runs
    /// the same binary without it, and criterion then runs each benchmark once.
    fn benchmarking() -> bool {
        std::env::args().any(|arg| arg == "--bench")
    }

    /// Times `method` on the generated returns at each of `sizes`, under `name/assets`.
    fn bench_method(
        c: &mut Criterion,
        name: &str,
        method: &dyn Allocation,
        sizes: &[(usize, usize)],
    ) {
        let sizes = if benchmarking() { sizes } else { &sizes[..1] };
        let mut group = c.benchmark_group(name);
        for &(assets, samples) in sizes {
            group.sample_size(samples);
            // Generated only if criterion's filter selects this benchmark.
            group.bench_function(BenchmarkId::from_parameter(assets), |b| {
                let data = returns(assets);
                b.iter(|| {
                    black_box(&data)
                        .allocate(method)
                        .unwrap_or_else(|e| panic!("{name} failed at {assets} assets: {e}"))
                });
            });
        }
        group.finish();
    }

    fn allocate(c: &mut Criterion) {
        bench_method(c, "MinVariance", &MinVariance::default(), &SLOW_SIZES);
        bench_method(c, "MinCvar", &MinCvar::default(), &SLOW_SIZES);
        bench_method(c, "Hrp", &Hrp::default(), &SIZES);
        bench_method(c, "Herc", &Herc::default(), &SIZES);
        bench_method(c, "RiskParity", &RiskParity::default(), &SIZES);
    }

    criterion_group!(benches, allocate);
}
