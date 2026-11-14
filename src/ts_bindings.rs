//! The TypeScript types the `ts-rs` feature generates, pinned against the JSON the `serde`
//! feature reads and writes: a key left out is optional, `null` stands where JSON can't hold
//! an infinite value, and each method is an object of its builders.

use crate::{
    Asset, BlackLitterman, CovarianceConfig, CutupError, EqualWeight, FrontierPoint, Herc, Hrp,
    InverseVariance, InverseVolatility, Linkage, MaxDiversification, MaxSharpe, MeanVariance,
    MinCvar, MinVariance, PerAsset, RiskParity, ShrinkageTarget, View, WeightBounds, Weights,
    Window,
};
use ts_rs::{Config, TS};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

fn decl<T: TS + ?Sized>() -> String {
    T::decl(&Config::new())
}

#[test]
fn values_and_bounds_are_their_json() {
    assert_eq!(decl::<Asset>(), "type Asset = number | string;");
    assert_eq!(
        decl::<PerAsset>(),
        "type PerAsset = Array<number | null> | Record<string, number | null> \
         | Array<[number | string, number | null]>;"
    );
    assert_eq!(decl::<Weights>(), "type Weights = Record<string, number>;");
    assert_eq!(
        decl::<WeightBounds>(),
        "type WeightBounds = { lower?: BoundsSide, upper?: BoundsSide, \
         assets?: Array<AssetBounds>, groups?: Array<GroupLimits>, };"
    );
    assert_eq!(
        decl::<CovarianceConfig>(),
        "type CovarianceConfig = { matrix?: Array<Array<number>> | null, \
         exponentially_weighted?: number | null, exponentially_weighted_decay?: number | null, \
         semicovariance?: boolean, semicovariance_below?: number | null, \
         shrinkage?: number | null, ledoit_wolf?: ShrinkageTarget | null, \
         regularization?: number | null, };"
    );
    assert_eq!(
        decl::<ShrinkageTarget>(),
        "type ShrinkageTarget = \"constant_variance\" | \"constant_correlation\";"
    );
    assert_eq!(
        decl::<Linkage>(),
        "type Linkage = \"single\" | \"complete\" | \"average\" | \"ward\";"
    );
}

#[test]
fn each_method_is_an_object_of_its_builders() {
    for no_settings in [
        decl::<EqualWeight>(),
        decl::<InverseVolatility>(),
        decl::<InverseVariance>(),
    ] {
        assert!(
            no_settings.ends_with(" = Record<string, never>;"),
            "{no_settings}"
        );
    }
    assert_eq!(
        decl::<MinVariance>(),
        "type MinVariance = { bounds?: WeightBounds | null, allow_short?: boolean, \
         covariance?: CovarianceConfig, current_weights?: PerAsset | null, \
         max_turnover?: number | null, turnover_penalty?: number | null, };"
    );
    assert_eq!(
        decl::<MeanVariance>(),
        "type MeanVariance = { risk_aversion?: number | null, target_return?: number | null, \
         target_volatility?: number | null, expected_returns?: PerAsset | null, \
         bounds?: WeightBounds | null, allow_short?: boolean, covariance?: CovarianceConfig, \
         current_weights?: PerAsset | null, max_turnover?: number | null, \
         turnover_penalty?: number | null, transaction_costs?: number | null, \
         transaction_costs_per_asset?: PerAsset | null, };"
    );
    assert_eq!(
        decl::<MaxSharpe>(),
        "type MaxSharpe = { risk_free_rate?: number, expected_returns?: PerAsset | null, \
         covariance?: CovarianceConfig, allow_short?: boolean, };"
    );
    assert_eq!(
        decl::<MaxDiversification>(),
        "type MaxDiversification = { covariance?: CovarianceConfig, };"
    );
    assert_eq!(
        decl::<RiskParity>(),
        "type RiskParity = { risk_budgets?: PerAsset | null, covariance?: CovarianceConfig, };"
    );
    assert_eq!(
        decl::<Hrp>(),
        "type Hrp = { linkage?: Linkage, covariance?: CovarianceConfig, };"
    );
    assert_eq!(
        decl::<Herc>(),
        "type Herc = { linkage?: Linkage, clusters?: number | null, max_clusters?: number, \
         covariance?: CovarianceConfig, };"
    );
    assert_eq!(
        decl::<MinCvar>(),
        "type MinCvar = { confidence?: number, bounds?: WeightBounds, \
         target_return?: number | null, };"
    );
    assert_eq!(
        decl::<BlackLitterman>(),
        "type BlackLitterman = { market_weights: PerAsset, risk_aversion: number, \
         risk_free_rate?: number, tau?: number, views?: Array<View>, \
         covariance?: CovarianceConfig, };"
    );
    assert_eq!(
        decl::<View>(),
        "type View = { absolute?: Asset | null, relative?: [Asset, Asset] | null, \
         portfolio?: PerAsset | null, q: number, variance?: number | null, \
         confidence?: number | null, };"
    );
}

#[test]
fn results_and_errors_are_their_json() {
    assert_eq!(
        decl::<FrontierPoint>(),
        "type FrontierPoint = { expected_return: number, volatility: number, weights: Weights, };"
    );
    assert_eq!(
        decl::<Window>(),
        "type Window = { start: number, end: number, weights?: Weights, error?: CutupError, };"
    );
    let error = decl::<CutupError>();
    assert!(
        error.starts_with("type CutupError = \"no_assets\" | "),
        "{error}"
    );
    assert!(error.contains("\"invalid_config\": { "), "{error}");
    // An infinite or NaN value is null in JSON, as for a target return's missing lower end or
    // a setting given as NaN. Counts, such as a row, are numbers.
    assert!(error.contains("lowest: number | null"), "{error}");
    assert!(error.contains("value: number | null"), "{error}");
    assert!(error.contains("row: number,"), "{error}");
}

// Natively only: exporting writes files, which wasm32-unknown-unknown can't.
#[cfg(not(target_arch = "wasm32"))]
mod export {
    use crate::{
        BlackLitterman, EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility,
        MaxDiversification, MaxSharpe, MeanVariance, MinCvar, MinVariance, RiskParity, Window,
    };
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use ts_rs::{Config, TS};

    /// A directory that's removed when dropped, so a failed assertion doesn't leave it behind.
    struct TempDir(PathBuf);

    impl TempDir {
        fn read(&self, file: &str) -> String {
            std::fs::read_to_string(self.0.join(file)).unwrap()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn exporting_writes_every_type_a_method_needs() {
        let dir = TempDir(std::env::temp_dir().join(format!("cutup-ts-{}", std::process::id())));
        let _ = std::fs::remove_dir_all(&dir.0);
        let cfg = Config::new().with_out_dir(&dir.0);
        MeanVariance::export_all(&cfg).unwrap();
        MinVariance::export_all(&cfg).unwrap();
        MaxSharpe::export_all(&cfg).unwrap();
        MaxDiversification::export_all(&cfg).unwrap();
        RiskParity::export_all(&cfg).unwrap();
        Hrp::export_all(&cfg).unwrap();
        Herc::export_all(&cfg).unwrap();
        MinCvar::export_all(&cfg).unwrap();
        EqualWeight::export_all(&cfg).unwrap();
        InverseVolatility::export_all(&cfg).unwrap();
        InverseVariance::export_all(&cfg).unwrap();
        BlackLitterman::export_all(&cfg).unwrap();
        Window::export_all(&cfg).unwrap();
        let files: BTreeSet<String> = std::fs::read_dir(&dir.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        for name in [
            "MeanVariance.ts",
            "WeightBounds.ts",
            "BoundsSide.ts",
            "AssetBounds.ts",
            "GroupLimits.ts",
            "PerAsset.ts",
            "CovarianceConfig.ts",
            "ShrinkageTarget.ts",
            "Linkage.ts",
            "BlackLitterman.ts",
            "View.ts",
            "Asset.ts",
            "Window.ts",
            "Weights.ts",
            "CutupError.ts",
        ] {
            assert!(files.contains(name), "{name} in {files:?}");
        }
        // A limit is spelled out where it's used, not exported as a type of its own.
        assert!(!files.contains("Limit.ts"), "{files:?}");
        // A side of the bounds, or an asset's or a group's cap, may be null, for no cap. An
        // asset's or a group's floor may not.
        for (file, decl) in [
            (
                "BoundsSide.ts",
                "export type BoundsSide = number | null | PerAsset;",
            ),
            (
                "AssetBounds.ts",
                "export type AssetBounds = { asset: Asset, lower: number, upper: number | null, };",
            ),
            (
                "GroupLimits.ts",
                "export type GroupLimits = { members: Array<Asset>, lower: number, \
                 upper: number | null, };",
            ),
        ] {
            let text = dir.read(file);
            assert!(text.contains(decl), "{file}: {text}");
        }
        // Every import names a file that was written.
        for file in &files {
            let text = dir.read(file);
            for line in text.lines().filter(|l| l.starts_with("import type")) {
                let target = line.rsplit('"').nth(1).unwrap().trim_start_matches("./");
                assert!(files.contains(&format!("{target}.ts")), "{file}: {line}");
            }
        }
    }
}
