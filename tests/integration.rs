use cutup::{
    Allocation, CovarianceConfig, CutupError, EqualWeight, Herc, Hrp, InverseVolatility,
    MaxDiversification, MaxSharpe, MeanVariance, MinCdar, MinCvar, MinMad, MinVariance, Returns,
    RiskParity, WeightBounds, Weights,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[test]
fn test_portfolio_allocations() {
    let prices = [
        [100.0, 200.0, 300.0],
        [110.0, 210.0, 310.0],
        [120.0, 220.0, 320.0],
    ];

    let allocator = Returns::from_prices(&prices).unwrap();

    let ew = allocator.allocate(&EqualWeight).unwrap();
    assert_eq!(ew.len(), 3);
    assert!((ew.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    let min_variance = allocator
        .allocate(&MinVariance::default().allow_short())
        .unwrap();
    assert_eq!(min_variance.len(), 3);
    assert!((min_variance.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    // Two rows of returns for three assets is too little data for an unshrunk covariance.
    let method = MeanVariance::risk_aversion(2.0)
        .covariance(CovarianceConfig::default().shrinkage(0.1))
        .allow_short();
    let mean_variance = allocator.allocate(&method).unwrap();
    assert_eq!(mean_variance.len(), 3);
    assert!((mean_variance.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    let hrp = allocator.allocate(&Hrp::default()).unwrap();
    assert_eq!(hrp.len(), 3);
    assert!((hrp.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

/// A method defined outside the crate: all weight on the lowest-volatility asset.
struct LowestVolatility;

impl Allocation for LowestVolatility {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let inverse = data.allocate(&InverseVolatility)?;
        let lowest = (0..inverse.len())
            .max_by(|&a, &b| inverse[a].total_cmp(&inverse[b]))
            .unwrap();
        let mut weights = vec![0.0; inverse.len()];
        weights[lowest] = 1.0;
        Ok(Weights::from(weights))
    }
}

/// A method that returns whatever weights it was given.
struct Fixed(Vec<f64>);

impl Allocation for Fixed {
    fn allocate(&self, _: &Returns) -> Result<Weights, CutupError> {
        Ok(Weights::from(self.0.clone()))
    }
}

fn allocator() -> Returns {
    let returns = [
        [0.010, -0.020, 0.005],
        [-0.004, 0.015, 0.002],
        [0.007, 0.003, -0.006],
        [0.002, -0.001, 0.004],
        [-0.003, 0.006, 0.001],
    ];
    Returns::new(&returns).unwrap()
}

#[test]
fn methods_mix_in_one_collection() {
    let allocator = allocator();
    let methods: Vec<Box<dyn Allocation>> = vec![
        Box::new(EqualWeight),
        Box::new(MinVariance::default().shrinkage(0.1)),
        Box::new(MeanVariance::risk_aversion(10.0)),
        Box::new(Hrp::default()),
        Box::new(LowestVolatility),
    ];
    for method in &methods {
        let weights = allocator.allocate(method).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    // Asset 2 has the lowest volatility.
    assert_eq!(
        allocator.allocate(&LowestVolatility).unwrap().as_slice(),
        [0.0, 0.0, 1.0]
    );
}

#[test]
fn allocate_rejects_malformed_weights() {
    let allocator = allocator();
    assert_eq!(
        allocator.allocate(&Fixed(vec![0.5, 0.5])),
        Err(CutupError::DimensionMismatch {
            field: "weights",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(
        allocator.allocate(&Fixed(vec![0.5, f64::NAN, 0.5])),
        Err(CutupError::NonFiniteWeight {
            column: 1,
            asset: "1".into()
        })
    );
    assert!(allocator.allocate(&Fixed(vec![0.2, 0.3, 0.5])).is_ok());
}

/// Every public type can be shared and sent between threads. This fails to compile, rather
/// than to run, if one stops being `Send + Sync`.
#[test]
fn public_types_are_send_and_sync() {
    fn send_and_sync<T: Send + Sync>() {}
    send_and_sync::<cutup::Asset>();
    send_and_sync::<cutup::BlackLitterman>();
    send_and_sync::<CovarianceConfig>();
    send_and_sync::<CutupError>();
    send_and_sync::<EqualWeight>();
    send_and_sync::<cutup::Herc>();
    send_and_sync::<Hrp>();
    send_and_sync::<cutup::InverseVariance>();
    send_and_sync::<InverseVolatility>();
    send_and_sync::<cutup::Linkage>();
    send_and_sync::<cutup::MaxDiversification>();
    send_and_sync::<cutup::MaxSharpe>();
    send_and_sync::<MeanVariance>();
    send_and_sync::<cutup::MinCdar>();
    send_and_sync::<cutup::MinCvar>();
    send_and_sync::<cutup::MinMad>();
    send_and_sync::<MinVariance>();
    send_and_sync::<cutup::PerAsset>();
    send_and_sync::<Returns>();
    send_and_sync::<cutup::RiskParity>();
    send_and_sync::<cutup::View>();
    send_and_sync::<cutup::WeightBounds>();
    send_and_sync::<Weights>();
}

#[test]
fn weights_display_as_a_labeled_table() {
    let returns = [[0.01, -0.02], [-0.004, 0.015], [0.007, 0.003]];
    let data = Returns::new(&returns)
        .unwrap()
        .labeled(["SPY", "GOOGL"])
        .unwrap();
    let weights = data.allocate(&EqualWeight).unwrap();
    assert_eq!(weights.to_string(), "SPY    0.5000\nGOOGL  0.5000");
    assert_eq!(format!("{weights:.2}"), "SPY    0.50\nGOOGL  0.50");
    let values: Vec<f64> = weights.into_iter().collect();
    assert_eq!(values, [0.5, 0.5]);
    // Short positions line up with long ones.
    let long_short = Weights::from(vec![1.25, -0.25]);
    assert_eq!(long_short.to_string(), "0   1.2500\n1  -0.2500");
}

#[test]
fn every_method_reports_its_settings_before_the_data() {
    // The second asset never moves, so every method that builds a covariance has a zero
    // variance to report. Each method here also has a setting that's wrong, and says so first:
    // what's wrong with a request doesn't depend on the data it's sent with.
    let rows = [
        [0.01, 0.0, 0.02],
        [-0.02, 0.0, 0.01],
        [0.03, 0.0, -0.01],
        [0.01, 0.0, 0.03],
        [-0.01, 0.0, -0.02],
        [0.02, 0.0, 0.01],
    ];
    let data = Returns::new(&rows).unwrap();
    assert!(matches!(
        data.allocate(&MinVariance::default()),
        Err(CutupError::ZeroVariance { column: 1, .. })
    ));
    let crossed = WeightBounds::uniform(0.5, 0.2);
    let narrow = WeightBounds::uniform(0.0, 0.1);
    let shrunk = CovarianceConfig::default().shrinkage(-1.0);
    let current = [0.2, 0.3, 0.5];
    type Method = Box<dyn Allocation>;
    let config: Vec<(&str, Method)> = vec![
        (
            "shrinkage",
            Box::new(MinVariance::default().shrinkage(-1.0)),
        ),
        (
            "shrinkage",
            Box::new(MinVariance::default().allow_short().shrinkage(-1.0)),
        ),
        (
            "upper_bound",
            Box::new(MinVariance::default().bounds(crossed.clone())),
        ),
        (
            "max_turnover",
            Box::new(
                MinVariance::default()
                    .current_weights(current)
                    .max_turnover(-0.1),
            ),
        ),
        ("risk_aversion", Box::new(MeanVariance::risk_aversion(-1.0))),
        (
            "risk_aversion",
            Box::new(MeanVariance::risk_aversion(-1.0).allow_short()),
        ),
        (
            "target_return",
            Box::new(MeanVariance::target_return(f64::NAN)),
        ),
        (
            "target_return",
            Box::new(MeanVariance::target_return(f64::NAN).allow_short()),
        ),
        (
            "target_volatility",
            Box::new(MeanVariance::target_volatility(-1.0)),
        ),
        (
            "target_volatility",
            Box::new(MeanVariance::target_volatility(-1.0).allow_short()),
        ),
        (
            "shrinkage",
            Box::new(MeanVariance::risk_aversion(1.0).covariance(shrunk)),
        ),
        (
            "upper_bound",
            Box::new(MeanVariance::risk_aversion(1.0).bounds(crossed.clone())),
        ),
        (
            "transaction_costs",
            Box::new(
                MeanVariance::risk_aversion(1.0)
                    .current_weights(current)
                    .transaction_costs(-0.1),
            ),
        ),
        (
            "risk_free_rate",
            Box::new(MaxSharpe::default().risk_free_rate(f64::NAN)),
        ),
        (
            "risk_free_rate",
            Box::new(MaxSharpe::default().allow_short().risk_free_rate(f64::NAN)),
        ),
        ("shrinkage", Box::new(MaxSharpe::default().shrinkage(-1.0))),
        (
            "shrinkage",
            Box::new(MaxDiversification::default().shrinkage(-1.0)),
        ),
        ("shrinkage", Box::new(RiskParity::default().shrinkage(-1.0))),
        ("shrinkage", Box::new(Hrp::default().shrinkage(-1.0))),
        ("shrinkage", Box::new(Herc::default().shrinkage(-1.0))),
        ("clusters", Box::new(Herc::default().clusters(0))),
        ("confidence", Box::new(MinCvar::default().confidence(1.5))),
        (
            "upper_bound",
            Box::new(MinCvar::default().bounds(crossed.clone())),
        ),
        ("threshold", Box::new(MinMad::default().downside(f64::NAN))),
        (
            "upper_bound",
            Box::new(MinMad::default().bounds(crossed.clone())),
        ),
        ("confidence", Box::new(MinCdar::default().confidence(1.5))),
        ("upper_bound", Box::new(MinCdar::default().bounds(crossed))),
    ];
    for (field, method) in &config {
        match data.allocate(method.as_ref()) {
            Err(CutupError::InvalidConfig { field: named, .. }) => assert_eq!(named, *field),
            other => panic!("{field}: {other:?}"),
        }
    }
    // So do settings that are wrong only for this data: one value per asset, and bounds no
    // portfolio of these three assets can meet.
    let sized: Vec<(&str, Method)> = vec![
        (
            "expected_returns",
            Box::new(MeanVariance::risk_aversion(1.0).expected_returns([0.1, 0.2])),
        ),
        (
            "expected_returns",
            Box::new(MaxSharpe::default().expected_returns([0.1, 0.2])),
        ),
        (
            "risk_budgets",
            Box::new(RiskParity::default().risk_budgets([0.5, 0.5])),
        ),
    ];
    for (field, method) in &sized {
        match data.allocate(method.as_ref()) {
            Err(CutupError::DimensionMismatch { field: named, .. }) => assert_eq!(named, *field),
            other => panic!("{field}: {other:?}"),
        }
    }
    let infeasible: [Method; 3] = [
        Box::new(MinVariance::default().bounds(narrow.clone())),
        Box::new(MeanVariance::target_return(0.01).bounds(narrow.clone())),
        Box::new(
            MinVariance::default()
                .bounds(narrow)
                .current_weights(current)
                .max_turnover(0.5),
        ),
    ];
    for method in &infeasible {
        let result = data.allocate(method.as_ref());
        assert!(
            matches!(result, Err(CutupError::InfeasibleBounds { .. })),
            "{result:?}"
        );
    }
    assert!(matches!(
        data.allocate(&RiskParity::default().risk_budgets([0.5, -0.5, 1.0])),
        Err(CutupError::InvalidAssetValue {
            field: "risk_budgets",
            column: 1,
            ..
        })
    ));
}

/// Six periods of four assets' returns, given in thousandths, with each asset's scaled by
/// `2^(−gap · tier)`.
fn scaled(base: [[i32; 4]; 6], tiers: [i32; 4], gap: i32) -> Returns {
    let rows: Vec<Vec<f64>> = base
        .iter()
        .map(|row| {
            (0..4)
                .map(|i| f64::from(row[i]) / 1000.0 * 2f64.powi(-gap * tiers[i]))
                .collect()
        })
        .collect();
    Returns::new(&rows).unwrap()
}

/// A problem's name, its data and method, and its optimum.
type Problem = (&'static str, Returns, Box<dyn Allocation>, [f64; 4]);

#[test]
fn weights_are_the_optimum_whatever_the_assets_scales() {
    // Some assets' returns are from 2⁻²⁰ to 2⁻⁴⁸ times the size of the others', as cash would
    // be next to anything else, and far past it. The expected weights are each problem's
    // optimum, solved for in exact rational arithmetic.
    //
    // The solver's tolerances were all measured against the largest quantity in the problem,
    // which in its units is nothing like the same for every asset. Each case notes what was
    // returned then, and which of the two fixes it fails without:
    //
    // * solving again from where a step lands, when that's far smaller than where it started;
    // * releasing a held weight by its own share of the multipliers, not the largest share.
    let capped = || WeightBounds::uniform(0.0, 0.4);
    let cases: [Problem; 6] = [
        (
            // 0.4834 and 0.5166. The first step, from the equal weights, lands 2⁴⁰ times
            // smaller in the solver's units, where a short clamped to 0 was within the
            // rounding of the start. It needs the second solve.
            "long-only, two assets 2⁻⁴⁰ the others' size",
            scaled(
                [
                    [-25, -1, 23, 2],
                    [-18, 24, 27, 27],
                    [-9, -24, 19, 23],
                    [-28, 30, -20, 3],
                    [15, 0, 18, -15],
                    [-10, 24, 6, -26],
                ],
                [0, 1, 0, 1],
                40,
            ),
            Box::new(MinVariance::default()),
            [
                4.638488265036829e-13,
                0.4972206244096623,
                0.0,
                0.5027793755898738,
            ],
        ),
        (
            // `SingularCovariance`. It needs both.
            "capped, assets at 2⁻²⁴ and 2⁻⁴⁸",
            scaled(
                [
                    [28, -14, 3, 18],
                    [1, 3, 27, -30],
                    [10, -8, 2, 12],
                    [-6, 26, -14, 13],
                    [-18, -3, 29, 5],
                    [4, -22, 28, -19],
                ],
                [0, 1, 2, 0],
                24,
            ),
            Box::new(MinVariance::default().bounds(capped())),
            [0.1298676316210788, 0.4, 0.4, 0.07013236837892113],
        ),
        (
            // [0, 0.4, 0.2, 0.4]. The asset at 2⁻³² is at its cap, with a share of the
            // budget's multiplier 2³² times the others', next to which the first asset's rate
            // passed for rounding. It needs its own share.
            "capped, one asset at 2⁻³²",
            scaled(
                [
                    [8, -13, 16, 7],
                    [-23, -14, 7, -9],
                    [-10, -9, -26, 3],
                    [-4, -11, -15, 20],
                    [-3, -2, 7, 12],
                    [19, -2, 28, 7],
                ],
                [0, 2, 0, 0],
                16,
            ),
            Box::new(MinVariance::default().bounds(capped())),
            [0.023032069967994447, 0.4, 0.1769679300320055, 0.4],
        ),
        (
            // [0.2705, 0.4, 0.3295, 0]. It needs its own share.
            "capped, one asset at 2⁻⁴⁰",
            scaled(
                [
                    [6, -14, 27, -29],
                    [30, 12, -29, 18],
                    [29, 10, 10, 26],
                    [27, 3, -4, 13],
                    [-20, 4, 6, 23],
                    [-21, -3, -27, 8],
                ],
                [0, 1, 0, 0],
                40,
            ),
            Box::new(MinVariance::default().bounds(capped())),
            [
                0.0751645720802539,
                0.4,
                0.2501573161007181,
                0.27467811181902796,
            ],
        ),
        (
            // All of the weight in the second asset, the one at 2⁻³². It needs its own share.
            "a risk aversion of 4, one asset at 2⁻³²",
            scaled(
                [
                    [6, 20, 28, -28],
                    [28, -28, -26, -29],
                    [-12, 8, -25, 22],
                    [-10, 29, 22, 10],
                    [3, 9, -15, -20],
                    [4, -27, -20, -7],
                ],
                [0, 1, 0, 0],
                32,
            ),
            Box::new(MeanVariance::risk_aversion(4.0)),
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            // [0, 0, 0.5, 0.5]. The two assets at 2⁻³² are capped at 0.5 between them. It
            // needs its own share.
            "a group capped at 0.5, its two assets at 2⁻³²",
            scaled(
                [
                    [-22, -3, 23, 23],
                    [-19, 4, 10, -19],
                    [29, -27, -27, 17],
                    [-26, -18, 4, -1],
                    [-12, -23, -12, -14],
                    [14, -5, -27, 24],
                ],
                [0, 1, 1, 0],
                32,
            ),
            Box::new(MinVariance::default().bounds(WeightBounds::default().group(
                [1, 2],
                0.0,
                0.5,
            ))),
            [0.17127071831415666, 0.0, 0.5, 0.32872928168584337],
        ),
    ];
    // To 1e-12 of each weight, not of 1: the first problem's optimum holds 4.6e-13 of its first
    // asset, which a weight of 0 is within 1e-12 of. A weight the optimum doesn't hold is 0.
    let missed: Vec<String> = cases
        .iter()
        .filter_map(
            |(name, data, method, exact)| match data.allocate(method.as_ref()) {
                Ok(weights)
                    if weights
                        .iter()
                        .zip(exact)
                        .all(|(w, e)| (w - e).abs() <= 1e-12 * e.abs()) =>
                {
                    None
                }
                other => Some(format!("{name}: {other:?}")),
            },
        )
        .collect();
    assert!(missed.is_empty(), "{missed:#?}");
}

#[test]
fn a_group_capped_at_zero_leaves_its_assets_out() {
    // Two of six assets are in a group capped at 0, so the weights are those of the other
    // four on their own. A third of these failed with `SingularCovariance`: the group's row
    // has no terms, since its assets hold nothing, and one of them left at 1e-32 by a solve
    // was the whole of the row's sum, which no fraction of its terms allowed for.
    let mut state = 7_u64;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    for case in 0..60 {
        let load: Vec<f64> = (0..6).map(|_| 1.5 * uniform()).collect();
        let rows: Vec<Vec<f64>> = (0..24)
            .map(|_| {
                let factor = 0.04 * uniform() - 0.02;
                (0..6)
                    .map(|i| load[i] * factor + 0.03 * (uniform() - 0.5))
                    .collect()
            })
            .collect();
        let rest: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| vec![row[0], row[2], row[3], row[5]])
            .collect();
        let grouped = WeightBounds::default().group([1, 4], 0.0, 0.0);
        let methods: [(Box<dyn Allocation>, Box<dyn Allocation>); 2] = [
            (
                Box::new(MinVariance::default().bounds(grouped.clone())),
                Box::new(MinVariance::default()),
            ),
            (
                Box::new(MeanVariance::risk_aversion(4.0).bounds(grouped)),
                Box::new(MeanVariance::risk_aversion(4.0)),
            ),
        ];
        for (with_group, without) in methods {
            let weights = Returns::new(&rows)
                .unwrap()
                .allocate(with_group.as_ref())
                .unwrap_or_else(|e| panic!("case {case}: {e:?}"));
            let expected = Returns::new(&rest)
                .unwrap()
                .allocate(without.as_ref())
                .unwrap();
            assert_eq!([weights[1], weights[4]], [0.0, 0.0], "case {case}");
            for (w, e) in [0, 2, 3, 5].into_iter().zip(expected.iter()) {
                assert!((weights[w] - e).abs() < 1e-12, "case {case}: {weights:?}");
            }
        }
    }
}

#[test]
fn rebalancing_assets_of_different_scales_is_solved_as_before() {
    // Two problems the solver already had right, which tighter tolerances stopped with
    // `NotConverged` while this was being fixed. Rebalancing holds weights at their current
    // values and adds a second constraint, so its variables sit on the ends of their segments
    // with targets a rounding to either side.
    let thousandths = |rows: &[&[i32]], small: usize, gap: i32| {
        let rows: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, &r)| {
                        let scale = if i == small { 1.0 } else { 2f64.powi(-gap) };
                        f64::from(r) / 1000.0 * scale
                    })
                    .collect()
            })
            .collect();
        Returns::new(&rows).unwrap()
    };

    // All but the third asset at 2⁻²⁰, a target return, and turnover capped at 1. A target
    // counted as past its bound by 1e-12 of the problem's size in the weight's own units is
    // below what the solve itself rounds by there, from a gap of about 2¹⁴.
    let data = thousandths(
        &[
            &[-1, -13, -26, -14],
            &[25, -18, 14, 9],
            &[13, -23, 17, -24],
            &[1, -7, 12, 21],
            &[-7, 3, 13, -16],
            &[-24, -22, 17, 14],
            &[10, -3, 5, 14],
        ],
        2,
        20,
    );
    let current = vec![
        0.08281677794522875,
        0.0,
        0.25928936103487293,
        0.6578938610198983,
    ];
    let method = MeanVariance::target_return(0.0019261500894490408)
        .current_weights(current.clone())
        .max_turnover(1.0);
    let weights = data.allocate(&method).unwrap();
    let expected = [
        0.5828167779375923,
        0.0,
        0.25928924182557467,
        0.15789398023683293,
    ];
    for (w, e) in weights.iter().zip(expected) {
        assert!((w - e).abs() < 1e-9, "{weights:?}");
    }
    let turnover: f64 = weights
        .iter()
        .zip(&current)
        .map(|(w, c)| (w - c).abs())
        .sum();
    assert!(turnover <= 1.0 + 1e-9, "{turnover}");

    // All but the first asset at 2⁻¹⁷, a target return, and costs that leave no trade worth
    // making. Solved again from where its first step lands, the first asset is alone free at
    // 0 with a target of -4e-18: the rounding of the budget in its own units, and far more
    // than 1e-12 of the others there.
    let data = thousandths(
        &[
            &[10, 10, -5, 11, 27],
            &[26, -19, -5, 25, 19],
            &[19, -10, -2, 1, -4],
            &[2, 30, -7, 5, -11],
            &[25, -24, -15, -2, -4],
            &[-23, 20, -17, -22, 25],
            &[-14, -24, -24, -14, 5],
            &[-22, 6, -5, -26, 29],
            &[15, -18, -10, 13, 23],
            &[23, -15, 6, -15, -8],
            &[-30, 13, 20, 27, -25],
            &[12, 1, -16, -23, -1],
            &[-12, -1, 17, -23, -2],
        ],
        0,
        17,
    );
    let current = vec![
        0.0,
        0.020606946662768817,
        0.0,
        0.7239273274568093,
        0.25546572588042193,
    ];
    let method = MeanVariance::target_return(-0.000000007699052334077002)
        .current_weights(current.clone())
        .transaction_costs(0.000023137970629909118);
    let weights = data.allocate(&method).unwrap();
    assert_eq!(weights.as_slice(), current);
}

#[test]
fn a_target_that_leaves_out_the_one_asset_with_a_return_is_solved() {
    // Long-short, only the first asset has an expected return, and the target is 0: the
    // optimum holds none of it, so its term in the return's constraint is all rounding. That
    // was measured against itself, and the solve was refused as `SingularCovariance`.
    let rows = [
        [0.010, -0.020, 0.005],
        [-0.004, 0.015, 0.002],
        [0.007, 0.003, -0.006],
        [0.002, -0.001, 0.004],
        [-0.003, 0.006, 0.001],
    ];
    let data = Returns::new(&rows).unwrap();
    // Without the first asset, the least variance of the other two is in closed form.
    let column = |j: usize| rows.iter().map(|row| row[j]).collect::<Vec<f64>>();
    let covariance = |a: &[f64], b: &[f64]| {
        let n = a.len() as f64;
        let (mean_a, mean_b) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let products = a.iter().zip(b).map(|(x, y)| (x - mean_a) * (y - mean_b));
        products.sum::<f64>() / (n - 1.0)
    };
    let (second, third) = (column(1), column(2));
    let (v2, v3, c23) = (
        covariance(&second, &second),
        covariance(&third, &third),
        covariance(&second, &third),
    );
    let w2 = (v3 - c23) / (v2 + v3 - 2.0 * c23);
    for expected in [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]] {
        for target in [0.0, 1e-13, -1e-13] {
            let method = MeanVariance::target_return(target)
                .allow_short()
                .expected_returns(expected);
            let weights = data.allocate(&method).unwrap();
            let first = target / expected[0];
            for (got, want) in
                weights
                    .iter()
                    .zip([first, w2 * (1.0 - first), (1.0 - w2) * (1.0 - first)])
            {
                assert!((got - want).abs() < 1e-9, "{target}: {weights:?}");
            }
        }
        // And the frontier, whose last or first target is that 0.
        let method = MeanVariance::risk_aversion(1.0)
            .allow_short()
            .expected_returns(expected);
        let frontier = data.frontier(&method, 5).unwrap();
        assert_eq!(frontier.len(), 5);
    }
}
