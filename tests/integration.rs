use cutup::{
    Allocation, CovarianceConfig, CutupError, EqualWeight, Hrp, InverseVolatility, MeanVariance,
    MinVariance, Returns, Weights,
};

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
    send_and_sync::<cutup::MinCvar>();
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
