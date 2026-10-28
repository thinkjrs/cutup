use cutup::{
    run_portfolio_allocation, Allocation, CovarianceConfig, CutupError, EqualWeight, Hrp,
    InverseVolatility, MeanVariance, MeanVarianceConfig, MinVariance, PortfolioAllocator, Weights,
};
use nalgebra::dmatrix;

#[test]
fn test_portfolio_allocations() {
    let prices = dmatrix![
        100.0, 200.0, 300.0;
        110.0, 210.0, 310.0;
        120.0, 220.0, 320.0
    ];

    let allocator = PortfolioAllocator::from_prices(prices.clone()).unwrap();

    let ew = allocator.ew_allocation();
    assert_eq!(ew.len(), 3);
    assert!((ew.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    let min_variance = allocator.min_variance_allocation().unwrap();
    assert_eq!(min_variance.len(), 3);
    assert!((min_variance.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    // Two rows of returns for three assets is too little data for an unshrunk covariance.
    let config = MeanVarianceConfig::risk_aversion(2.0)
        .with_covariance(CovarianceConfig::default().with_shrinkage(0.1));
    let mean_variance = allocator.mean_variance_allocation(&config).unwrap();
    assert_eq!(mean_variance.len(), 3);
    assert!((mean_variance.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    let hrp = allocator.hrp_allocation().unwrap();
    assert_eq!(hrp.len(), 3);
    assert!((hrp.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    let external = run_portfolio_allocation(prices).unwrap();
    assert_eq!(external.len(), 3);
    assert!((external.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

/// A method defined outside the crate: all weight on the lowest-volatility asset.
struct LowestVolatility;

impl Allocation for LowestVolatility {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let inverse = data.allocate(&InverseVolatility::default())?;
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
    fn allocate(&self, _: &PortfolioAllocator) -> Result<Weights, CutupError> {
        Ok(Weights::from(self.0.clone()))
    }
}

fn allocator() -> PortfolioAllocator {
    let returns = dmatrix![
         0.010, -0.020,  0.005;
        -0.004,  0.015,  0.002;
         0.007,  0.003, -0.006;
         0.002, -0.001,  0.004;
        -0.003,  0.006,  0.001
    ];
    PortfolioAllocator::from_returns(returns).unwrap()
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
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(
        allocator.allocate(&Fixed(vec![0.5, f64::NAN, 0.5])),
        Err(CutupError::NonFiniteWeight { asset: 1 })
    );
    assert!(allocator.allocate(&Fixed(vec![0.2, 0.3, 0.5])).is_ok());
}
