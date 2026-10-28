use cutup::{
    run_portfolio_allocation, Allocation, CovarianceConfig, CutupError, EqualWeight, Hrp,
    MaxSharpe, MeanVariance, MeanVarianceConfig, MinVariance, PortfolioAllocator, Weights,
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

/// A method defined outside the crate: all weight on the asset with the highest mean return.
struct BestMean;

impl Allocation for BestMean {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let tangency = data.allocate(&MaxSharpe::default().allow_short())?;
        let best = (0..tangency.len())
            .max_by(|&a, &b| tangency[a].total_cmp(&tangency[b]))
            .unwrap();
        let mut weights = vec![0.0; tangency.len()];
        weights[best] = 1.0;
        Ok(Weights::from(weights))
    }
}

#[test]
fn methods_mix_in_one_collection() {
    let returns = dmatrix![
         0.010, -0.020,  0.005;
        -0.004,  0.015,  0.002;
         0.007,  0.003, -0.006;
         0.002, -0.001,  0.004;
        -0.003,  0.006,  0.001
    ];
    let allocator = PortfolioAllocator::from_returns(returns).unwrap();
    let methods: Vec<Box<dyn Allocation>> = vec![
        Box::new(EqualWeight),
        Box::new(MinVariance::default().shrinkage(0.1)),
        Box::new(MeanVariance::default().risk_aversion(10.0)),
        Box::new(Hrp::default()),
        Box::new(BestMean),
    ];
    for method in &methods {
        let weights = allocator.allocate(method.as_ref()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
