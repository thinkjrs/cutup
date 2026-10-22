use cutup::{run_portfolio_allocation, CovarianceConfig, MeanVarianceConfig, PortfolioAllocator};
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
