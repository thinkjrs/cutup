use cutup::{run_portfolio_allocation, PortfolioAllocator};
use nalgebra::dmatrix;

#[test]
fn test_portfolio_allocations() {
    let prices = dmatrix![
        100.0, 200.0, 300.0;
        110.0, 210.0, 310.0;
        120.0, 220.0, 320.0
    ];

    let allocator = PortfolioAllocator::new(prices.clone());

    let ew = allocator.ew_allocation();
    assert_eq!(ew.len(), 3);
    assert!((ew.values().sum::<f64>() - 1.0).abs() < 1e-6);

    let mvo = allocator.mvo_allocation();
    assert_eq!(mvo.len(), 3);
    assert!((mvo.values().sum::<f64>() - 1.0).abs() < 1e-6);

    let hrp = allocator.hrp_allocation();
    assert_eq!(hrp.len(), 3);
    assert!((hrp.values().sum::<f64>() - 1.0).abs() < 1e-6);

    let external = run_portfolio_allocation(prices);
    assert_eq!(external.len(), 3);
    assert!((external.values().sum::<f64>() - 1.0).abs() < 1e-6);
}
