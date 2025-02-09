use cutup::PortfolioAllocator;
use nalgebra::DMatrix;

fn main() {
    // Example price data (rows: time periods, columns: assets)
    let prices = DMatrix::from_row_slice(
        4,
        4,
        &[
            125.0, 1500.0, 210.0, 600.0, 123.0, 1520.0, 215.0, 620.0, 130.0, 1510.0, 220.0, 610.0,
            128.0, 1530.0, 225.0, 630.0,
        ],
    );

    let allocator = PortfolioAllocator::new(prices);

    let mvo_weights = allocator.mvo_allocation();
    let ew_weights = allocator.ew_allocation();
    let hrp_weights = allocator.hrp_allocation();

    println!("MVO Weights: {:?}", mvo_weights);
    println!("EW Weights: {:?}", ew_weights);
    println!("HRP Weights: {:?}", hrp_weights);
}
