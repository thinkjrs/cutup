use cutup::run_portfolio_allocation;
use nalgebra::DMatrix;

fn main() {
    let prices = DMatrix::from_row_slice(
        4,
        4,
        &[
            125.0, 1500.0, 210.0, 600.0, 123.0, 1520.0, 215.0, 620.0, 130.0, 1510.0, 220.0, 610.0,
            128.0, 1530.0, 225.0, 630.0,
        ],
    );

    let weights = run_portfolio_allocation(prices);
    println!("Portfolio Weights: {:?}", weights);
}
