//! The [`Allocation`] trait, which every allocation method implements.

use crate::{CutupError, Returns, Weights};

/// An allocation method: anything that turns [`Returns`] into [`Weights`].
///
/// Each built-in method is a value that implements it, built from `default()` and chained
/// settings, and run with [`Returns::allocate`]:
///
/// ```
/// use cutup::{Hrp, Linkage, MaxSharpe, MinVariance, Returns, WeightBounds};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
/// ];
/// let data = Returns::new(&returns)?;
///
/// let capped = data.allocate(&MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.5)))?;
/// let tangency = data.allocate(&MaxSharpe::default().expected_returns([0.004, 0.006, 0.002]))?;
/// let hrp = data.allocate(&Hrp::default().linkage(Linkage::Ward))?;
///
/// assert!(capped.iter().all(|&w| w <= 0.5 + 1e-12));
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// Methods can be mixed in one collection:
///
/// ```
/// use cutup::{Allocation, EqualWeight, InverseVolatility, Returns, RiskParity};
///
/// let data = Returns::new(&[
///      [ 0.01,  0.02],
///      [-0.01, -0.02],
///      [ 0.01, -0.02],
/// ])?;
/// let methods: Vec<Box<dyn Allocation>> = vec![
///     Box::new(EqualWeight),
///     Box::new(InverseVolatility),
///     Box::new(RiskParity::default()),
/// ];
/// for method in &methods {
///     assert_eq!(data.allocate(method)?.len(), 2);
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// Implement it to add a method of your own, returning one weight per asset that sum to 1, as
/// every method in this crate does. Build the [`Weights`] from a `Vec<f64>`, in column order;
/// [`Returns::allocate`] labels them.
pub trait Allocation {
    /// Computes the weights for the returns in `data`.
    ///
    /// # Errors
    ///
    /// Whatever the method can't compute weights for. See each method's type.
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError>;
}

impl<A: Allocation + ?Sized> Allocation for &A {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        (**self).allocate(data)
    }
}

impl<A: Allocation + ?Sized> Allocation for Box<A> {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        (**self).allocate(data)
    }
}
