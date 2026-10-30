//! The names from before 0.3.0: [`PortfolioAllocator`], now [`Returns`], and
//! [`Returns::from_returns`], now [`Returns::new`]. Deprecated since 0.3.0, and removed in
//! 1.0.0. Each is the same item under its old name.
//!
//! Each one warns where it's used:
//!
//! ```compile_fail
//! #![deny(deprecated)]
//! let _ = cutup::PortfolioAllocator::from_prices(&[[1.0], [1.1], [1.2]]);
//! ```
//!
//! ```compile_fail
//! #![deny(deprecated)]
//! let _ = cutup::Returns::from_returns(&[[0.01], [-0.01]]);
//! ```
//!
//! An [`Allocation`](crate::Allocation) written against the old name still compiles, since
//! the alias is the same type:
//!
//! ```
//! #![allow(deprecated)]
//! use cutup::{Allocation, CutupError, PortfolioAllocator, Returns, Weights};
//!
//! struct AllInFirst;
//!
//! impl Allocation for AllInFirst {
//!     fn allocate(&self, _data: &PortfolioAllocator) -> Result<Weights, CutupError> {
//!         Ok(vec![1.0, 0.0].into())
//!     }
//! }
//!
//! let data = Returns::new(&[[0.01, 0.02], [-0.01, 0.01]])?;
//! assert_eq!(data.allocate(&AllInFirst)?.as_slice(), [1.0, 0.0]);
//! # Ok::<(), CutupError>(())
//! ```
//!
//! Under the new names, the same code compiles:
//!
//! ```
//! #![deny(deprecated)]
//! let _ = cutup::Returns::from_prices(&[[1.0], [1.1], [1.2]]);
//! let _ = cutup::Returns::new(&[[0.01], [-0.01]]);
//! ```

use crate::{CutupError, Returns};

/// The name of [`Returns`] before 0.3.0.
#[deprecated(
    since = "0.3.0",
    note = "renamed to `Returns`; `PortfolioAllocator` is removed in 1.0.0"
)]
pub type PortfolioAllocator = Returns;

impl Returns {
    /// The name of [`Returns::new`] before 0.3.0.
    #[deprecated(
        since = "0.3.0",
        note = "use `Returns::new`; `from_returns` is removed in 1.0.0"
    )]
    pub fn from_returns<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, CutupError> {
        Returns::new(rows)
    }
}
