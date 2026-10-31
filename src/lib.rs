#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// The `covariance`, `shrinkage` and `regularization` setters, for a method with a
/// `covariance: CovarianceConfig` field.
macro_rules! covariance_setters {
    () => {
        /// Applies `covariance` adjustments to the covariance matrix.
        pub fn covariance(mut self, covariance: CovarianceConfig) -> Self {
            self.covariance = covariance;
            self
        }

        /// Shrinks the covariance toward a scaled identity with intensity `lambda`. See
        /// [`CovarianceConfig::shrinkage`].
        pub fn shrinkage(mut self, lambda: f64) -> Self {
            self.covariance = self.covariance.shrinkage(lambda);
            self
        }

        /// Adds `epsilon · I` to the covariance. See [`CovarianceConfig::regularization`].
        pub fn regularization(mut self, epsilon: f64) -> Self {
            self.covariance = self.covariance.regularization(epsilon);
            self
        }
    };
}

/// The `current_weights` and `max_turnover` setters, for a method with a `trading: Trading`
/// field.
macro_rules! turnover_setters {
    () => {
        /// Sets your current weights, [one per asset](PerAsset) in column order or by asset:
        /// what turnover and trading costs are measured against. By asset, those left out are
        /// 0. They must sum to 1, within `1e-9`; to hold cash, include it as an asset. On their
        /// own they change nothing.
        pub fn current_weights(mut self, current: impl Into<PerAsset>) -> Self {
            self.trading.current = Some(current.into());
            self
        }

        /// Caps the turnover `Σᵢ|wᵢ − w₀ᵢ|`, the total weight bought and sold, at `cap`.
        /// Needs [current weights](Self::current_weights). A cap of 0 keeps the current
        /// weights, if they're within the bounds.
        pub fn max_turnover(mut self, cap: f64) -> Self {
            self.trading.max_turnover = Some(cap);
            self
        }
    };
}

mod allocation;
mod asset;
mod black_litterman;
mod bounds;
mod cluster;
mod covariance;
mod cvar;
mod deprecated;
mod eigen;
mod error;
mod lp;
mod methods;
mod qp;
mod returns;
#[cfg(test)]
mod test_support;
mod turnover;
mod weights;

pub use allocation::Allocation;
pub use asset::{Asset, PerAsset};
pub use black_litterman::{BlackLitterman, View};
pub use bounds::WeightBounds;
pub use cluster::Linkage;
pub use covariance::CovarianceConfig;
pub use cvar::MinCvar;
#[allow(deprecated)]
pub use deprecated::PortfolioAllocator;
pub use error::CutupError;
pub use methods::{
    EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility, MaxDiversification, MaxSharpe,
    MeanVariance, MinVariance, RiskParity,
};
pub use returns::Returns;
pub use weights::Weights;

use bounds::{ReturnRange, TargetProblem, interior_start};
use methods::{correlation, in_null_space, long_only_max_ratio, long_only_weights, normalized};
