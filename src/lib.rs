#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// The `covariance`, `shrinkage`, `ledoit_wolf` and `regularization` setters, for a method
/// with a `covariance: CovarianceConfig` field. `$method` is its name and `$base` a value of it,
/// written out, for the examples.
macro_rules! covariance_setters {
    ($method:ident, $base:literal) => {
        /// Applies `covariance` adjustments to the covariance matrix, in place of any set
        /// before.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use cutup::{", stringify!($method), ", CovarianceConfig};")]
        ///
        #[doc = concat!("let method = ", $base, ";")]
        /// let adjusted = CovarianceConfig::default().shrinkage(0.1).regularization(1e-6);
        /// // The same as setting each adjustment on the method.
        /// assert_eq!(
        ///     method.clone().covariance(adjusted),
        ///     method.clone().shrinkage(0.1).regularization(1e-6)
        /// );
        /// // The default config adjusts nothing.
        /// assert_eq!(method.clone().covariance(CovarianceConfig::default()), method);
        /// ```
        pub fn covariance(mut self, covariance: CovarianceConfig) -> Self {
            self.covariance = covariance;
            self
        }

        /// Shrinks the covariance toward a scaled identity with intensity `lambda`. See
        /// [`CovarianceConfig::shrinkage`].
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use cutup::{", stringify!($method), ", CovarianceConfig};")]
        ///
        #[doc = concat!("let method = ", $base, ";")]
        /// assert_eq!(
        ///     method.clone().shrinkage(0.1),
        ///     method.covariance(CovarianceConfig::default().shrinkage(0.1))
        /// );
        /// ```
        pub fn shrinkage(mut self, lambda: f64) -> Self {
            self.covariance = self.covariance.shrinkage(lambda);
            self
        }

        /// Shrinks the sample covariance toward `target` with the intensity Ledoit and Wolf
        /// estimate from the returns. See [`CovarianceConfig::ledoit_wolf`].
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use cutup::{", stringify!($method), ", CovarianceConfig, ShrinkageTarget};")]
        ///
        #[doc = concat!("let method = ", $base, ";")]
        /// let target = ShrinkageTarget::ConstantCorrelation;
        /// assert_eq!(
        ///     method.clone().ledoit_wolf(target),
        ///     method.covariance(CovarianceConfig::default().ledoit_wolf(target))
        /// );
        /// ```
        pub fn ledoit_wolf(mut self, target: crate::ShrinkageTarget) -> Self {
            self.covariance = self.covariance.ledoit_wolf(target);
            self
        }

        /// Adds `epsilon · I` to the covariance. See [`CovarianceConfig::regularization`].
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use cutup::{", stringify!($method), ", CovarianceConfig};")]
        ///
        #[doc = concat!("let method = ", $base, ";")]
        /// assert_eq!(
        ///     method.clone().regularization(1e-6),
        ///     method.covariance(CovarianceConfig::default().regularization(1e-6))
        /// );
        /// ```
        pub fn regularization(mut self, epsilon: f64) -> Self {
            self.covariance = self.covariance.regularization(epsilon);
            self
        }
    };
}

/// The `current_weights` and `max_turnover` setters, for a method with a `trading: Trading`
/// field. `$method` is its name and `$base` a value of it, written out, for the examples.
macro_rules! turnover_setters {
    ($method:ident, $base:literal) => {
        /// Sets your current weights, [one per asset](PerAsset) in column order or by asset:
        /// what turnover and trading costs are measured against. By asset, those left out are
        /// 0. They must sum to 1, within `1e-9`; to hold cash, include it as an asset. On their
        /// own they change nothing.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use cutup::{", stringify!($method), ", Returns};")]
        ///
        /// let returns = [
        ///     [ 0.010, -0.020,  0.005],
        ///     [-0.004,  0.015,  0.002],
        ///     [ 0.007,  0.003, -0.006],
        ///     [ 0.002, -0.001,  0.004],
        ///     [-0.003,  0.006,  0.001],
        /// ];
        /// let data = Returns::new(&returns)?;
        #[doc = concat!("let method = ", $base, ";")]
        ///
        /// // Without a cap, costs or a penalty, the weights are as without them.
        /// let held = method.clone().current_weights([0.2, 0.3, 0.5]);
        /// assert_eq!(data.allocate(&held)?, data.allocate(&method)?);
        /// # Ok::<(), cutup::CutupError>(())
        /// ```
        pub fn current_weights(mut self, current: impl Into<PerAsset>) -> Self {
            self.trading.current = Some(current.into());
            self
        }

        /// Caps the turnover `Σᵢ|wᵢ − w₀ᵢ|`, the total weight bought and sold, at `cap`.
        /// Needs [current weights](Self::current_weights). A cap of 0 keeps the current
        /// weights, if they're within the bounds.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use cutup::{", stringify!($method), ", Returns};")]
        ///
        /// let returns = [
        ///     [ 0.010, -0.020,  0.005],
        ///     [-0.004,  0.015,  0.002],
        ///     [ 0.007,  0.003, -0.006],
        ///     [ 0.002, -0.001,  0.004],
        ///     [-0.003,  0.006,  0.001],
        /// ];
        /// let data = Returns::new(&returns)?;
        /// let held = [0.2, 0.3, 0.5];
        #[doc = concat!("let method = ", $base, ".current_weights(held);")]
        ///
        /// // A cap of 0 keeps the current weights.
        /// let kept = data.allocate(&method.clone().max_turnover(0.0))?;
        /// assert_eq!(kept.as_slice(), &held);
        ///
        /// // Any other cap is met.
        /// let capped = data.allocate(&method.max_turnover(0.1))?;
        /// let turnover: f64 = capped.iter().zip(held).map(|(w, h)| (w - h).abs()).sum();
        /// assert!(turnover <= 0.1 + 1e-12, "{turnover}");
        /// # Ok::<(), cutup::CutupError>(())
        /// ```
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
mod cdar;
mod cluster;
mod covariance;
mod cvar;
mod deprecated;
mod dual;
mod eigen;
mod error;
mod expected_returns;
mod lp;
mod mad;
mod methods;
mod qp;
mod returns;
mod rolling;
#[cfg(feature = "serde")]
mod serde_support;
mod stats;
#[cfg(test)]
mod test_support;
#[cfg(all(test, feature = "ts-rs"))]
mod ts_bindings;
mod turnover;
mod weights;

pub use allocation::Allocation;
pub use asset::{Asset, PerAsset};
pub use black_litterman::{BlackLitterman, View};
pub use bounds::WeightBounds;
pub use cdar::MinCdar;
pub use cluster::Linkage;
pub use covariance::{CovarianceConfig, ShrinkageTarget};
pub use cvar::MinCvar;
#[allow(deprecated)]
pub use deprecated::PortfolioAllocator;
pub use error::CutupError;
pub use mad::MinMad;
pub use methods::{
    EqualWeight, FrontierPoint, Herc, Hrp, InverseVariance, InverseVolatility, MaxDiversification,
    MaxSharpe, MeanVariance, MinVariance, RiskParity,
};
pub use returns::Returns;
pub use rolling::{Rolling, RollingAllocation, Window};
pub use stats::PortfolioStats;
pub use weights::Weights;
