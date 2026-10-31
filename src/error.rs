//! The errors every constructor and method can return.

use crate::Asset;
use std::fmt;

/// Errors returned when building a `Returns` or computing an allocation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CutupError {
    /// The input has no assets to allocate to: no rows, empty rows, or zero assets.
    NoAssets,
    /// The input has too few rows (observations) for a sample covariance.
    InsufficientData {
        /// Minimum number of rows needed.
        required: usize,
        /// Number of rows supplied.
        actual: usize,
    },
    /// A value in the input is NaN or infinite.
    NonFiniteValue {
        /// Row of the offending value.
        row: usize,
        /// Column (asset) of the offending value.
        col: usize,
    },
    /// A price is zero or negative, so a return can't be computed from it.
    NonPositivePrice {
        /// Row of the offending price.
        row: usize,
        /// Column (asset) of the offending price.
        col: usize,
    },
    /// An asset has zero variance, which the method can't divide by.
    ZeroVariance {
        /// The asset's column.
        column: usize,
        /// The asset's label.
        asset: String,
    },
    /// The input is finite but too extreme for a finite covariance matrix (for example, a
    /// return large enough to overflow when squared).
    NonFiniteCovariance,
    /// The covariance matrix is singular and no finite weights could be computed from it.
    SingularCovariance,
    /// A configuration value is out of range.
    InvalidConfig {
        /// Name of the offending field.
        field: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// One asset's value in a per-asset setting is out of range, such as a negative lower
    /// bound or a current weight that isn't finite.
    InvalidAssetValue {
        /// Name of the setting.
        field: &'static str,
        /// The asset's column.
        column: usize,
        /// The asset's label.
        asset: String,
        /// The value supplied.
        value: f64,
    },
    /// A row of the input has a different number of values from the first row. For flat
    /// input, the last row is short: the number of values isn't a multiple of the number of
    /// assets.
    RaggedRows {
        /// The row, counting from 0.
        row: usize,
        /// The number of values in the first row, or the number of assets for flat input.
        expected: usize,
        /// The number of values in this row.
        actual: usize,
    },
    /// A setting, or an [`Allocation`](crate::Allocation)'s weights, don't have one value per
    /// asset.
    DimensionMismatch {
        /// Name of the setting, or `weights`.
        field: &'static str,
        /// Number of assets.
        expected: usize,
        /// Number of values supplied.
        actual: usize,
    },
    /// An [`Asset`] isn't in the data: no asset has its label, or its column is out of range.
    UnknownAsset {
        /// The asset, by label or by column, as given.
        asset: Asset,
    },
    /// An asset is given more than once: a label repeated in
    /// [`Returns::labeled`](crate::Returns::labeled), an asset named twice in the pairs of a
    /// [`PerAsset`](crate::PerAsset), or a relative [`View`](crate::View) of an asset against
    /// itself.
    DuplicateAsset {
        /// The asset's column: for a repeated label, the column it's repeated in.
        column: usize,
        /// The asset's label.
        asset: String,
    },
    /// [`Weights`](crate::Weights) are labeled with other assets than the data they're used with,
    /// or in another order.
    LabelMismatch {
        /// The first column whose labels differ.
        column: usize,
        /// The data's label for it.
        expected: String,
        /// The weights' label for it.
        actual: String,
    },
    /// Every asset has the same expected return, so no portfolio can target a different one.
    DegenerateExpectedReturns,
    /// The mean-variance objective has no maximum: the covariance matrix is singular, and some
    /// fully invested portfolio gains expected return without adding variance. Add shrinkage or
    /// regularization, or use more observations.
    Unbounded,
    /// The risk-free rate is at or above (or within rounding of) the minimum variance
    /// portfolio's expected return, so no fully invested portfolio attains the maximum Sharpe
    /// ratio: it is only approached as positions grow without limit.
    RiskFreeRateTooHigh {
        /// The risk-free rate supplied.
        risk_free_rate: f64,
        /// The minimum variance portfolio's expected return, which the rate must be below.
        min_variance_return: f64,
    },
    /// An iterative solver stopped before reaching a solution, although one exists. This
    /// happens only with extreme inputs, such as risk budgets more than about 10¹⁴ apart.
    NotConverged,
    /// No long-only portfolio has the target return: it must be between the lowest and the
    /// highest expected return of any asset, or of any portfolio within the weight bounds.
    TargetReturnOutOfRange {
        /// The target return supplied.
        target_return: f64,
        /// The lowest expected return of any long-only portfolio within the bounds.
        lowest: f64,
        /// The highest expected return of any long-only portfolio within the bounds.
        highest: f64,
    },
    /// No asset's expected return is above the risk-free rate by more than rounding, so no
    /// long-only portfolio has a positive Sharpe ratio.
    NoPositiveExcessReturn {
        /// The risk-free rate supplied.
        risk_free_rate: f64,
        /// The highest expected return of any asset, which the rate must be below.
        highest_return: f64,
    },
    /// No portfolio summing to 1 satisfies the weight bounds: the lower bounds sum to more
    /// than 1, or the upper bounds to less.
    InfeasibleBounds {
        /// The sum of the lower bounds.
        lower_sum: f64,
        /// The sum of the upper bounds.
        upper_sum: f64,
    },
    /// An [`Allocation`](crate::Allocation) returned a weight that is NaN or infinite.
    NonFiniteWeight {
        /// The asset's column.
        column: usize,
        /// The asset's label.
        asset: String,
    },
    /// The views given to [`BlackLitterman`](crate::BlackLitterman) are degenerate: certain views
    /// that repeat or contradict each other, or a view whose variance follows its portfolio's on a
    /// portfolio with no variance, which would make it certain. Give such a view an explicit
    /// [`View::variance`](crate::View::variance).
    DegenerateViews,
    /// A turnover cap, turnover penalty or transaction costs were set without current weights
    /// to measure them against.
    MissingCurrentWeights,
    /// No portfolio within the weight bounds can be reached from the current weights within
    /// the turnover cap: moving the current weights into the bounds alone turns over more.
    InfeasibleTurnover {
        /// The least turnover that reaches the bounds.
        minimum: f64,
        /// The turnover cap.
        cap: f64,
    },
    /// A setting isn't supported in combination with another yet.
    NotSupported {
        /// The setting.
        setting: &'static str,
        /// What it can't be combined with.
        with: &'static str,
    },
}

impl fmt::Display for CutupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CutupError::NoAssets => write!(f, "input has no assets"),
            CutupError::InsufficientData { required, actual } => write!(
                f,
                "input has {actual} rows but at least {required} are required"
            ),
            CutupError::NonFiniteValue { row, col } => {
                write!(f, "value at row {row}, column {col} is NaN or infinite")
            }
            CutupError::NonPositivePrice { row, col } => {
                write!(f, "price at row {row}, column {col} is zero or negative")
            }
            CutupError::ZeroVariance { column, asset } => {
                write!(f, "{} has zero variance", AssetName(*column, asset))
            }
            CutupError::NonFiniteCovariance => {
                write!(
                    f,
                    "input is too extreme to compute a finite covariance matrix"
                )
            }
            CutupError::SingularCovariance => write!(
                f,
                "covariance matrix is singular; no finite weights could be computed"
            ),
            CutupError::InvalidConfig { field, value } => {
                write!(f, "config value `{field}` is out of range: {value}")
            }
            CutupError::InvalidAssetValue {
                field,
                column,
                asset,
                value,
            } => write!(
                f,
                "`{field}` for {} is out of range: {value}",
                AssetName(*column, asset)
            ),
            CutupError::RaggedRows {
                row,
                expected,
                actual,
            } => write!(
                f,
                "row {row} has the wrong number of values: {actual}, not {expected}"
            ),
            CutupError::DimensionMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "`{field}` needs one value per asset ({expected}) but got {actual}"
            ),
            CutupError::UnknownAsset { asset } => match asset.label() {
                Some(label) => write!(f, "no asset is labeled `{label}`"),
                None => write!(f, "there is no asset in {asset}"),
            },
            CutupError::DuplicateAsset { column, asset } => {
                write!(f, "{} is given more than once", AssetName(*column, asset))
            }
            CutupError::LabelMismatch {
                column,
                expected,
                actual,
            } => write!(
                f,
                "the weights label column {column} `{actual}`, but the data labels it `{expected}`"
            ),
            CutupError::DegenerateExpectedReturns => write!(
                f,
                "every asset has the same expected return, so no other target return can be reached"
            ),
            CutupError::Unbounded => write!(
                f,
                "objective is unbounded: expected return can grow without adding variance"
            ),
            CutupError::RiskFreeRateTooHigh {
                risk_free_rate,
                min_variance_return,
            } => write!(
                f,
                "risk-free rate {risk_free_rate} must be below the minimum variance portfolio's \
                 expected return {min_variance_return}"
            ),
            CutupError::NotConverged => {
                write!(f, "the solver stopped before reaching a solution")
            }
            CutupError::TargetReturnOutOfRange {
                target_return,
                lowest,
                highest,
            } => write!(
                f,
                "target return {target_return} is outside the reachable expected returns, \
                 from {lowest} to {highest}, so no long-only portfolio reaches it"
            ),
            CutupError::NoPositiveExcessReturn {
                risk_free_rate,
                highest_return,
            } => write!(
                f,
                "risk-free rate {risk_free_rate} must be below the highest expected return \
                 {highest_return} for a long-only portfolio to have a positive Sharpe ratio"
            ),
            CutupError::InfeasibleBounds {
                lower_sum,
                upper_sum,
            } => write!(
                f,
                "weight bounds can't sum to 1: lower bounds sum to {lower_sum} and upper \
                 bounds to {upper_sum}"
            ),
            CutupError::NonFiniteWeight { column, asset } => {
                write!(
                    f,
                    "weight for {} is NaN or infinite",
                    AssetName(*column, asset)
                )
            }
            CutupError::MissingCurrentWeights => write!(
                f,
                "turnover and transaction cost settings need current weights to measure from"
            ),
            CutupError::InfeasibleTurnover { minimum, cap } => write!(
                f,
                "turnover cap {cap} is below {minimum}, the least turnover that reaches the \
                 weight bounds from the current weights"
            ),
            CutupError::NotSupported { setting, with } => {
                write!(f, "`{setting}` isn't supported with {with} yet")
            }
            CutupError::DegenerateViews => write!(
                f,
                "views repeat or contradict each other, or one with a default or confidence \
                 variance is on a portfolio with no variance; give it an explicit variance"
            ),
        }
    }
}

impl std::error::Error for CutupError {}

/// An asset in an error message: by label and column, or by column alone when its label is
/// its column, as without [`Returns::labeled`].
struct AssetName<'a>(usize, &'a str);

impl fmt::Display for AssetName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let AssetName(column, label) = *self;
        if label == column.to_string() {
            write!(f, "the asset in column {column}")
        } else {
            write!(f, "asset `{label}` (column {column})")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CutupError::InsufficientData {
            required: 3,
            actual: 1,
        };
        assert_eq!(
            err.to_string(),
            "input has 1 rows but at least 3 are required"
        );
        assert_eq!(
            CutupError::ZeroVariance {
                column: 2,
                asset: "GLD".into()
            }
            .to_string(),
            "asset `GLD` (column 2) has zero variance"
        );
        assert_eq!(
            CutupError::UnknownAsset {
                asset: "QQQ".into()
            }
            .to_string(),
            "no asset is labeled `QQQ`"
        );
        assert_eq!(
            CutupError::UnknownAsset {
                asset: Asset::from(3)
            }
            .to_string(),
            "there is no asset in column 3"
        );
        assert_eq!(
            CutupError::InvalidAssetValue {
                field: "lower_bound",
                column: 0,
                asset: "SPY".into(),
                value: -0.1
            }
            .to_string(),
            "`lower_bound` for asset `SPY` (column 0) is out of range: -0.1"
        );
        assert_eq!(
            CutupError::NonFiniteWeight {
                column: 1,
                asset: "TLT".into()
            }
            .to_string(),
            "weight for asset `TLT` (column 1) is NaN or infinite"
        );
        assert_eq!(
            CutupError::DuplicateAsset {
                column: 0,
                asset: "SPY".into()
            }
            .to_string(),
            "asset `SPY` (column 0) is given more than once"
        );
        // Without labels, the column alone.
        assert_eq!(
            CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            }
            .to_string(),
            "the asset in column 1 has zero variance"
        );
        assert_eq!(
            CutupError::LabelMismatch {
                column: 1,
                expected: "TLT".into(),
                actual: "GLD".into()
            }
            .to_string(),
            "the weights label column 1 `GLD`, but the data labels it `TLT`"
        );
        assert_eq!(
            CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            }
            .to_string(),
            "config value `shrinkage` is out of range: 1.5"
        );
        assert_eq!(
            CutupError::DimensionMismatch {
                field: "expected_returns",
                expected: 3,
                actual: 2
            }
            .to_string(),
            "`expected_returns` needs one value per asset (3) but got 2"
        );
        assert_eq!(
            CutupError::RiskFreeRateTooHigh {
                risk_free_rate: 0.02,
                min_variance_return: 0.01
            }
            .to_string(),
            "risk-free rate 0.02 must be below the minimum variance portfolio's expected return 0.01"
        );
        assert_eq!(
            CutupError::Unbounded.to_string(),
            "objective is unbounded: expected return can grow without adding variance"
        );
        assert_eq!(
            CutupError::DegenerateExpectedReturns.to_string(),
            "every asset has the same expected return, so no other target return can be reached"
        );
        assert_eq!(
            CutupError::NotConverged.to_string(),
            "the solver stopped before reaching a solution"
        );
        assert_eq!(
            CutupError::TargetReturnOutOfRange {
                target_return: 0.5,
                lowest: 0.1,
                highest: 0.2
            }
            .to_string(),
            "target return 0.5 is outside the reachable expected returns, from 0.1 to 0.2, so \
             no long-only portfolio reaches it"
        );
        assert_eq!(
            CutupError::NoPositiveExcessReturn {
                risk_free_rate: 0.03,
                highest_return: 0.02
            }
            .to_string(),
            "risk-free rate 0.03 must be below the highest expected return 0.02 for a long-only \
             portfolio to have a positive Sharpe ratio"
        );
        assert_eq!(
            CutupError::InfeasibleBounds {
                lower_sum: 0.0,
                upper_sum: 0.9
            }
            .to_string(),
            "weight bounds can't sum to 1: lower bounds sum to 0 and upper bounds to 0.9"
        );
    }
}
