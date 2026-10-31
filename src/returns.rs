//! [`Returns`], the data every allocation method works on.

use crate::{Allocation, CutupError, PerAsset, Weights, asset};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

/// Asset returns, with their sample covariance: the data every allocation method works on. Pass an
/// [`Allocation`], such as [`MinVariance`](crate::MinVariance) or [`Hrp`](crate::Hrp), to
/// [`Returns::allocate`] to get [`Weights`].
///
/// All covariance-based methods work on the covariance of asset returns, never of price levels.
/// Build one from prices with [`Returns::from_prices`], which converts them to returns, or from
/// returns you have already computed with [`Returns::new`].
///
/// Before 0.3.0 this was `PortfolioAllocator`, which stays as a deprecated alias until 1.0.0.
#[derive(Debug, Clone)]
pub struct Returns {
    pub(crate) returns: DMatrix<f64>,
    pub(crate) cov_matrix: DMatrix<f64>,
    pub(crate) labels: Arc<[String]>,
}

/// Equal when the returns and labels are: the covariance follows from the returns.
impl PartialEq for Returns {
    fn eq(&self, other: &Self) -> bool {
        self.labels == other.labels && self.returns == other.returns
    }
}

impl Returns {
    /// Creates a new `Returns` from asset prices, one row per period and one value
    /// per asset in each row.
    ///
    /// Prices are converted to simple returns, `r_t = p_t / p_{t-1} - 1`, before the covariance
    /// matrix is computed. This matches `pandas.DataFrame.pct_change`. `T` rows of prices
    /// produce `T - 1` rows of returns, so at least three rows of prices are needed for a
    /// sample covariance.
    ///
    /// Rows can be arrays, `Vec`s or slices: anything that is `AsRef<[f64]>`. For one flat,
    /// row-major slice, use [`Returns::from_prices_flat`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if there are no rows, or the rows are empty.
    /// * [`CutupError::RaggedRows`] if a row has a different length from the first.
    /// * [`CutupError::InsufficientData`] if there are fewer than three rows.
    /// * [`CutupError::NonFiniteValue`] if any price is NaN or infinite.
    /// * [`CutupError::NonPositivePrice`] if any price is zero or negative.
    /// * [`CutupError::NonFiniteCovariance`] if the returns are too extreme for a finite
    ///   covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let prices = [
    ///    [100.0, 200.0, 300.0],
    ///    [110.0, 210.0, 310.0],
    ///    [120.0, 220.0, 320.0],
    /// ];
    ///
    /// let data = Returns::from_prices(&prices)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn from_prices<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, CutupError> {
        Returns::prices_matrix(Returns::rows_matrix(rows)?)
    }

    /// Creates a new `Returns` from asset prices in one flat slice, row-major: the
    /// first `assets` values are the first period's prices, the next `assets` the second's,
    /// and so on. This is the layout of a JavaScript `Float64Array` or a CSV read row by row.
    /// Otherwise it's the same as [`Returns::from_prices`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if `assets` is 0.
    /// * [`CutupError::RaggedRows`] if the number of values isn't a multiple of `assets`.
    /// * Otherwise as for [`Returns::from_prices`].
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let prices = [
    ///     100.0, 200.0, 300.0,
    ///     110.0, 210.0, 310.0,
    ///     120.0, 220.0, 320.0,
    /// ];
    ///
    /// let data = Returns::from_prices_flat(&prices, 3)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn from_prices_flat(values: &[f64], assets: usize) -> Result<Self, CutupError> {
        Returns::prices_matrix(Returns::flat_matrix(values, assets)?)
    }

    /// Creates a new `Returns` from a matrix of asset prices (rows: time, columns:
    /// assets), with the `nalgebra` feature. Otherwise it's the same as
    /// [`Returns::from_prices`].
    ///
    /// The feature tracks the nalgebra version cutup depends on, so a nalgebra release that
    /// isn't compatible can change which `DMatrix` this takes.
    ///
    /// # Errors
    ///
    /// As for [`Returns::from_prices`], except [`CutupError::RaggedRows`], which a
    /// matrix can't have.
    #[cfg(feature = "nalgebra")]
    #[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
    pub fn from_prices_matrix(prices: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::prices_matrix(prices)
    }

    /// Creates a new `Returns` from asset returns, one row per period and one value
    /// per asset in each row.
    ///
    /// Use this when you already have returns (for example log returns, or returns net of a
    /// benchmark). They are used as-is; at least two rows are needed for a sample covariance.
    /// For prices, use [`Returns::from_prices`]: in 0.1, `PortfolioAllocator::new` took prices,
    /// but this takes returns, and prices passed here are treated as returns.
    ///
    /// Rows can be arrays, `Vec`s or slices: anything that is `AsRef<[f64]>`. For one flat,
    /// row-major slice, use [`Returns::new_flat`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if there are no rows, or the rows are empty.
    /// * [`CutupError::RaggedRows`] if a row has a different length from the first.
    /// * [`CutupError::InsufficientData`] if there are fewer than two rows.
    /// * [`CutupError::NonFiniteValue`] if any return is NaN or infinite.
    /// * [`CutupError::NonFiniteCovariance`] if the returns are too extreme for a finite
    ///   covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let returns = [
    ///    [0.010, -0.020, 0.005],
    ///    [-0.004, 0.015, 0.002],
    ///    [0.007, 0.003, -0.006],
    /// ];
    ///
    /// let data = Returns::new(&returns)?;
    ///
    /// // Rows built at runtime work the same way.
    /// let rows: Vec<Vec<f64>> = returns.iter().map(|row| row.to_vec()).collect();
    /// let from_vecs = Returns::new(&rows)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn new<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, CutupError> {
        Returns::returns_matrix(Returns::rows_matrix(rows)?)
    }

    /// Creates a new `Returns` from asset returns in one flat slice, row-major: the
    /// first `assets` values are the first period's returns, the next `assets` the second's,
    /// and so on. This is the layout of a JavaScript `Float64Array` or a CSV read row by row.
    /// Otherwise it's the same as [`Returns::new`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if `assets` is 0.
    /// * [`CutupError::RaggedRows`] if the number of values isn't a multiple of `assets`.
    /// * Otherwise as for [`Returns::new`].
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let returns = [
    ///     0.010, -0.020, 0.005,
    ///     -0.004, 0.015, 0.002,
    ///     0.007, 0.003, -0.006,
    /// ];
    ///
    /// let data = Returns::new_flat(&returns, 3)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn new_flat(values: &[f64], assets: usize) -> Result<Self, CutupError> {
        Returns::returns_matrix(Returns::flat_matrix(values, assets)?)
    }

    /// Creates a new `Returns` from a matrix of asset returns (rows: time, columns:
    /// assets), with the `nalgebra` feature. Otherwise it's the same as
    /// [`Returns::new`].
    ///
    /// The feature tracks the nalgebra version cutup depends on, so a nalgebra release that
    /// isn't compatible can change which `DMatrix` this takes.
    ///
    /// # Errors
    ///
    /// As for [`Returns::new`], except [`CutupError::RaggedRows`], which a
    /// matrix can't have.
    #[cfg(feature = "nalgebra")]
    #[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
    pub fn from_matrix(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::returns_matrix(returns)
    }

    /// Builds the matrix for rows, checking that they all have the first row's length.
    fn rows_matrix<R: AsRef<[f64]>>(rows: &[R]) -> Result<DMatrix<f64>, CutupError> {
        let assets = rows.first().map_or(0, |row| row.as_ref().len());
        if let Some((row, values)) = rows
            .iter()
            .map(AsRef::as_ref)
            .enumerate()
            .find(|(_, values)| values.len() != assets)
        {
            return Err(CutupError::RaggedRows {
                row,
                expected: assets,
                actual: values.len(),
            });
        }
        Ok(DMatrix::from_row_iterator(
            rows.len(),
            assets,
            rows.iter().flat_map(|row| row.as_ref().iter().copied()),
        ))
    }

    /// Builds the matrix for flat, row-major values, checking that they fill whole rows.
    fn flat_matrix(values: &[f64], assets: usize) -> Result<DMatrix<f64>, CutupError> {
        if assets == 0 {
            return Err(CutupError::NoAssets);
        }
        if values.len() % assets != 0 {
            return Err(CutupError::RaggedRows {
                row: values.len() / assets,
                expected: assets,
                actual: values.len() % assets,
            });
        }
        Ok(DMatrix::from_row_slice(
            values.len() / assets,
            assets,
            values,
        ))
    }

    /// The `Returns` for a matrix of prices.
    pub(crate) fn prices_matrix(prices: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::validate(&prices, 3)?;
        if let Some((row, col)) = Returns::find(&prices, |p| p <= 0.0) {
            return Err(CutupError::NonPositivePrice { row, col });
        }
        Returns::build(Returns::compute_returns(&prices))
    }

    /// The `Returns` for a matrix of returns.
    pub(crate) fn returns_matrix(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::validate(&returns, 2)?;
        Returns::build(returns)
    }

    /// Labels the assets, in column order, for example with their tickers. The [`Weights`] carry
    /// the labels, and every setting that takes an [`Asset`](crate::Asset) or [`PerAsset`] values
    /// can name assets by them. Without labels, each asset's label is its column: `"0"`, `"1"`, and
    /// so on.
    ///
    /// # Errors
    ///
    /// * [`CutupError::DimensionMismatch`] if there isn't one label per asset.
    /// * [`CutupError::DuplicateAsset`] if a label is repeated.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{InverseVariance, Returns};
    ///
    /// let returns = [
    ///    [0.010, -0.020, 0.005],
    ///    [-0.004, 0.015, 0.002],
    ///    [0.007, 0.003, -0.006],
    /// ];
    ///
    /// let data = Returns::new(&returns)?.labeled(["SPY", "TLT", "GLD"])?;
    /// assert_eq!(data.labels(), ["SPY", "TLT", "GLD"]);
    ///
    /// let weights = data.allocate(&InverseVariance)?;
    /// assert_eq!(weights.weight("TLT"), Some(weights[1]));
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn labeled<L: Into<String>>(
        mut self,
        labels: impl IntoIterator<Item = L>,
    ) -> Result<Self, CutupError> {
        let labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        let n = self.returns.ncols();
        if labels.len() != n {
            return Err(CutupError::DimensionMismatch {
                field: "labels",
                expected: n,
                actual: labels.len(),
            });
        }
        let mut seen = std::collections::HashSet::with_capacity(n);
        if let Some((column, label)) = labels
            .iter()
            .enumerate()
            .find(|(_, label)| !seen.insert(label.as_str()))
        {
            return Err(CutupError::DuplicateAsset {
                column,
                asset: label.clone(),
            });
        }
        self.labels = labels.into();
        Ok(self)
    }

    /// Returns the assets' labels, in column order: those set with [`Returns::labeled`], or
    /// each asset's column.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Computes weights with `method`, any [`Allocation`]: one of this crate's methods, such as
    /// [`MinVariance`](crate::MinVariance) or [`Hrp`](crate::Hrp), or one of your own. The weights
    /// carry the data's [labels](Returns::labels).
    ///
    /// # Errors
    ///
    /// * Whatever `method` returns. Each method's type lists its errors.
    /// * [`CutupError::DimensionMismatch`] if `method` doesn't return one weight per asset.
    /// * [`CutupError::NonFiniteWeight`] if `method` returns a weight that is NaN or infinite.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MinVariance, Returns};
    ///
    /// let returns = [
    ///      [ 0.010, -0.020,  0.005],
    ///      [-0.004,  0.015,  0.002],
    ///      [ 0.007,  0.003, -0.006],
    ///      [ 0.002, -0.001,  0.004],
    /// ];
    /// let data = Returns::new(&returns)?;
    ///
    /// let long_only = data.allocate(&MinVariance::default())?;
    /// let long_short = data.allocate(&MinVariance::default().allow_short())?;
    ///
    /// assert!(long_only.iter().all(|&w| w >= 0.0));
    /// assert!((long_short.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn allocate<A: Allocation + ?Sized>(&self, method: &A) -> Result<Weights, CutupError> {
        let weights = method.allocate(self)?;
        let n = self.returns.ncols();
        if weights.len() != n {
            return Err(CutupError::DimensionMismatch {
                field: "weights",
                expected: n,
                actual: weights.len(),
            });
        }
        if let Some(column) = weights.iter().position(|w| !w.is_finite()) {
            return Err(CutupError::NonFiniteWeight {
                column,
                asset: self.labels[column].clone(),
            });
        }
        Ok(self.weights(weights.values))
    }

    /// `values` of the setting `field`, given for every asset, in column order as a vector.
    pub(crate) fn resolve(
        &self,
        field: &'static str,
        values: Option<&PerAsset>,
    ) -> Result<Option<DVector<f64>>, CutupError> {
        values
            .map(|values| {
                Ok(DVector::from_vec(values.resolve(
                    field,
                    &self.labels,
                    None,
                )?))
            })
            .transpose()
    }

    /// The error for the asset in `column` having zero variance.
    pub(crate) fn zero_variance(&self, column: usize) -> CutupError {
        CutupError::ZeroVariance {
            column,
            asset: self.labels[column].clone(),
        }
    }

    /// `values`, one per asset in column order, as weights labeled with the data's labels.
    pub(crate) fn weights(&self, values: Vec<f64>) -> Weights {
        Weights {
            values,
            labels: Arc::clone(&self.labels),
        }
    }

    /// Builds a `Returns` from returns that have already been validated, rejecting a
    /// covariance matrix that overflowed.
    fn build(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        let cov_matrix = Returns::compute_covariance_matrix(&returns);
        if cov_matrix.iter().any(|x| !x.is_finite()) {
            return Err(CutupError::NonFiniteCovariance);
        }
        let labels = asset::column_labels(returns.ncols());
        Ok(Returns {
            returns,
            cov_matrix,
            labels,
        })
    }

    /// Returns the first asset whose returns are constant, allowing for floating-point rounding:
    /// a standard deviation below `1e-10` of the asset's largest absolute return counts as zero.
    pub(crate) fn find_zero_variance(&self) -> Option<usize> {
        (0..self.cov_matrix.ncols()).find(|&i| {
            let scale = self.returns.column(i).amax();
            self.cov_matrix[(i, i)].sqrt() <= 1e-10 * scale
        })
    }

    /// Checks that `data` has at least one column, at least `min_rows` rows, and only finite
    /// values.
    fn validate(data: &DMatrix<f64>, min_rows: usize) -> Result<(), CutupError> {
        if data.ncols() == 0 {
            return Err(CutupError::NoAssets);
        }
        if data.nrows() < min_rows {
            return Err(CutupError::InsufficientData {
                required: min_rows,
                actual: data.nrows(),
            });
        }
        if let Some((row, col)) = Returns::find(data, |x| !x.is_finite()) {
            return Err(CutupError::NonFiniteValue { row, col });
        }
        Ok(())
    }

    /// Returns the `(row, col)` of the first value matching `predicate`, scanning row by row.
    fn find(data: &DMatrix<f64>, predicate: impl Fn(f64) -> bool) -> Option<(usize, usize)> {
        (0..data.nrows())
            .flat_map(|row| (0..data.ncols()).map(move |col| (row, col)))
            .find(|&(row, col)| predicate(data[(row, col)]))
    }

    /// Converts a price matrix into simple returns, `r_t = p_t / p_{t-1} - 1`.
    ///
    /// # Arguments
    ///
    /// * `prices` - A `DMatrix<f64>` of prices (rows: time, columns: assets).
    ///
    /// # Returns
    ///
    /// * A `DMatrix<f64>` with one fewer row than `prices`.
    pub(crate) fn compute_returns(prices: &DMatrix<f64>) -> DMatrix<f64> {
        let rows = prices.nrows().saturating_sub(1);
        DMatrix::from_fn(rows, prices.ncols(), |t, j| {
            prices[(t + 1, j)] / prices[(t, j)] - 1.0
        })
    }

    /// Computes the covariance matrix from asset returns.
    ///
    /// # Arguments
    ///
    /// * `returns` - A `DMatrix<f64>` representing asset returns.
    ///
    /// # Returns
    ///
    /// * A `DMatrix<f64>` representing the covariance matrix.
    fn compute_covariance_matrix(returns: &DMatrix<f64>) -> DMatrix<f64> {
        let mean = returns.row_mean();
        let mean_matrix = DMatrix::from_rows(&vec![mean.clone(); returns.nrows()]);
        let centered = returns - mean_matrix;
        (centered.transpose() * centered) / (returns.nrows() as f64 - 1.0)
    }

    /// Returns `given` after checking it, or the sample mean of the returns when it's `None`.
    pub(crate) fn expected_returns(
        &self,
        given: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, CutupError> {
        let n = self.cov_matrix.ncols();
        match given {
            Some(mu) if mu.len() != n => Err(CutupError::DimensionMismatch {
                field: "expected_returns",
                expected: n,
                actual: mu.len(),
            }),
            Some(mu) => {
                asset::check_values(
                    "expected_returns",
                    &self.labels,
                    mu.as_slice(),
                    f64::is_finite,
                )?;
                Ok(mu.clone())
            }
            None => Ok(self.returns.row_mean().transpose()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{full_rank_prices, labeled};
    use crate::{
        Asset, BlackLitterman, EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility,
        MaxDiversification, MaxSharpe, MeanVariance, MinCvar, MinVariance, RiskParity, View,
        WeightBounds,
    };
    use nalgebra::dmatrix;

    #[test]
    fn test_compute_returns() {
        let prices = dmatrix![
            100.0, 200.0;
            110.0, 190.0;
            121.0, 209.0
        ];
        let returns = Returns::compute_returns(&prices);
        let expected = dmatrix![
            0.1, -0.05;
            0.1, 0.1
        ];
        assert_eq!(returns.shape(), (2, 2));
        assert!((returns - expected).abs().max() < 1e-12);
    }

    #[test]
    fn test_from_prices_matches_new() {
        let prices = full_rank_prices();
        let returns = Returns::compute_returns(&prices);
        let from_prices = Returns::prices_matrix(prices).unwrap();
        let new = Returns::returns_matrix(returns).unwrap();
        assert_eq!(from_prices.cov_matrix, new.cov_matrix);
    }

    #[test]
    fn test_plain_inputs_match_the_matrix() {
        let rows = [
            [0.01, -0.02, 0.005],
            [-0.004, 0.015, 0.002],
            [0.007, 0.003, -0.006],
        ];
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let matrix = DMatrix::from_fn(3, 3, |t, j| rows[t][j]);
        let expected = Returns::returns_matrix(matrix.clone()).unwrap();
        let vecs: Vec<Vec<f64>> = rows.iter().map(|row| row.to_vec()).collect();
        let slices: Vec<&[f64]> = rows.iter().map(|row| row.as_slice()).collect();
        for allocator in [
            Returns::new(&rows).unwrap(),
            Returns::new(&vecs).unwrap(),
            Returns::new(&slices).unwrap(),
            Returns::new_flat(&flat, 3).unwrap(),
        ] {
            assert_eq!(allocator.returns, expected.returns);
            assert_eq!(allocator.cov_matrix, expected.cov_matrix);
        }

        let prices = [[100.0, 50.0], [102.0, 49.0], [101.0, 51.5]];
        let flat: Vec<f64> = prices.iter().flatten().copied().collect();
        let expected = Returns::prices_matrix(DMatrix::from_fn(3, 2, |t, j| prices[t][j])).unwrap();
        for allocator in [
            Returns::from_prices(&prices).unwrap(),
            Returns::from_prices_flat(&flat, 2).unwrap(),
        ] {
            assert_eq!(allocator.returns, expected.returns);
            assert_eq!(allocator.cov_matrix, expected.cov_matrix);
        }
    }

    #[test]
    #[allow(deprecated)]
    fn test_old_names_are_the_new_ones() {
        let rows = [[0.01, -0.02], [-0.004, 0.015], [0.007, 0.003]];
        let old: crate::PortfolioAllocator =
            crate::PortfolioAllocator::from_returns(&rows).unwrap();
        let new = Returns::new(&rows).unwrap();
        assert_eq!(old.returns, new.returns);
        assert_eq!(old.cov_matrix, new.cov_matrix);
    }

    #[test]
    fn test_plain_inputs_reject_bad_shapes() {
        let ragged = vec![vec![0.01, 0.02], vec![0.03], vec![0.04, 0.05]];
        let error = Some(CutupError::RaggedRows {
            row: 1,
            expected: 2,
            actual: 1,
        });
        assert_eq!(Returns::new(&ragged).err(), error);
        assert_eq!(Returns::from_prices(&ragged).err(), error);
        // Seven values don't fill rows of three: the third row has one.
        let flat = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let error = Some(CutupError::RaggedRows {
            row: 2,
            expected: 3,
            actual: 1,
        });
        assert_eq!(Returns::new_flat(&flat, 3).err(), error);
        assert_eq!(Returns::from_prices_flat(&flat, 3).err(), error);

        let none: [[f64; 2]; 0] = [];
        let empty: [[f64; 0]; 3] = [[], [], []];
        for result in [
            Returns::new(&none),
            Returns::new(&empty),
            Returns::from_prices(&none),
            Returns::new_flat(&flat, 0),
            Returns::new_flat(&[], 0),
            Returns::from_prices_flat(&flat, 0),
        ] {
            assert_eq!(result.err(), Some(CutupError::NoAssets));
        }
        // No values for a known number of assets is too few rows, not too few assets.
        assert_eq!(
            Returns::new_flat(&[], 3).err(),
            Some(CutupError::InsufficientData {
                required: 2,
                actual: 0
            })
        );
        assert_eq!(
            CutupError::RaggedRows {
                row: 1,
                expected: 2,
                actual: 1
            }
            .to_string(),
            "row 1 has the wrong number of values: 1, not 2"
        );
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_matrix_inputs_and_vector_weights() {
        let prices = dmatrix![100.0, 50.0; 102.0, 49.0; 101.0, 51.5];
        let allocator = Returns::from_prices_matrix(prices.clone()).unwrap();
        assert_eq!(
            allocator.cov_matrix,
            Returns::prices_matrix(prices).unwrap().cov_matrix
        );
        let returns = allocator.returns.clone();
        let weights = Returns::from_matrix(returns)
            .unwrap()
            .allocate(&EqualWeight)
            .unwrap();
        assert_eq!(DVector::from(weights), DVector::from_element(2, 0.5));
    }

    #[test]
    fn test_from_prices_rejects_no_assets() {
        let result = Returns::prices_matrix(DMatrix::zeros(3, 0));
        assert_eq!(result.err(), Some(CutupError::NoAssets));
    }

    #[test]
    fn test_from_prices_rejects_too_few_rows() {
        let result = Returns::prices_matrix(dmatrix![100.0, 200.0; 110.0, 210.0]);
        assert_eq!(
            result.err(),
            Some(CutupError::InsufficientData {
                required: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn test_new_rejects_too_few_rows() {
        let result = Returns::returns_matrix(dmatrix![0.01, -0.02]);
        assert_eq!(
            result.err(),
            Some(CutupError::InsufficientData {
                required: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn test_from_prices_rejects_non_finite_values() {
        let mut prices = full_rank_prices();
        prices[(2, 1)] = f64::NAN;
        prices[(4, 0)] = f64::INFINITY;
        let result = Returns::prices_matrix(prices);
        assert_eq!(
            result.err(),
            Some(CutupError::NonFiniteValue { row: 2, col: 1 })
        );
    }

    #[test]
    fn test_new_rejects_non_finite_values() {
        let result = Returns::returns_matrix(dmatrix![0.01, 0.02; f64::NEG_INFINITY, 0.0]);
        assert_eq!(
            result.err(),
            Some(CutupError::NonFiniteValue { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_from_prices_rejects_non_positive_prices() {
        let mut prices = full_rank_prices();
        prices[(3, 2)] = 0.0;
        let result = Returns::prices_matrix(prices.clone());
        assert_eq!(
            result.err(),
            Some(CutupError::NonPositivePrice { row: 3, col: 2 })
        );

        prices[(1, 0)] = -5.0;
        let result = Returns::prices_matrix(prices);
        assert_eq!(
            result.err(),
            Some(CutupError::NonPositivePrice { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_new_accepts_negative_returns() {
        let returns = dmatrix![-0.5, 0.01; 0.2, -0.03; -0.1, 0.02];
        assert!(Returns::returns_matrix(returns).is_ok());
    }

    #[test]
    fn test_constructors_reject_overflowing_covariance() {
        let prices = dmatrix![100.0, 50.0; 1e300, 51.0; 100.0, 50.5];
        assert_eq!(
            Returns::prices_matrix(prices).err(),
            Some(CutupError::NonFiniteCovariance)
        );

        let returns = dmatrix![1e200, 0.01; -1e200, 0.02];
        assert_eq!(
            Returns::returns_matrix(returns).err(),
            Some(CutupError::NonFiniteCovariance)
        );
    }

    #[test]
    fn test_rounding_noise_counts_as_zero_variance() {
        // Constant returns whose computed variance is rounding noise rather than exactly zero.
        let allocator =
            Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            allocator.allocate(&Hrp::default()).err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );

        // Prices growing at a constant 10% per period.
        let prices = dmatrix![100.0, 50.0; 110.0, 49.0; 121.0, 51.5; 133.1, 51.0];
        let allocator = Returns::prices_matrix(prices).unwrap();
        assert_eq!(
            allocator.allocate(&Hrp::default()).err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
    }

    #[test]
    fn test_single_asset() {
        let prices = dmatrix![
            100.0;
            110.0;
            120.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let ew_weights = allocator.allocate(&EqualWeight).unwrap();
        assert_eq!(ew_weights.as_slice(), &[1.0]);
    }

    #[test]
    fn test_labels_default_to_columns() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        assert_eq!(allocator.labels(), ["0", "1", "2"]);
        let weights = allocator.allocate(&InverseVariance).unwrap();
        assert_eq!(weights.labels(), ["0", "1", "2"]);
        assert_eq!(weights.weight("1"), Some(weights[1]));
        assert_eq!(weights, Weights::from(weights.as_slice().to_vec()));
    }

    #[test]
    fn test_labeled_rejects_a_wrong_count_and_repeats() {
        let allocator = || Returns::prices_matrix(full_rank_prices()).unwrap();
        assert_eq!(
            allocator().labeled(["SPY", "TLT"]).err(),
            Some(CutupError::DimensionMismatch {
                field: "labels",
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            allocator().labeled(["SPY", "TLT", "SPY"]).err(),
            Some(CutupError::DuplicateAsset {
                column: 2,
                asset: "SPY".into()
            })
        );
        let labels: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(
            allocator().labeled(labels).unwrap().labels(),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn test_every_method_labels_its_weights() {
        let allocator = labeled();
        let current = [0.4, 0.3, 0.3];
        let methods: Vec<Box<dyn Allocation>> = vec![
            Box::new(EqualWeight),
            Box::new(InverseVolatility),
            Box::new(InverseVariance),
            Box::new(MinVariance::default()),
            Box::new(MinVariance::default().allow_short()),
            Box::new(
                MinVariance::default()
                    .current_weights(current)
                    .max_turnover(0.1),
            ),
            Box::new(MeanVariance::risk_aversion(2.0)),
            Box::new(MeanVariance::risk_aversion(2.0).allow_short()),
            Box::new(
                MeanVariance::risk_aversion(2.0)
                    .current_weights(current)
                    .transaction_costs(0.001),
            ),
            Box::new(MaxSharpe::default()),
            Box::new(MaxDiversification::default()),
            Box::new(RiskParity::default()),
            Box::new(Hrp::default()),
            Box::new(Herc::default()),
            Box::new(MinCvar::default()),
        ];
        for (i, method) in methods.iter().enumerate() {
            let weights = allocator.allocate(method).unwrap();
            assert_eq!(weights.labels(), ["SPY", "TLT", "GLD"], "method {i}");
            // Called directly, not through `Returns::allocate`, too.
            assert_eq!(method.allocate(&allocator).unwrap(), weights);
        }
    }

    #[test]
    fn test_own_allocations_get_the_labels() {
        struct Fixed;
        impl Allocation for Fixed {
            fn allocate(&self, _data: &Returns) -> Result<Weights, CutupError> {
                Ok(vec![0.2, 0.3, 0.5].into())
            }
        }
        let weights = labeled().allocate(&Fixed).unwrap();
        assert_eq!(weights.labels(), ["SPY", "TLT", "GLD"]);
        assert_eq!(weights.weight("GLD"), Some(0.5));
    }

    #[test]
    fn test_assets_by_label_match_by_column() {
        let allocator = labeled();
        let same = |a: &dyn Allocation, b: &dyn Allocation| {
            assert_eq!(
                allocator.allocate(a).unwrap(),
                allocator.allocate(b).unwrap()
            );
        };

        let in_order = WeightBounds::per_asset(vec![0.1, 0.0, 0.0], vec![0.5, 0.6, 0.4]);
        let by_label = WeightBounds::per_asset(
            [("GLD", 0.0), ("SPY", 0.1), ("TLT", 0.0)],
            [("TLT", 0.6), ("SPY", 0.5), ("GLD", 0.4)],
        );
        // Each asset's bounds replace the uniform ones, and the last for an asset wins.
        let overridden = WeightBounds::uniform(0.0, 0.6)
            .asset("SPY", 0.0, 0.2)
            .asset("SPY", 0.1, 0.5)
            .asset(2, 0.0, 0.4);
        for bounds in [by_label, overridden] {
            same(
                &MinVariance::default().bounds(in_order.clone()),
                &MinVariance::default().bounds(bounds.clone()),
            );
            same(
                &MinCvar::default().bounds(in_order.clone()),
                &MinCvar::default().bounds(bounds.clone()),
            );
            same(
                &MeanVariance::risk_aversion(2.0)
                    .bounds(in_order.clone())
                    .current_weights([0.4, 0.3, 0.3])
                    .max_turnover(0.3),
                &MeanVariance::risk_aversion(2.0)
                    .bounds(bounds)
                    .current_weights([0.4, 0.3, 0.3])
                    .max_turnover(0.3),
            );
        }

        // Current weights left out are 0.
        let current = [("TLT", 0.4), ("SPY", 0.6)];
        same(
            &MinVariance::default()
                .current_weights([0.6, 0.4, 0.0])
                .max_turnover(0.2),
            &MinVariance::default()
                .current_weights(current)
                .max_turnover(0.2),
        );
        same(
            &MeanVariance::target_return(0.01)
                .current_weights([0.6, 0.4, 0.0])
                .max_turnover(0.5),
            &MeanVariance::target_return(0.01)
                .current_weights(current)
                .max_turnover(0.5),
        );

        let mu = [0.001, 0.002, 0.0015];
        let mu_by_label = [("GLD", 0.0015), ("TLT", 0.002), ("SPY", 0.001)];
        same(
            &MeanVariance::risk_aversion(2.0).expected_returns(mu),
            &MeanVariance::risk_aversion(2.0).expected_returns(mu_by_label),
        );
        same(
            &MaxSharpe::default().expected_returns(mu),
            &MaxSharpe::default().expected_returns(mu_by_label),
        );
        same(
            &MeanVariance::risk_aversion(2.0)
                .current_weights([0.4, 0.3, 0.3])
                .transaction_costs_per_asset([0.001, 0.002, 0.003]),
            &MeanVariance::risk_aversion(2.0)
                .current_weights([("GLD", 0.3), ("SPY", 0.4), ("TLT", 0.3)])
                .transaction_costs_per_asset([(2, 0.003), (0, 0.001), (1, 0.002)]),
        );
        same(
            &RiskParity::default().risk_budgets([1.0, 2.0, 3.0]),
            &RiskParity::default().risk_budgets([("TLT", 2.0), ("GLD", 3.0), ("SPY", 1.0)]),
        );
    }

    #[test]
    fn test_views_by_label_match_by_column() {
        let allocator = labeled();
        let in_order = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
            .view(View::absolute(0, 0.001))
            .view(View::relative(1, 2, 0.0005))
            .view(View::portfolio([0.5, 0.0, -0.5], 0.0002));
        let by_label = BlackLitterman::new([("TLT", 0.3), ("GLD", 0.2), ("SPY", 0.5)], 2.5)
            .view(View::absolute("SPY", 0.001))
            .view(View::relative("TLT", "GLD", 0.0005))
            .view(View::portfolio([("GLD", -0.5), ("SPY", 0.5)], 0.0002));
        assert_eq!(
            in_order.expected_returns(&allocator).unwrap(),
            by_label.expected_returns(&allocator).unwrap()
        );
        assert_eq!(
            in_order.prior_returns(&allocator).unwrap(),
            by_label.prior_returns(&allocator).unwrap()
        );
    }

    #[test]
    fn test_assets_not_in_the_data_or_given_twice() {
        let allocator = labeled();
        let model = || BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let unknown = |asset: &str| CutupError::UnknownAsset {
            asset: asset.into(),
        };
        assert_eq!(
            allocator.allocate(
                &MinVariance::default()
                    .bounds(WeightBounds::uniform(0.0, 1.0).asset("QQQ", 0.0, 0.5))
            ),
            Err(unknown("QQQ"))
        );
        assert_eq!(
            allocator.allocate(
                &MinVariance::default().bounds(WeightBounds::uniform(0.0, 1.0).asset(3, 0.0, 0.5))
            ),
            Err(CutupError::UnknownAsset {
                asset: Asset::from(3)
            })
        );
        assert_eq!(
            model()
                .view(View::absolute("QQQ", 0.001))
                .expected_returns(&allocator),
            Err(unknown("QQQ"))
        );
        // The same asset by label and by column.
        assert_eq!(
            model()
                .view(View::relative("SPY", 0, 0.001))
                .expected_returns(&allocator),
            Err(CutupError::DuplicateAsset {
                column: 0,
                asset: "SPY".into()
            })
        );
        assert_eq!(
            allocator.allocate(
                &MinVariance::default()
                    .current_weights([("SPY", 0.5), ("SPY", 0.5)])
                    .max_turnover(0.1)
            ),
            Err(CutupError::DuplicateAsset {
                column: 0,
                asset: "SPY".into()
            })
        );
        // Every asset needs an expected return.
        assert_eq!(
            allocator.allocate(
                &MeanVariance::risk_aversion(2.0)
                    .expected_returns([("SPY", 0.001), ("TLT", 0.002)])
            ),
            Err(CutupError::DimensionMismatch {
                field: "expected_returns",
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn test_errors_name_assets_by_label() {
        let returns = dmatrix![
            0.01, 0.0, -0.02;
            -0.004, 0.0, 0.015;
            0.007, 0.0, 0.003
        ];
        let allocator = Returns::returns_matrix(returns)
            .unwrap()
            .labeled(["SPY", "CASH", "TLT"])
            .unwrap();
        assert_eq!(
            allocator.allocate(&InverseVariance),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "CASH".into()
            })
        );
        let bad_current = MinVariance::default()
            .shrinkage(0.5)
            .current_weights([("SPY", 0.5), ("TLT", f64::INFINITY)])
            .max_turnover(0.1);
        assert_eq!(
            allocator.allocate(&bad_current),
            Err(CutupError::InvalidAssetValue {
                field: "current_weights",
                column: 2,
                asset: "TLT".into(),
                value: f64::INFINITY
            })
        );
        let capped = MinVariance::default()
            .shrinkage(0.5)
            .bounds(WeightBounds::uniform(0.0, 1.0).asset("TLT", 0.5, 0.4));
        assert_eq!(
            allocator.allocate(&capped),
            Err(CutupError::InvalidAssetValue {
                field: "upper_bound",
                column: 2,
                asset: "TLT".into(),
                value: 0.4
            })
        );
    }
}
