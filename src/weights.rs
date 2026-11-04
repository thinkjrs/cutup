//! [`Weights`], what every allocation method returns.

use crate::CutupError;
use crate::asset;
#[cfg(feature = "nalgebra")]
use nalgebra::DVector;
use std::fmt;
use std::ops::Index;
use std::sync::Arc;

/// Portfolio weights, one per asset, in the order of the columns of the
/// [`Returns`](crate::Returns), and labeled with their assets.
///
/// Like a slice, it indexes and iterates by column. [`Weights::weight`] and [`Weights::labeled`]
/// go by label, which is the asset's column unless the data was
/// [labeled](crate::Returns::labeled).
///
/// # Example
///
/// ```
/// use cutup::{EqualWeight, Returns};
///
/// let prices = [
///    [100.0, 200.0],
///    [110.0, 190.0],
///    [121.0, 209.0],
/// ];
///
/// let weights = Returns::from_prices(&prices)?.allocate(&EqualWeight)?;
///
/// assert_eq!(weights.len(), 2);
/// assert_eq!(weights[0], 0.5);
/// assert_eq!(weights.get(1), Some(0.5));
/// assert_eq!(weights.iter().sum::<f64>(), 1.0);
///
/// let labeled = Returns::from_prices(&prices)?.labeled(["SPY", "TLT"])?.allocate(&EqualWeight)?;
/// assert_eq!(labeled.weight("TLT"), Some(0.5));
/// assert_eq!(labeled["SPY"], 0.5);
/// for (label, weight) in labeled.labeled() {
///     println!("{label}: {weight}");
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    values: Vec<f64>,
    labels: Arc<[String]>,
}

impl Weights {
    /// `values` labeled with `labels`, one per value.
    pub(crate) fn new(values: Vec<f64>, labels: Arc<[String]>) -> Self {
        debug_assert_eq!(values.len(), labels.len());
        Weights { values, labels }
    }

    /// Returns the weight of the asset in column `asset`, or `None` if out of range.
    pub fn get(&self, asset: usize) -> Option<f64> {
        self.values.get(asset).copied()
    }

    /// Returns the weight of the asset labeled `label`, or `None` if no asset has that label.
    pub fn weight(&self, label: &str) -> Option<f64> {
        self.position(label).map(|i| self.values[i])
    }

    /// Returns the number of assets.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if there are no weights.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterates over the weights in column order.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.values.iter()
    }

    /// Iterates over each asset's label and weight, in column order.
    pub fn labeled(&self) -> impl DoubleEndedIterator<Item = (&str, f64)> + ExactSizeIterator {
        self.labels
            .iter()
            .map(String::as_str)
            .zip(self.values.iter().copied())
    }

    /// Returns the assets' labels, in column order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the weights as a slice, in column order.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Consumes the weights and returns them as a `Vec`, in column order.
    pub fn into_vec(self) -> Vec<f64> {
        self.values
    }

    /// Returns these weights with every weight smaller in size than `cutoff` set to 0, and the
    /// rest rescaled to sum to 1, so a result doesn't carry weights like 0.0003. A weight of
    /// exactly `cutoff` is kept, and a cutoff of 0 only rescales.
    ///
    /// Unlike PyPortfolioOpt's `clean_weights`, which only sets them to 0, this rescales the rest,
    /// so the weights still sum to 1 and can be passed back as
    /// [current weights](crate::MinVariance::current_weights). Rescaling moves every kept weight,
    /// so they're no longer exactly the optimum: a weight at its upper bound, or held to a turnover
    /// cap, can end up just past it. Dropping long positions scales the rest up; dropping short
    /// ones scales them down, which can leave a kept weight below `cutoff`.
    ///
    /// # Errors
    ///
    /// [`CutupError::InvalidConfig`] for the `cutoff` if it's negative or not finite, or if the
    /// weights it keeps don't sum to a positive, finite number: it drops every weight, or the
    /// short positions it keeps outweigh the long ones.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Weights;
    ///
    /// let weights = Weights::from(vec![0.6, 0.3, 0.0997, 0.0003]);
    /// let cleaned = weights.clean(0.001)?;
    ///
    /// assert_eq!(cleaned[3], 0.0);
    /// assert!((cleaned[0] - 0.6 / 0.9997).abs() < 1e-15);
    /// assert!((cleaned.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn clean(&self, cutoff: f64) -> Result<Weights, CutupError> {
        let invalid = CutupError::InvalidConfig {
            field: "cutoff",
            value: cutoff,
        };
        if !(cutoff >= 0.0 && cutoff.is_finite()) {
            return Err(invalid);
        }
        let mut kept: Vec<f64> = self
            .values
            .iter()
            .map(|&w| if w.abs() < cutoff { 0.0 } else { w })
            .collect();
        // Weights large enough that their sum overflows are first divided by the largest, as
        // the methods do: dividing them by an infinite sum would make each one 0. Only a
        // nonzero weight can overflow, so the largest isn't 0.
        let largest = kept.iter().fold(0.0f64, |m, w| m.max(w.abs()));
        if !kept.iter().sum::<f64>().is_finite() && largest.is_finite() {
            kept.iter_mut().for_each(|w| *w /= largest);
        }
        let total: f64 = kept.iter().sum();
        if !(total > 0.0 && total.is_finite()) {
            return Err(invalid);
        }
        kept.iter_mut().for_each(|w| *w /= total);
        Ok(Weights {
            values: kept,
            labels: Arc::clone(&self.labels),
        })
    }

    fn position(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }
}

impl Index<usize> for Weights {
    type Output = f64;

    fn index(&self, asset: usize) -> &f64 {
        &self.values[asset]
    }
}

/// The weight of the asset labeled with the index.
///
/// # Panics
///
/// If no asset has that label. [`Weights::weight`] returns `None` instead.
impl Index<&str> for Weights {
    type Output = f64;

    fn index(&self, label: &str) -> &f64 {
        match self.position(label) {
            Some(i) => &self.values[i],
            None => panic!("no asset is labeled `{label}`"),
        }
    }
}

impl<'a> IntoIterator for &'a Weights {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

/// The weights in column order, by value.
impl IntoIterator for Weights {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

/// A table of each asset's label and weight, one per line in column order, with the labels
/// padded to the same width and the weights right-aligned, so negative weights line up too.
/// Weights have 4 decimals, or the precision given, as in `{:.2}`.
///
/// ```
/// use cutup::Weights;
///
/// let weights = Weights::from(vec![0.6, 0.3, 0.1]);
/// assert_eq!(weights.to_string(), "0  0.6000\n1  0.3000\n2  0.1000");
/// assert_eq!(format!("{weights:.1}"), "0  0.6\n1  0.3\n2  0.1");
/// ```
impl fmt::Display for Weights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(4);
        let width = self
            .labels
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0);
        let values: Vec<String> = self
            .values
            .iter()
            .map(|w| format!("{w:.precision$}"))
            .collect();
        let value_width = values.iter().map(String::len).max().unwrap_or(0);
        for (i, (label, value)) in self.labels.iter().zip(&values).enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{label:<width$}  {value:>value_width$}")?;
        }
        Ok(())
    }
}

impl From<Weights> for Vec<f64> {
    fn from(weights: Weights) -> Self {
        weights.values
    }
}

/// The weights as a column vector, with the `nalgebra` feature.
#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
impl From<Weights> for DVector<f64> {
    fn from(weights: Weights) -> Self {
        DVector::from_vec(weights.values)
    }
}

/// Weights in column order, for an [`Allocation`](crate::Allocation) of your own.
/// [`Returns::allocate`](crate::Returns::allocate) checks there's one finite weight per asset, but
/// not the sum, so make them sum to 1, as every method in this crate does. It labels them with the
/// data's labels; until then each is labeled with its column.
impl From<Vec<f64>> for Weights {
    fn from(values: Vec<f64>) -> Self {
        let labels = asset::column_labels(values.len());
        Weights { values, labels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::labeled;
    use crate::{EqualWeight, InverseVariance};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_weights_accessors() {
        let weights = Weights::from(vec![0.2, 0.3, 0.5]);
        assert_eq!(weights.len(), 3);
        assert!(!weights.is_empty());
        assert_eq!(weights[1], 0.3);
        assert_eq!(weights.get(2), Some(0.5));
        assert_eq!(weights.get(3), None);
        assert_eq!(weights.as_slice(), &[0.2, 0.3, 0.5]);
        assert_eq!((&weights).into_iter().count(), 3);

        assert_eq!(weights.clone().into_vec(), vec![0.2, 0.3, 0.5]);
        assert_eq!(Vec::from(weights), vec![0.2, 0.3, 0.5]);
        assert!(Weights::from(Vec::new()).is_empty());
    }

    #[test]
    fn test_weights_by_label() {
        let weights = labeled().allocate(&InverseVariance).unwrap();
        assert_eq!(weights.weight("TLT"), Some(weights[1]));
        assert_eq!(weights.weight("QQQ"), None);
        assert_eq!(weights["GLD"], weights[2]);
        let pairs: Vec<(&str, f64)> = weights.labeled().collect();
        assert_eq!(
            pairs,
            [
                ("SPY", weights[0]),
                ("TLT", weights[1]),
                ("GLD", weights[2])
            ]
        );
        assert_eq!(weights.labeled().len(), 3);
        assert_eq!(weights.labeled().next_back(), Some(("GLD", weights[2])));
    }

    #[test]
    fn test_clean_drops_small_weights_and_rescales_the_rest() {
        let weights = labeled().allocate(&InverseVariance).unwrap();
        let smallest = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let cleaned = weights.clean(smallest * 1.000001).unwrap();
        assert_eq!(cleaned.labels(), weights.labels());
        let kept = 1.0 - smallest;
        for (c, w) in cleaned.iter().zip(&weights) {
            let expected = if *w == smallest { 0.0 } else { w / kept };
            assert!((c - expected).abs() < 1e-15, "{cleaned:?}");
        }
        assert!((cleaned.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_clean_keeps_a_weight_of_exactly_the_cutoff() {
        let weights = Weights::from(vec![0.5, 0.25, 0.25]);
        assert_eq!(weights.clean(0.25).unwrap(), weights);
        let cleaned = weights.clean(0.25 + 1e-12).unwrap();
        assert_eq!(cleaned.as_slice(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clean_with_no_cutoff_only_rescales() {
        let weights = Weights::from(vec![0.5, 1.5, 0.0]);
        assert_eq!(weights.clean(0.0).unwrap().as_slice(), &[0.25, 0.75, 0.0]);
    }

    #[test]
    fn test_clean_long_short() {
        // Dropping a long scales the rest up.
        let cleaned = Weights::from(vec![1.2, -0.3, 0.1]).clean(0.2).unwrap();
        for (c, e) in cleaned.iter().zip([1.2 / 0.9, -0.3 / 0.9, 0.0]) {
            assert!((c - e).abs() < 1e-15, "{cleaned:?}");
        }
        // Dropping a short scales them down, here leaving 0.4 below the cutoff.
        let cleaned = Weights::from(vec![0.7, 0.4, -0.1]).clean(0.38).unwrap();
        for (c, e) in cleaned.iter().zip([0.7 / 1.1, 0.4 / 1.1, 0.0]) {
            assert!((c - e).abs() < 1e-15, "{cleaned:?}");
        }
        assert!(cleaned[1] < 0.38);
        // Without its longs, what's left is short.
        assert!(Weights::from(vec![0.6, 0.6, -0.2]).clean(0.7).is_err());
    }

    #[test]
    fn test_clean_rejects_invalid_cutoffs() {
        let weights = Weights::from(vec![0.5, 0.5]);
        for cutoff in [-1e-9, f64::NAN, f64::INFINITY] {
            match weights.clean(cutoff) {
                Err(CutupError::InvalidConfig { field, value }) => {
                    assert_eq!(field, "cutoff");
                    assert!(value.to_bits() == cutoff.to_bits());
                }
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn test_clean_rejects_dropping_every_weight() {
        let invalid = |value| {
            Err(CutupError::InvalidConfig {
                field: "cutoff",
                value,
            })
        };
        assert_eq!(Weights::from(vec![0.5, 0.5]).clean(0.6), invalid(0.6));
        // Weights that sum to 0 can't be rescaled to 1 either.
        assert_eq!(Weights::from(vec![0.5, -0.5]).clean(0.0), invalid(0.0));
        assert_eq!(Weights::from(Vec::new()).clean(0.0), invalid(0.0));
    }

    #[test]
    fn test_clean_rejects_weights_that_are_not_finite() {
        assert!(Weights::from(vec![0.5, f64::NAN]).clean(0.1).is_err());
        assert!(Weights::from(vec![0.5, f64::INFINITY]).clean(0.1).is_err());
    }

    #[test]
    fn test_clean_weights_whose_sum_overflows() {
        let weights = Weights::from(vec![f64::MAX, f64::MAX, -f64::MAX, 1.0]);
        let cleaned = weights.clean(2.0).unwrap();
        assert_eq!(cleaned.as_slice(), &[1.0, 1.0, -1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "no asset is labeled `QQQ`")]
    fn test_weights_index_by_unknown_label_panics() {
        let _ = labeled().allocate(&EqualWeight).unwrap()["QQQ"];
    }
}
