//! Assets by column or by label, and values given for each asset.

use crate::CutupError;
use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasher;
use std::sync::Arc;

/// An asset, by its column in the data, counting from 0, or by its label: anything that takes
/// an `impl Into<Asset>` takes a `usize` or a string.
///
/// Labels are set with [`Returns::labeled`](crate::Returns::labeled). Without them, each asset's
/// label is its column, so `"0"` names the first asset too.
///
/// ```
/// use cutup::{View, WeightBounds};
///
/// let by_label = View::relative("SPY", "TLT", 0.0005);
/// let by_column = View::relative(0, 1, 0.0005);
/// let capped = WeightBounds::uniform(0.0, 0.4).asset("GLD", 0.0, 0.1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asset(Key);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Column(usize),
    Label(String),
}

impl From<usize> for Asset {
    fn from(column: usize) -> Self {
        Asset(Key::Column(column))
    }
}

impl From<&str> for Asset {
    fn from(label: &str) -> Self {
        Asset(Key::Label(label.to_owned()))
    }
}

impl From<String> for Asset {
    fn from(label: String) -> Self {
        Asset(Key::Label(label))
    }
}

impl From<&String> for Asset {
    fn from(label: &String) -> Self {
        Asset(Key::Label(label.clone()))
    }
}

impl Asset {
    /// The asset's column among the assets `labels`.
    pub(crate) fn column(&self, labels: &[String]) -> Result<usize, CutupError> {
        self.find(labels.len(), |label| labels.iter().position(|l| l == label))
    }

    /// The asset's column among `n` assets, finding a label's with `position`.
    fn find(
        &self,
        n: usize,
        position: impl Fn(&str) -> Option<usize>,
    ) -> Result<usize, CutupError> {
        match &self.0 {
            Key::Column(column) if *column < n => Ok(*column),
            Key::Column(column) => Err(CutupError::UnknownAsset {
                asset: column.to_string(),
            }),
            Key::Label(label) => position(label).ok_or_else(|| CutupError::UnknownAsset {
                asset: label.clone(),
            }),
        }
    }
}

/// A value for each asset, such as an expected return or a current weight: anything that takes
/// an `impl Into<PerAsset>` takes either form.
///
/// * **In column order:** a `Vec<f64>`, an array or a slice, with one value per asset.
/// * **By asset:** pairs of an [`Asset`] and its value, as a `Vec`, an array, a slice, a
///   `HashMap` or a `BTreeMap`. They can be in any order, and each asset can appear only once.
///   Unless the setter says otherwise, every asset needs a value.
///
/// ```
/// use cutup::{MinVariance, Returns};
///
/// let data = Returns::new(&[
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
/// ])?
/// .labeled(["SPY", "TLT", "GLD"])?;
///
/// let in_order = MinVariance::default().current_weights([0.6, 0.4, 0.0]);
/// // Current weights left out are 0.
/// let by_label = MinVariance::default().current_weights([("TLT", 0.4), ("SPY", 0.6)]);
/// assert_eq!(
///     data.allocate(&in_order.max_turnover(0.2))?,
///     data.allocate(&by_label.max_turnover(0.2))?,
/// );
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PerAsset(Values);

#[derive(Debug, Clone, PartialEq)]
enum Values {
    Ordered(Vec<f64>),
    Keyed(Vec<(Asset, f64)>),
}

impl PerAsset {
    fn keyed<K: Into<Asset>>(pairs: impl IntoIterator<Item = (K, f64)>) -> Self {
        PerAsset(Values::Keyed(
            pairs.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        ))
    }

    /// The values in column order for the assets `labels`. In column order they're as given,
    /// for the caller to check their number. Pairs that leave out an asset give it `missing`,
    /// or without it are a [`CutupError::DimensionMismatch`].
    pub(crate) fn resolve(
        &self,
        labels: &[String],
        missing: Option<f64>,
    ) -> Result<Vec<f64>, CutupError> {
        let pairs = match &self.0 {
            Values::Ordered(values) => return Ok(values.clone()),
            Values::Keyed(pairs) => pairs,
        };
        let n = labels.len();
        let columns: HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, label)| (label.as_str(), i))
            .collect();
        let mut values = vec![None; n];
        for (asset, value) in pairs {
            let column = asset.find(n, |label| columns.get(label).copied())?;
            if values[column].replace(*value).is_some() {
                return Err(CutupError::DuplicateAsset {
                    asset: labels[column].clone(),
                });
            }
        }
        values
            .into_iter()
            .map(|value| {
                value.or(missing).ok_or(CutupError::DimensionMismatch {
                    expected: n,
                    actual: pairs.len(),
                })
            })
            .collect()
    }
}

impl From<Vec<f64>> for PerAsset {
    fn from(values: Vec<f64>) -> Self {
        PerAsset(Values::Ordered(values))
    }
}

impl From<&[f64]> for PerAsset {
    fn from(values: &[f64]) -> Self {
        values.to_vec().into()
    }
}

impl From<&Vec<f64>> for PerAsset {
    fn from(values: &Vec<f64>) -> Self {
        values.clone().into()
    }
}

impl<const N: usize> From<[f64; N]> for PerAsset {
    fn from(values: [f64; N]) -> Self {
        values.to_vec().into()
    }
}

impl<const N: usize> From<&[f64; N]> for PerAsset {
    fn from(values: &[f64; N]) -> Self {
        values.to_vec().into()
    }
}

impl<K: Into<Asset>> From<Vec<(K, f64)>> for PerAsset {
    fn from(pairs: Vec<(K, f64)>) -> Self {
        PerAsset::keyed(pairs)
    }
}

impl<K: Into<Asset> + Clone> From<&[(K, f64)]> for PerAsset {
    fn from(pairs: &[(K, f64)]) -> Self {
        PerAsset::keyed(pairs.iter().cloned())
    }
}

impl<K: Into<Asset>, const N: usize> From<[(K, f64); N]> for PerAsset {
    fn from(pairs: [(K, f64); N]) -> Self {
        PerAsset::keyed(pairs)
    }
}

impl<K: Into<Asset> + Clone, const N: usize> From<&[(K, f64); N]> for PerAsset {
    fn from(pairs: &[(K, f64); N]) -> Self {
        PerAsset::keyed(pairs.iter().cloned())
    }
}

impl<K: Into<Asset>, S: BuildHasher> From<HashMap<K, f64, S>> for PerAsset {
    fn from(pairs: HashMap<K, f64, S>) -> Self {
        PerAsset::keyed(pairs)
    }
}

impl<K: Into<Asset>> From<BTreeMap<K, f64>> for PerAsset {
    fn from(pairs: BTreeMap<K, f64>) -> Self {
        PerAsset::keyed(pairs)
    }
}

/// The labels of `n` assets without labels of their own: their columns.
pub(crate) fn column_labels(n: usize) -> Arc<[String]> {
    (0..n).map(|i| i.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        ["SPY", "TLT", "GLD"].map(String::from).to_vec()
    }

    #[test]
    fn assets_by_column_and_label() {
        let labels = labels();
        assert_eq!(Asset::from(2).column(&labels), Ok(2));
        assert_eq!(Asset::from("TLT").column(&labels), Ok(1));
        assert_eq!(Asset::from(String::from("GLD")).column(&labels), Ok(2));
        assert_eq!(
            Asset::from(3).column(&labels),
            Err(CutupError::UnknownAsset { asset: "3".into() })
        );
        assert_eq!(
            Asset::from("QQQ").column(&labels),
            Err(CutupError::UnknownAsset {
                asset: "QQQ".into()
            })
        );
    }

    #[test]
    fn values_in_column_order_are_as_given() {
        let labels = labels();
        // Their number is the caller's to check.
        assert_eq!(
            PerAsset::from([1.0, 2.0]).resolve(&labels, None),
            Ok(vec![1.0, 2.0])
        );
        assert_eq!(
            PerAsset::from(vec![1.0, 2.0, 3.0]).resolve(&labels, Some(0.0)),
            Ok(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn pairs_resolve_to_column_order() {
        let labels = labels();
        let expected = Ok(vec![0.1, 0.2, 0.3]);
        let pairs = [("GLD", 0.3), ("SPY", 0.1), ("TLT", 0.2)];
        assert_eq!(PerAsset::from(pairs).resolve(&labels, None), expected);
        assert_eq!(PerAsset::from(&pairs).resolve(&labels, None), expected);
        assert_eq!(PerAsset::from(&pairs[..]).resolve(&labels, None), expected);
        assert_eq!(
            PerAsset::from(pairs.to_vec()).resolve(&labels, None),
            expected
        );
        let map: HashMap<&str, f64> = pairs.into_iter().collect();
        assert_eq!(PerAsset::from(map).resolve(&labels, None), expected);
        let map: BTreeMap<String, f64> = pairs.map(|(k, v)| (k.to_owned(), v)).into();
        assert_eq!(PerAsset::from(map).resolve(&labels, None), expected);
        let columns = [(2, 0.3), (0, 0.1), (1, 0.2)];
        assert_eq!(PerAsset::from(columns).resolve(&labels, None), expected);
    }

    #[test]
    fn pairs_that_leave_out_an_asset() {
        let labels = labels();
        let partial = PerAsset::from([("TLT", 0.4)]);
        assert_eq!(partial.resolve(&labels, Some(0.0)), Ok(vec![0.0, 0.4, 0.0]));
        assert_eq!(
            partial.resolve(&labels, None),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 1
            })
        );
    }

    #[test]
    fn pairs_naming_an_asset_twice_or_one_not_in_the_data() {
        let labels = labels();
        let twice = PerAsset::from(vec![(Asset::from("SPY"), 0.5), (Asset::from(0), 0.5)]);
        assert_eq!(
            twice.resolve(&labels, Some(0.0)),
            Err(CutupError::DuplicateAsset {
                asset: "SPY".into()
            })
        );
        assert_eq!(
            PerAsset::from([("SPY", 0.5), ("QQQ", 0.5)]).resolve(&labels, Some(0.0)),
            Err(CutupError::UnknownAsset {
                asset: "QQQ".into()
            })
        );
        assert_eq!(
            PerAsset::from([(5, 1.0)]).resolve(&labels, Some(0.0)),
            Err(CutupError::UnknownAsset { asset: "5".into() })
        );
    }

    #[test]
    fn column_labels_count_from_zero() {
        assert_eq!(&*column_labels(3), ["0", "1", "2"]);
        assert!(column_labels(0).is_empty());
    }
}
