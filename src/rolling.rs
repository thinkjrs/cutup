//! Allocation over rolling windows of the data: weights over time.

use crate::{Allocation, CutupError, Returns, Weights};
use std::ops::Range;
use std::sync::Arc;

#[cfg(feature = "serde")]
mod serde_impl;

/// Rolling windows of [`Returns`], from [`Returns::rolling`]: the same number of periods each,
/// moving forward `step` periods at a time, 1 by default. The last window ends at the last period,
/// so with a step over 1 it's the first windows that leave periods out, not the most recent.
///
/// A period is a row of returns: with `Returns::from_prices`, period `t` is the return from the
/// price at row `t` to the price at row `t + 1`.
///
/// This is pure computation, not a backtest: each window is allocated on its own, from its own
/// covariance and expected returns, and none feeds its weights into the next. A method's settings,
/// such as its current weights, are the same for every window.
///
/// # Example
///
/// ```
/// use cutup::{InverseVolatility, Returns};
///
/// let returns = [
///     [ 0.010, -0.020],
///     [-0.004,  0.015],
///     [ 0.007,  0.003],
///     [ 0.002, -0.001],
///     [-0.003,  0.006],
/// ];
/// let data = Returns::new(&returns)?;
///
/// // Windows of 2, 2 apart, end at the last period, so it's period 0 that's left out.
/// let windows: Vec<_> = data.rolling(2).step(2).allocate(&InverseVolatility)?.collect();
/// assert_eq!(windows[0].periods(), 1..3);
/// assert_eq!(windows[1].periods(), 3..5);
///
/// // Each is the method on its own periods alone.
/// for window in windows {
///     let alone = Returns::new(&returns[window.periods()])?;
///     assert_eq!(window.into_weights(), alone.allocate(&InverseVolatility));
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Rolling<'a> {
    data: &'a Returns,
    window: usize,
    step: usize,
}

impl Returns {
    /// Rolling windows of `window` periods each, to allocate over with [`Rolling::allocate`].
    /// The window and the step are checked when allocating.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{InverseVolatility, Returns};
    ///
    /// let returns = [
    ///     [ 0.010, -0.020],
    ///     [-0.004,  0.015],
    ///     [ 0.007,  0.003],
    ///     [ 0.002, -0.001],
    ///     [-0.003,  0.006],
    /// ];
    /// let data = Returns::new(&returns)?.labeled(["SPY", "TLT"])?;
    ///
    /// // Three windows of three periods: 0..3, 1..4 and 2..5.
    /// let windows: Vec<_> = data.rolling(3).allocate(&InverseVolatility)?.collect();
    /// assert_eq!(windows.len(), 3);
    /// assert_eq!(windows[2].periods(), 2..5);
    /// let latest = windows[2].weights().as_ref().unwrap();
    /// assert_eq!(latest.labels(), ["SPY", "TLT"]);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn rolling(&self, window: usize) -> Rolling<'_> {
        Rolling {
            data: self,
            window,
            step: 1,
        }
    }

    /// The returns of `periods` alone, with the same labels.
    fn window(&self, periods: Range<usize>) -> Result<Returns, CutupError> {
        let rows = self.returns.rows(periods.start, periods.len()).into_owned();
        let mut data = Returns::returns_matrix(rows)?;
        data.labels = Arc::clone(&self.labels);
        Ok(data)
    }
}

impl<'a> Rolling<'a> {
    /// Moves each window forward `step` periods from the one before. Defaults to 1.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{EqualWeight, Returns};
    ///
    /// # let returns = [[0.010, -0.020], [-0.004, 0.015], [0.007, 0.003], [0.002, -0.001], [-0.003, 0.006]];
    /// let data = Returns::new(&returns)?;
    /// let periods = |step| -> Result<Vec<_>, cutup::CutupError> {
    ///     Ok(data.rolling(3).step(step).allocate(EqualWeight)?.map(|w| w.periods()).collect())
    /// };
    ///
    /// assert_eq!(periods(1)?, [0..3, 1..4, 2..5]);
    /// assert_eq!(periods(2)?, [0..3, 2..5]);
    /// // With no room for a second window, the one there is ends at the last period.
    /// assert_eq!(periods(3)?, [2..5]);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Allocates with `method`, by value or by reference, over each window, oldest first. Each [`Window`] has its own
    /// result, so a window the method can't allocate, such as one where an asset's returns don't
    /// vary, doesn't stop the others. Collect into a `Result` to stop at the first that fails
    /// instead.
    ///
    /// The windows are allocated as the iterator reaches them.
    ///
    /// # Errors
    ///
    /// [`CutupError::InvalidConfig`] if the window is under 2 periods or longer than the data, or
    /// the step is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CutupError, MinVariance, Returns};
    ///
    /// // The first asset's returns don't vary over the first 3 periods.
    /// let returns = [
    ///     [ 0.001, -0.020],
    ///     [ 0.001,  0.015],
    ///     [ 0.001,  0.003],
    ///     [ 0.002, -0.001],
    ///     [-0.003,  0.006],
    /// ];
    /// let data = Returns::new(&returns)?;
    /// let method = MinVariance::default();
    ///
    /// // Only the first window fails; the others still allocate.
    /// let windows: Vec<_> = data.rolling(3).allocate(&method)?.collect();
    /// assert!(matches!(windows[0].weights(), Err(CutupError::ZeroVariance { column: 0, .. })));
    /// assert!(windows[1].weights().is_ok() && windows[2].weights().is_ok());
    ///
    /// // Collected into a `Result`, the first failure is the result.
    /// let windows = data.rolling(3).allocate(&method)?;
    /// let all: Result<Vec<_>, _> = windows.map(|w| w.into_weights()).collect();
    /// assert!(matches!(all, Err(CutupError::ZeroVariance { .. })));
    ///
    /// // A window must have 2 periods and fit in the data, and the step must move.
    /// assert!(data.rolling(1).allocate(&method).is_err());
    /// assert!(data.rolling(6).allocate(&method).is_err());
    /// assert!(data.rolling(3).step(0).allocate(&method).is_err());
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn allocate<A: Allocation>(
        self,
        method: A,
    ) -> Result<RollingAllocation<'a, A>, CutupError> {
        let periods = self.data.period_count();
        if self.window < 2 || self.window > periods {
            return Err(CutupError::InvalidConfig {
                field: "window",
                value: self.window as f64,
            });
        }
        if self.step == 0 {
            return Err(CutupError::InvalidConfig {
                field: "step",
                value: 0.0,
            });
        }
        let count = (periods - self.window) / self.step + 1;
        // The first window starts where the last, ending at the last period, leaves room for
        // the others before it.
        let first = periods - self.window - (count - 1) * self.step;
        Ok(RollingAllocation {
            data: self.data,
            method,
            window: self.window,
            step: self.step,
            first,
            windows: 0..count,
        })
    }
}

/// The weights over each rolling window, oldest first, from [`Rolling::allocate`].
///
/// # Example
///
/// ```
/// use cutup::{EqualWeight, Returns};
///
/// # let returns = [[0.010, -0.020], [-0.004, 0.015], [0.007, 0.003], [0.002, -0.001], [-0.003, 0.006]];
/// let data = Returns::new(&returns)?;
///
/// // The number of windows is known before any is allocated.
/// let mut windows = data.rolling(2).allocate(EqualWeight)?;
/// assert_eq!(windows.len(), 4);
///
/// // Newest first, from the back.
/// assert_eq!(windows.next_back().map(|w| w.periods()), Some(3..5));
/// assert_eq!(windows.next().map(|w| w.periods()), Some(0..2));
/// assert_eq!(windows.len(), 2);
/// # Ok::<(), cutup::CutupError>(())
/// ```
pub struct RollingAllocation<'a, A> {
    data: &'a Returns,
    method: A,
    window: usize,
    step: usize,
    /// The first window's start.
    first: usize,
    /// The windows left, by their place in order.
    windows: Range<usize>,
}

impl<A: Allocation> RollingAllocation<'_, A> {
    /// The window `k`th in order, allocated.
    fn window(&self, k: usize) -> Window {
        let start = self.first + k * self.step;
        let periods = start..start + self.window;
        let weights = self
            .data
            .window(periods.clone())
            .and_then(|data| data.allocate(&self.method));
        Window { periods, weights }
    }
}

/// Skipping windows, with `nth`, `skip`, `last` or from the back, allocates only those reached,
/// and counting them allocates none.
impl<A: Allocation> Iterator for RollingAllocation<'_, A> {
    type Item = Window;

    fn next(&mut self) -> Option<Window> {
        let k = self.windows.next()?;
        Some(self.window(k))
    }

    fn count(self) -> usize {
        self.windows.len()
    }

    fn nth(&mut self, n: usize) -> Option<Window> {
        let k = self.windows.nth(n)?;
        Some(self.window(k))
    }

    fn last(mut self) -> Option<Window> {
        self.next_back()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.windows.size_hint()
    }
}

impl<A: Allocation> DoubleEndedIterator for RollingAllocation<'_, A> {
    fn next_back(&mut self) -> Option<Window> {
        let k = self.windows.next_back()?;
        Some(self.window(k))
    }

    fn nth_back(&mut self, n: usize) -> Option<Window> {
        let k = self.windows.nth_back(n)?;
        Some(self.window(k))
    }
}

impl<A: Allocation> ExactSizeIterator for RollingAllocation<'_, A> {}

impl<A> std::fmt::Debug for RollingAllocation<'_, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RollingAllocation")
            .field("window", &self.window)
            .field("step", &self.step)
            .field("remaining", &self.windows.len())
            .finish_non_exhaustive()
    }
}

/// One window's weights, from [`Rolling::allocate`].
///
/// # Example
///
/// ```
/// use cutup::{InverseVariance, Returns};
///
/// # let returns = [[0.010, -0.020], [-0.004, 0.015], [0.007, 0.003], [0.002, -0.001], [-0.003, 0.006]];
/// let data = Returns::new(&returns)?.labeled(["x", "y"])?;
///
/// // The most recent window: the last 3 periods, and its weights over them.
/// let latest = data.rolling(3).allocate(InverseVariance)?.last().unwrap();
/// assert_eq!(latest.periods(), 2..5);
/// let recent = Returns::new(&returns[2..])?.labeled(["x", "y"])?;
/// assert_eq!(latest.into_weights()?, recent.allocate(&InverseVariance)?);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "ts-rs",
    derive(ts_rs::TS),
    ts(as = "serde_impl::Wire<'static>")
)]
#[non_exhaustive]
pub struct Window {
    periods: Range<usize>,
    weights: Result<Weights, CutupError>,
}

impl Window {
    /// The periods the window covers, as rows of the returns: the last is the most recent.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{EqualWeight, Returns};
    ///
    /// # let returns = [[0.010, -0.020], [-0.004, 0.015], [0.007, 0.003], [0.002, -0.001], [-0.003, 0.006]];
    /// let data = Returns::new(&returns)?;
    /// let windows: Vec<_> = data.rolling(4).allocate(EqualWeight)?.collect();
    ///
    /// assert_eq!(windows[0].periods(), 0..4);
    /// // The newest window ends at the last period.
    /// assert_eq!(windows[1].periods(), 1..5);
    /// assert_eq!(windows[1].periods().end, data.period_count());
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    #[must_use]
    pub fn periods(&self) -> Range<usize> {
        self.periods.clone()
    }

    /// The window's weights, or why the method couldn't allocate over it.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{InverseVolatility, Returns};
    ///
    /// // The second asset's returns don't vary over the last 2 periods.
    /// let returns = [[0.010, -0.020], [-0.004, 0.015], [0.007, 0.003], [0.002, 0.003]];
    /// let data = Returns::new(&returns)?;
    ///
    /// for window in data.rolling(2).allocate(&InverseVolatility)? {
    ///     let alone = Returns::new(&returns[window.periods()])?.allocate(&InverseVolatility);
    ///     assert_eq!(window.weights(), &alone);
    ///     // Only the last window can't be allocated.
    ///     assert_eq!(window.weights().is_err(), window.periods() == (2..4));
    /// }
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    // With a reason: clippy before 1.93 takes a bare attribute on a function that returns a
    // reference to a `Result` as said twice, and clippy since asks for one.
    #[must_use = "reading a window's weights does nothing by itself"]
    pub fn weights(&self) -> &Result<Weights, CutupError> {
        &self.weights
    }

    /// The window's weights, or why the method couldn't allocate over it, by value.
    ///
    /// # Errors
    ///
    /// The error the method returned for this window's returns, if it couldn't allocate them.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CutupError, MinVariance, Returns, Weights};
    ///
    /// # let returns = [[0.010, -0.020], [-0.004, 0.015], [0.007, 0.003], [0.002, -0.001], [-0.003, 0.006]];
    /// let data = Returns::new(&returns)?;
    ///
    /// // Every window's weights, or the first error.
    /// let history: Vec<Weights> = data
    ///     .rolling(3)
    ///     .allocate(MinVariance::default())?
    ///     .map(|w| w.into_weights())
    ///     .collect::<Result<_, CutupError>>()?;
    /// assert_eq!(history.len(), 3);
    /// assert_eq!(history[2], Returns::new(&returns[2..])?.allocate(&MinVariance::default())?);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn into_weights(self) -> Result<Weights, CutupError> {
        self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeanVariance, MinVariance};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    /// 12 periods of 3 assets' returns, varied enough for every window to have a covariance.
    fn returns() -> Vec<[f64; 3]> {
        (0..12)
            .map(|t| {
                let t = t as f64;
                [
                    0.01 * (0.7 * t).sin(),
                    0.02 * (1.3 * t + 1.0).cos(),
                    0.015 * (0.4 * t + 2.0).sin(),
                ]
            })
            .collect()
    }

    fn data() -> Returns {
        Returns::new(&returns())
            .unwrap()
            .labeled(["A", "B", "C"])
            .unwrap()
    }

    #[test]
    fn each_window_is_the_method_on_its_own_periods() {
        // 12 periods in windows of 5, 2 apart, ending at the last: 1..6, 3..8, 5..10 and 7..12.
        let data = data();
        let methods: [&dyn Allocation; 2] =
            [&MinVariance::default(), &MeanVariance::risk_aversion(10.0)];
        for method in methods {
            let windows: Vec<Window> = data.rolling(5).step(2).allocate(method).unwrap().collect();
            let periods: Vec<Range<usize>> = windows.iter().map(Window::periods).collect();
            assert_eq!(periods, [1..6, 3..8, 5..10, 7..12]);
            for window in windows {
                let alone = Returns::new(&returns()[window.periods()])
                    .unwrap()
                    .labeled(["A", "B", "C"])
                    .unwrap();
                assert_eq!(window.into_weights(), alone.allocate(method));
            }
        }
    }

    #[test]
    fn a_window_as_long_as_the_data_is_one_window() {
        let data = data();
        let method = MinVariance::default();
        let mut windows = data.rolling(12).step(3).allocate(&method).unwrap();
        assert_eq!(windows.len(), 1);
        let only = windows.next().unwrap();
        assert_eq!(only.periods(), 0..12);
        assert_eq!(only.into_weights(), data.allocate(&method));
        assert!(windows.next().is_none());
    }

    #[test]
    fn the_count_and_size_are_known_up_front() {
        // (12 − 4) / 3 + 1 = 3 windows, the first starting at 12 − 4 − 2·3 = 2.
        let data = data();
        let mut windows = data
            .rolling(4)
            .step(3)
            .allocate(MinVariance::default())
            .unwrap();
        assert_eq!(windows.size_hint(), (3, Some(3)));
        assert_eq!(windows.next().unwrap().periods(), 2..6);
        assert_eq!(windows.len(), 2);
        assert_eq!(
            format!("{windows:?}"),
            "RollingAllocation { window: 4, step: 3, remaining: 2, .. }"
        );
    }

    #[test]
    fn skipping_windows_allocates_only_those_reached() {
        /// Equal weights, counting how many windows it's asked for.
        struct Counting(std::cell::Cell<usize>);

        impl Allocation for Counting {
            fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
                self.0.set(self.0.get() + 1);
                crate::EqualWeight.allocate(data)
            }
        }

        // 11 windows of 2 over 12 periods: 0..2 up to 10..12.
        let data = data();
        let method = Counting(std::cell::Cell::new(0));
        let mut windows = data.rolling(2).allocate(&method).unwrap();
        assert_eq!(windows.nth(5).unwrap().periods(), 5..7);
        assert_eq!(windows.next_back().unwrap().periods(), 10..12);
        assert_eq!(windows.nth_back(1).unwrap().periods(), 8..10);
        assert_eq!(windows.len(), 2);
        assert_eq!(method.0.get(), 3);
        let all = data.rolling(2).allocate(&method).unwrap();
        assert_eq!(all.last().unwrap().periods(), 10..12);
        assert_eq!(method.0.get(), 4);
        // Counting the windows, or those left, allocates none of them.
        let mut windows = data.rolling(2).allocate(&method).unwrap();
        assert_eq!(windows.nth(2).unwrap().periods(), 2..4);
        assert_eq!(windows.count(), 8);
        assert_eq!(method.0.get(), 5);
        let backward: Vec<_> = data
            .rolling(5)
            .step(4)
            .allocate(&method)
            .unwrap()
            .rev()
            .map(|w| w.periods())
            .collect();
        assert_eq!(backward, [7..12, 3..8]);
    }

    #[test]
    fn a_window_or_step_out_of_range_is_rejected() {
        let data = data();
        let method = MinVariance::default();
        for (window, step, field, value) in [
            (1, 1, "window", 1.0),
            (13, 1, "window", 13.0),
            (4, 0, "step", 0.0),
        ] {
            assert_eq!(
                data.rolling(window).step(step).allocate(&method).err(),
                Some(CutupError::InvalidConfig { field, value })
            );
        }
        assert!(data.rolling(2).allocate(&method).is_ok());
    }

    #[test]
    fn a_window_that_fails_leaves_the_others() {
        // A's returns are constant over the first 4 periods, so the first window of 4 has no
        // variance for it, and the later ones do.
        let mut returns = returns();
        for row in &mut returns[..4] {
            row[0] = 0.001;
        }
        let data = Returns::new(&returns)
            .unwrap()
            .labeled(["A", "B", "C"])
            .unwrap();
        let windows: Vec<Window> = data
            .rolling(4)
            .allocate(&MinVariance::default())
            .unwrap()
            .collect();
        assert_eq!(windows.len(), 9);
        assert_eq!(
            windows[0].weights(),
            &Err(CutupError::ZeroVariance {
                column: 0,
                asset: "A".to_string()
            })
        );
        for window in &windows[1..] {
            let weights = window.weights().as_ref().unwrap();
            assert_eq!(weights.labels(), ["A", "B", "C"]);
        }
    }
}
