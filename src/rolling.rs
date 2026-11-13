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
#[derive(Debug, Clone, Copy)]
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
    pub fn allocate<A: Allocation>(
        self,
        method: A,
    ) -> Result<RollingAllocation<'a, A>, CutupError> {
        let periods = self.data.returns.nrows();
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

impl<A: Allocation> Iterator for RollingAllocation<'_, A> {
    type Item = Window;

    fn next(&mut self) -> Option<Window> {
        let start = self.first + self.windows.next()? * self.step;
        let periods = start..start + self.window;
        let weights = self
            .data
            .window(periods.clone())
            .and_then(|data| data.allocate(&self.method));
        Some(Window { periods, weights })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.windows.size_hint()
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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Window {
    periods: Range<usize>,
    weights: Result<Weights, CutupError>,
}

impl Window {
    /// The periods the window covers, as rows of the returns: the last is the most recent.
    pub fn periods(&self) -> Range<usize> {
        self.periods.clone()
    }

    /// The window's weights, or why the method couldn't allocate over it.
    pub fn weights(&self) -> &Result<Weights, CutupError> {
        &self.weights
    }

    /// The window's weights, or why the method couldn't allocate over it, by value.
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
