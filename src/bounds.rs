//! Weight bounds, and the expected returns a portfolio within them can reach.

use crate::{Asset, CutupError, PerAsset, asset, lp, qp};
use nalgebra::{DMatrix, DVector};

/// Lower and upper limits on each asset's weight, for
/// [`MinVariance::bounds`](crate::MinVariance::bounds),
/// [`MeanVariance::bounds`](crate::MeanVariance::bounds) and
/// [`MinCvar::bounds`](crate::MinCvar::bounds).
///
/// Lower bounds must be finite and nonnegative, and each upper bound at least its lower bound.
/// An upper bound of `f64::INFINITY` means no cap. The default, `0 ≤ w` with no cap, is the
/// plain long-only constraint.
///
/// Caps that sum to just under 1, or floors to just over, by at most `1e-12`, are taken as
/// rounding: ten caps of 0.1 sum to 0.9999999999999999. No portfolio meets such bounds
/// exactly, so they're rescaled to sum to 1, which moves each by at most `1e-12` of itself.
///
/// [Groups](WeightBounds::group) limit the total weight of several assets at once, such as
/// every ad channel on one platform, or every signal from one source.
///
/// ```
/// use cutup::WeightBounds;
///
/// // No asset above 30%.
/// let capped = WeightBounds::uniform(0.0, 0.3);
///
/// // At least 5% in each of three assets, and at most 50% in the first.
/// let mixed = WeightBounds::per_asset(vec![0.05; 3], vec![0.5, 1.0, 1.0]);
///
/// // No asset above 40%, except between 10% and 60% in TLT.
/// let labeled = WeightBounds::uniform(0.0, 0.4).asset("TLT", 0.1, 0.6);
///
/// // No channel above 30%, search at most 40% together, and the last two columns at least 20%.
/// let grouped = WeightBounds::uniform(0.0, 0.3)
///     .group(["search", "shopping"], 0.0, 0.4)
///     .group([5, 6], 0.2, 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WeightBounds {
    lower: Limit,
    upper: Limit,
    assets: Vec<(Asset, f64, f64)>,
    groups: Vec<(Vec<Asset>, f64, f64)>,
}

/// One side of [`WeightBounds`].
#[derive(Debug, Clone, PartialEq)]
enum Limit {
    Uniform(f64),
    PerAsset(PerAsset),
}

impl Limit {
    fn resolve(&self, field: &'static str, labels: &[String]) -> Result<DVector<f64>, CutupError> {
        let n = labels.len();
        match self {
            Limit::Uniform(value) => Ok(DVector::from_element(n, *value)),
            Limit::PerAsset(values) => {
                let values = values.resolve(field, labels, None)?;
                Ok(DVector::from_vec(values))
            }
        }
    }
}

impl Default for WeightBounds {
    fn default() -> Self {
        Self::uniform(0.0, f64::INFINITY)
    }
}

impl WeightBounds {
    /// The same bounds `lower ≤ wᵢ ≤ upper` for every asset.
    pub fn uniform(lower: f64, upper: f64) -> Self {
        Self {
            lower: Limit::Uniform(lower),
            upper: Limit::Uniform(upper),
            assets: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Bounds `lower[i] ≤ wᵢ ≤ upper[i]` for each asset `i`: [`PerAsset`] values, in column
    /// order or by asset, with a value for every asset.
    pub fn per_asset(lower: impl Into<PerAsset>, upper: impl Into<PerAsset>) -> Self {
        Self {
            lower: Limit::PerAsset(lower.into()),
            upper: Limit::PerAsset(upper.into()),
            assets: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Bounds `lower ≤ w ≤ upper` for `asset`, by label or by column, in place of its other
    /// bounds. Setting the same asset again replaces them.
    pub fn asset(mut self, asset: impl Into<Asset>, lower: f64, upper: f64) -> Self {
        self.assets.push((asset.into(), lower, upper));
        self
    }

    /// Limits `lower ≤ Σᵢ wᵢ ≤ upper` on the total weight of the assets `members`, by label or
    /// by column, on top of each asset's own bounds. Groups may overlap, and an asset listed
    /// twice in one counts once. An upper limit of `f64::INFINITY` means no cap.
    ///
    /// Like the bounds, a group's limits within `1e-12` of what its members can hold together
    /// are taken as rounding. A group holds to the rounding of its sum, as the weights sum to
    /// 1 to theirs.
    ///
    /// Groups are supported by [`MinVariance`](crate::MinVariance), by every
    /// [`MeanVariance`](crate::MeanVariance) objective, with or without rebalancing, and by
    /// [`Returns::frontier`](crate::Returns::frontier). [`MinCvar`](crate::MinCvar) returns
    /// [`CutupError::NotSupported`] with groups.
    ///
    /// # Errors
    ///
    /// When the bounds are used:
    ///
    /// * [`CutupError::InvalidConfig`] if `members` is empty, `lower` isn't finite and
    ///   nonnegative, or `upper` is NaN or below `lower`.
    /// * [`CutupError::UnknownAsset`] if a member isn't one of the assets.
    /// * [`CutupError::InfeasibleGroups`] naming the group if no portfolio summing to 1 within
    ///   the bounds meets it, or with no name if each group can be met but not all at once.
    pub fn group<A: Into<Asset>>(
        mut self,
        members: impl IntoIterator<Item = A>,
        lower: f64,
        upper: f64,
    ) -> Self {
        let members = members.into_iter().map(Into::into).collect();
        self.groups.push((members, lower, upper));
        self
    }

    /// Checks the bounds for the assets `labels`, and returns them with each upper bound
    /// lowered to what the other assets' lower bounds leave room for, `1 − Σⱼ≠ᵢ lⱼ`. That
    /// doesn't change which portfolios are feasible, and makes every upper bound finite.
    pub(crate) fn resolve(&self, labels: &[String]) -> Result<qp::Bounds, CutupError> {
        let n = labels.len();
        // Uniform bounds aren't any one asset's, so they're checked as settings.
        if let Limit::Uniform(l) = self.lower {
            if !(l.is_finite() && l >= 0.0) {
                return Err(CutupError::InvalidConfig {
                    field: "lower_bound",
                    value: l,
                });
            }
        }
        if let Limit::Uniform(u) = self.upper {
            let below = matches!(self.lower, Limit::Uniform(l) if u < l);
            if u.is_nan() || below {
                return Err(CutupError::InvalidConfig {
                    field: "upper_bound",
                    value: u,
                });
            }
        }
        let mut lower = self.lower.resolve("lower_bound", labels)?;
        let mut upper = self.upper.resolve("upper_bound", labels)?;
        for (asset, l, u) in &self.assets {
            let i = asset.resolve(labels)?;
            lower[i] = *l;
            upper[i] = *u;
        }
        asset::check_values("lower_bound", labels, lower.as_slice(), |l| {
            l.is_finite() && l >= 0.0
        })?;
        if let Some(i) = (0..n).find(|&i| upper[i].is_nan() || upper[i] < lower[i]) {
            return Err(asset::invalid_value("upper_bound", labels, i, upper[i]));
        }
        let (lower_sum, upper_sum) = (lower.sum(), upper.sum());
        // Within rounding of a budget of 1, as when ten caps of 0.1 sum to 0.9999999999999999.
        if lower_sum > 1.0 + 1e-12 || upper_sum < 1.0 - 1e-12 {
            return Err(CutupError::InfeasibleBounds {
                lower_sum,
                upper_sum,
            });
        }
        // Within rounding but short of the budget, no portfolio meets the bounds exactly, so
        // they're rescaled onto it, which moves each by at most 1e-12 of itself.
        let (lower, lower_sum) = if lower_sum > 1.0 {
            (lower / lower_sum, 1.0)
        } else {
            (lower, lower_sum)
        };
        let upper = if upper_sum < 1.0 {
            upper / upper_sum
        } else {
            upper
        };
        // Rounding can put that room a hair below lᵢ when the lower bounds sum to 1.
        let upper = DVector::from_fn(n, |i, _| {
            upper[i].min(1.0 - (lower_sum - lower[i])).max(lower[i])
        });
        let rows = self.rows(labels, &lower, &upper)?;
        Ok(qp::Bounds { lower, upper, rows })
    }

    /// The groups as the solver's rows, for the assets `labels` within resolved `lower` and
    /// `upper` bounds: one column of `a` per group, 1 for each member, with the group's limits.
    /// A limit the bounds and the budget already meet, at or beyond the least or the most its
    /// members can hold in a portfolio summing to 1, is dropped, made infinite. Otherwise the
    /// solver could hold the group's total at it while holding every member at its bound
    /// too, and that row would depend on the others, leaving no unique step. A limit just
    /// past what the members can hold, within rounding, is moved onto it.
    fn rows(
        &self,
        labels: &[String],
        lower: &DVector<f64>,
        upper: &DVector<f64>,
    ) -> Result<qp::Rows, CutupError> {
        let (n, r) = (labels.len(), self.groups.len());
        let mut a = DMatrix::zeros(n, r);
        let (mut row_lower, mut row_upper) = (DVector::zeros(r), DVector::zeros(r));
        // Every group is checked before any is found infeasible, so that a mistake in one
        // isn't hidden behind another that can't be met.
        for (j, (members, l, u)) in self.groups.iter().enumerate() {
            let (l, u) = (*l, *u);
            if members.is_empty() {
                return Err(CutupError::InvalidConfig {
                    field: "group",
                    value: 0.0,
                });
            }
            if !(l.is_finite() && l >= 0.0) {
                return Err(CutupError::InvalidConfig {
                    field: "group_lower_bound",
                    value: l,
                });
            }
            if u.is_nan() || u < l {
                return Err(CutupError::InvalidConfig {
                    field: "group_upper_bound",
                    value: u,
                });
            }
            for member in members {
                a[(member.resolve(labels)?, j)] = 1.0;
            }
        }
        for (j, &(_, l, u)) in self.groups.iter().enumerate() {
            let (least, most) = group_range(a.column(j).as_slice(), lower, upper);
            // Within rounding of what the members can hold, as for the bounds' sums.
            if l > most + 1e-12 || u < least - 1e-12 {
                let names: Vec<&str> = (0..n)
                    .filter(|&i| a[(i, j)] != 0.0)
                    .map(|i| labels[i].as_str())
                    .collect();
                return Err(CutupError::InfeasibleGroups {
                    group: Some(names.join(" + ")),
                });
            }
            row_lower[j] = if l <= least {
                f64::NEG_INFINITY
            } else {
                l.min(most)
            };
            row_upper[j] = if u >= most {
                f64::INFINITY
            } else {
                u.max(least)
            };
        }
        Ok(qp::Rows {
            a,
            lower: row_lower,
            upper: row_upper,
        })
    }
}

/// The least and the most total weight the assets with `member` 1 can hold in a portfolio
/// summing to 1 within resolved `lower` and `upper` bounds: their own bounds' sums, or what the
/// other assets' bounds leave of the budget, whichever is tighter.
fn group_range(member: &[f64], lower: &DVector<f64>, upper: &DVector<f64>) -> (f64, f64) {
    let sum = |bounds: &DVector<f64>, inside: bool| -> f64 {
        bounds
            .iter()
            .zip(member)
            .filter(|&(_, &m)| (m != 0.0) == inside)
            .map(|(b, _)| b)
            .sum()
    };
    (
        sum(lower, true).max(1.0 - sum(upper, false)),
        sum(upper, true).min(1.0 - sum(lower, false)),
    )
}

/// `bounds`, or [`CutupError::NotSupported`] if they have groups, which `with` doesn't take
/// yet.
pub(crate) fn without_groups(
    bounds: qp::Bounds,
    with: &'static str,
) -> Result<qp::Bounds, CutupError> {
    if bounds.rows.a.ncols() == 0 {
        Ok(bounds)
    } else {
        Err(CutupError::NotSupported {
            setting: "group",
            with,
        })
    }
}

/// A portfolio summing to 1 within resolved `bounds`, to start the solver from:
/// [`interior_start`] without groups. With them, a vertex from the simplex method's first
/// phase, which also finds whether the groups can all be met at once.
///
/// # Errors
///
/// [`CutupError::InfeasibleGroups`] with no name if they can't.
pub(crate) fn feasible_start(bounds: &qp::Bounds) -> Result<DVector<f64>, CutupError> {
    let rows = &bounds.rows;
    let (n, r) = rows.a.shape();
    if r == 0 {
        return Ok(interior_start(bounds));
    }
    let infeasible = CutupError::InfeasibleGroups { group: None };
    let program = Program::new(bounds, None);
    let w = match program.solve(&DVector::zeros(n + r))? {
        lp::Outcome::Optimal(solution) => solution.x.rows(0, n).into_owned(),
        // With no objective, nothing is unbounded.
        _ => return Err(infeasible),
    };
    // The simplex method accepts rows missed by up to 1e-9 of their terms, far more than the
    // rounding of a sum of weights, which the bounds and each group are held to. The solver
    // can't meet groups that only that looser tolerance can, so they can't all be met.
    if meets(bounds, &w, qp::sum_rounding(n, 1.0)) {
        Ok(w)
    } else {
        Err(infeasible)
    }
}

/// The linear program over the weights `w`, then each group's total `tⱼ`: `1ᵀw = 1`, then
/// `μᵀw = ρ` for a target `(μ, ρ)` if there's one, then `aⱼᵀw − tⱼ = 0` for each group, with the
/// weights within resolved bounds and the totals within the groups' limits.
struct Program {
    a: DMatrix<f64>,
    b: DVector<f64>,
    lower: DVector<f64>,
    upper: DVector<f64>,
}

impl Program {
    fn new(bounds: &qp::Bounds, target: Option<(&DVector<f64>, f64)>) -> Self {
        let rows = &bounds.rows;
        let (n, r) = rows.a.shape();
        let first = 1 + usize::from(target.is_some());
        let mut a = DMatrix::zeros(first + r, n + r);
        let mut b = DVector::zeros(first + r);
        a.row_mut(0).columns_mut(0, n).fill(1.0);
        b[0] = 1.0;
        if let Some((mu, rho)) = target {
            a.view_mut((1, 0), (1, n)).copy_from(&mu.transpose());
            b[1] = rho;
        }
        a.view_mut((first, 0), (r, n))
            .copy_from(&rows.a.transpose());
        a.view_mut((first, n), (r, r)).fill_diagonal(-1.0);
        let both = |weights: &DVector<f64>, groups: &DVector<f64>| {
            DVector::from_iterator(n + r, weights.iter().chain(groups.iter()).copied())
        };
        Program {
            a,
            b,
            lower: both(&bounds.lower, &rows.lower),
            upper: both(&bounds.upper, &rows.upper),
        }
    }

    fn solve(&self, cost: &DVector<f64>) -> Result<lp::Outcome, CutupError> {
        lp::solve(cost, &self.a, &self.b, &self.lower, &self.upper)
    }

    /// The lowest expected return `μᵀw` within resolved `bounds`, from this program without a
    /// target, or with `highest` the highest, and the bounds of every portfolio that reaches
    /// it. [`CutupError::InfeasibleGroups`] if the program has no solution, or only one that
    /// misses the bounds or a group by more than the rounding of a sum, as [`feasible_start`]
    /// finds.
    ///
    /// Those portfolios are the program's optimal face: at an optimum with multipliers `y`,
    /// each variable whose reduced cost `cⱼ − aⱼᵀy` isn't 0 is held where the optimum has it,
    /// at a bound, by every optimum. So the face holds those weights and groups' totals there,
    /// and keeps the others' bounds. A reduced cost counts as 0 within the simplex method's own
    /// rounding of it. A group's total is never held with all of its members: its row has a
    /// basic variable, a member, the total or the row's artificial variable, and either of the
    /// last two leaves the total a reduced cost of 0.
    fn end(
        &self,
        mu: &DVector<f64>,
        bounds: &qp::Bounds,
        highest: bool,
    ) -> Result<End, CutupError> {
        let (n, columns) = (mu.len(), self.a.ncols());
        let sign = if highest { -1.0 } else { 1.0 };
        let cost = DVector::from_fn(columns, |j, _| if j < n { sign * mu[j] } else { 0.0 });
        let lp::Outcome::Optimal(solution) = self.solve(&cost)? else {
            return Err(CutupError::InfeasibleGroups { group: None });
        };
        let x = &solution.x;
        let weights = x.rows(0, n).into_owned();
        // As for feasible_start: the simplex method accepts rows missed by up to 1e-9 of
        // their terms, so groups only that looser tolerance meets can't all be met.
        if !meets(bounds, &weights, qp::sum_rounding(n, 1.0)) {
            return Err(CutupError::InfeasibleGroups { group: None });
        }
        let (reduced, held) = reduced_costs(&cost, &self.a, &solution.y);
        let mut face = bounds.clone();
        for j in (0..n).filter(|&j| held[j]) {
            face.lower[j] = weights[j];
            face.upper[j] = weights[j];
        }
        for k in (0..bounds.rows.a.ncols()).filter(|&k| held[n + k]) {
            face.rows.lower[k] = x[n + k];
            face.rows.upper[k] = x[n + k];
        }
        // The asset a target just inside the end moves weight into: of those at their lower
        // bound with room above them, the one whose reduced cost is least, as the greedy
        // fill's next in order. Not only those held: at a degenerate end, one can sit in the
        // basis at its bound, with a reduced cost of 0.
        let next = (0..n)
            .filter(|&j| weights[j] == bounds.lower[j] && weights[j] < bounds.upper[j])
            .min_by(|&i, &j| reduced[i].abs().total_cmp(&reduced[j].abs()))
            .map_or(0.0, |j| mu[j].abs());
        Ok(End {
            value: mu.dot(&weights),
            rounding: (mu.component_mul(&weights).abs() * 1e-12).sum() + next * 1e-12,
            weights,
            face,
        })
    }
}

/// The reduced costs `cⱼ − aⱼᵀy` of a linear program `min cᵀx` subject to `Ax = b` at an
/// optimum with multipliers `y`, and which aren't 0. As the simplex method counts them, each is
/// a difference of terms up to `|cⱼ| + |aⱼ|ᵀ|y|`, so only beyond `1e-9` of that counts, but
/// never less than `1e-14` of the largest such sum.
fn reduced_costs(
    cost: &DVector<f64>,
    a: &DMatrix<f64>,
    y: &DVector<f64>,
) -> (DVector<f64>, Vec<bool>) {
    let reduced = cost - a.tr_mul(y);
    let magnitude = cost.abs() + a.abs().tr_mul(&y.abs());
    let floor = 1e-14 * magnitude.amax();
    let nonzero = (0..cost.len())
        .map(|j| reduced[j].abs() > (1e-9 * magnitude[j]).max(floor))
        .collect();
    (reduced, nonzero)
}

/// An end of the range of expected returns within the bounds, from [`Program::end`]: its
/// expected return, the rounding of that, a portfolio that reaches it, and the bounds of
/// every one that does.
struct End {
    value: f64,
    rounding: f64,
    weights: DVector<f64>,
    face: qp::Bounds,
}

/// Whether `w` sums to 1 and meets resolved `bounds`, groups included, each to within
/// `rounding`.
fn meets(bounds: &qp::Bounds, w: &DVector<f64>, rounding: f64) -> bool {
    let within = |x: &DVector<f64>, lower: &DVector<f64>, upper: &DVector<f64>| {
        (0..x.len()).all(|i| lower[i] - rounding <= x[i] && x[i] <= upper[i] + rounding)
    };
    let rows = &bounds.rows;
    let one = DVector::from_element(1, 1.0);
    within(&DVector::from_element(1, w.sum()), &one, &one)
        && within(w, &bounds.lower, &bounds.upper)
        && within(&rows.a.tr_mul(w), &rows.lower, &rows.upper)
}

/// A point strictly inside resolved `bounds` with weights summing to 1, where they allow one:
/// `l + t(u − l)` for the `t` that sums to 1. With the default bounds that's equal weights.
pub(crate) fn interior_start(bounds: &qp::Bounds) -> DVector<f64> {
    let room = &bounds.upper - &bounds.lower;
    let t = match room.sum() {
        total if total > 0.0 => ((1.0 - bounds.lower.sum()) / total).clamp(0.0, 1.0),
        _ => 0.0,
    };
    &bounds.lower + room * t
}

/// A target-return problem from [`ReturnRange::target_problem`].
pub(crate) struct TargetProblem {
    pub(crate) e: DMatrix<f64>,
    pub(crate) b: DVector<f64>,
    pub(crate) bounds: qp::Bounds,
    pub(crate) start: DVector<f64>,
}

/// The lowest and highest expected return of any portfolio within resolved bounds, and the
/// portfolios that reach them. Without groups: start every asset at its lower bound, then
/// raise assets to their upper bounds in order of expected return until the weights sum to 1.
/// With groups, which that can break, from the simplex method.
pub(crate) struct ReturnRange {
    lowest: f64,
    highest: f64,
    lowest_weights: DVector<f64>,
    highest_weights: DVector<f64>,
    /// How far from the lowest and the highest a target is within rounding of them.
    tolerances: (f64, f64),
    ends: Ends,
}

/// How a [`ReturnRange`] found its ends, and what it keeps of them.
enum Ends {
    /// Filled in order of expected return.
    Filled {
        /// The expected return of the last asset raised above its lower bound, for the lowest
        /// and the highest, or `None` if the lower bounds already sum to 1.
        margins: (Option<f64>, Option<f64>),
        /// The assets by expected return, highest first, ties in column order.
        descending: Vec<usize>,
    },
    /// From the simplex method: the bounds of the portfolios at the lowest, and at the highest.
    Faces(Box<(qp::Bounds, qp::Bounds)>),
}

impl ReturnRange {
    /// [`CutupError::InfeasibleGroups`] if no portfolio within the bounds meets every group.
    pub(crate) fn new(mu: &DVector<f64>, bounds: &qp::Bounds) -> Result<Self, CutupError> {
        if bounds.rows.a.ncols() == 0 {
            return Ok(ReturnRange::filled(mu, bounds));
        }
        let program = Program::new(bounds, None);
        let (low, high) = (
            program.end(mu, bounds, false)?,
            program.end(mu, bounds, true)?,
        );
        Ok(ReturnRange {
            lowest: low.value,
            highest: high.value,
            lowest_weights: low.weights,
            highest_weights: high.weights,
            tolerances: (low.rounding, high.rounding),
            ends: Ends::Faces(Box::new((low.face, high.face))),
        })
    }

    /// [`ReturnRange::new`] without groups.
    fn filled(mu: &DVector<f64>, bounds: &qp::Bounds) -> Self {
        // Stable sorts, so that ties are filled in column order, as argmin and argmax pick.
        let mut ascending: Vec<usize> = (0..mu.len()).collect();
        ascending.sort_by(|&i, &j| mu[i].total_cmp(&mu[j]));
        let mut descending: Vec<usize> = (0..mu.len()).collect();
        descending.sort_by(|&i, &j| mu[j].total_cmp(&mu[i]));
        let fill = |order: &[usize]| {
            let mut weights = bounds.lower.clone();
            let mut left = 1.0 - bounds.lower.sum();
            let mut margin = None;
            for &i in order {
                if left <= 0.0 {
                    break;
                }
                let raise = (bounds.upper[i] - bounds.lower[i]).min(left);
                if raise > 0.0 {
                    weights[i] += raise;
                    left -= raise;
                    margin = Some(mu[i]);
                }
            }
            (mu.dot(&weights), weights, margin)
        };
        let (lowest, lowest_weights, low_margin) = fill(&ascending);
        let (highest, highest_weights, high_margin) = fill(&descending);
        // The terms an end's expected return sums, and the next asset a target just inside it
        // moves weight into: the first in fill order with room left whose expected return isn't
        // tied with the margin's, since moving weight between those doesn't change the return.
        // Each term is scaled before summing, so that means near f64::MAX can't overflow.
        let rounding = |w: &DVector<f64>, order: &[usize], margin: Option<f64>| {
            let next = order
                .iter()
                .find(|&&i| w[i] < bounds.upper[i] && Some(mu[i]) != margin)
                .map_or(0.0, |&i| mu[i].abs());
            (mu.component_mul(w).abs() * 1e-12).sum() + next * 1e-12
        };
        ReturnRange {
            lowest,
            highest,
            tolerances: (
                rounding(&lowest_weights, &ascending, low_margin),
                rounding(&highest_weights, &descending, high_margin),
            ),
            lowest_weights,
            highest_weights,
            ends: Ends::Filled {
                margins: (low_margin, high_margin),
                descending,
            },
        }
    }

    /// The highest expected return of any portfolio within the bounds.
    pub(crate) fn highest(&self) -> f64 {
        self.highest
    }

    /// Returns `target` if some portfolio within the bounds reaches it: one within rounding
    /// of the lowest or highest is moved onto it.
    ///
    /// Rounding is measured at each end on its own, as `1e-12` of `Σ|μᵢwᵢ| + |μₖ|`: the terms
    /// that end's portfolio sums for its expected return, and the expected return of `k`, the
    /// next asset in fill order with room left and not tied with the margin, which a target
    /// just inside the end moves weight into. A target closer than that needs a change in
    /// `k`'s weight below the rounding of the others'. A mean far from the rest widens only
    /// the tolerance of an end that holds its asset or would move into it next.
    pub(crate) fn reachable(&self, target: f64) -> Result<f64, CutupError> {
        let (lowest, highest) = (self.lowest, self.highest);
        let (low, high) = self.tolerances;
        if target < lowest - low || target > highest + high {
            return Err(CutupError::TargetReturnOutOfRange {
                target_return: target,
                lowest,
                highest,
            });
        }
        // From either side: just inside an end, only a sliver of portfolios next to its face
        // reach the target, and the solvers resolve that to rounding no better than the face.
        // In a range narrower than that, to the nearer end.
        let (below, above) = (target - lowest, highest - target);
        Ok(match (below <= low, above <= high) {
            (false, false) => target,
            (true, false) => lowest,
            (false, true) => highest,
            (true, true) if below < above => lowest,
            (true, true) => highest,
        })
    }

    /// The constraints `eᵀw = b` and bounds of a long-only problem at expected return
    /// `target`, and a start that meets them. At an end of the range, the bounds are those of
    /// the portfolios that reach it, its face, and the target needs no constraint of its own.
    /// Without groups, those fill the bounds in order of expected return, so every asset but
    /// those tied at the margin is fixed, and a start inside the range holds only assets next
    /// to each other in expected return. With groups, the face and the start come from the
    /// simplex method.
    pub(crate) fn target_problem(
        mu: &DVector<f64>,
        bounds: qp::Bounds,
        target: f64,
    ) -> Result<TargetProblem, CutupError> {
        let n = mu.len();
        let range = ReturnRange::new(mu, &bounds)?;
        let target = range.reachable(target)?;
        if target == range.lowest || target == range.highest {
            let bounds = range.face(mu, &bounds, target == range.highest);
            return Ok(TargetProblem {
                e: DMatrix::from_element(n, 1, 1.0),
                b: DVector::from_element(1, 1.0),
                start: feasible_start(&bounds)?,
                bounds,
            });
        }
        Ok(TargetProblem {
            e: DMatrix::from_columns(&[DVector::from_element(n, 1.0), mu.clone()]),
            b: DVector::from_vec(vec![1.0, target]),
            start: range.start(mu, &bounds, target)?,
            bounds,
        })
    }

    /// A portfolio within the bounds reaching `target`. With every asset at its lower bound,
    /// the budget left over is a window laid along the assets sorted by expected return, each
    /// taking up to its room above its lower bound. At one end the window holds the highest
    /// return portfolio and at the other the lowest, and between them its return falls
    /// linearly from one breakpoint, where it meets an asset's edge, to the next. So the target
    /// is reached exactly by holding only assets next to each other in expected return. An
    /// asset whose expected return is far from the rest is held only if the target needs it:
    /// a start holding a little of every asset would leave it free in the solver's first step,
    /// where its expected return swamps the target's constraint.
    ///
    /// With groups, a vertex from the simplex method of the portfolios that reach the target.
    fn start(
        &self,
        mu: &DVector<f64>,
        bounds: &qp::Bounds,
        target: f64,
    ) -> Result<DVector<f64>, CutupError> {
        let Ends::Filled { descending, .. } = &self.ends else {
            let program = Program::new(bounds, Some((mu, target)));
            let columns = program.a.ncols();
            return match program.solve(&DVector::zeros(columns))? {
                lp::Outcome::Optimal(solution) => Ok(solution.x.rows(0, mu.len()).into_owned()),
                // The target is inside the range, so some portfolio reaches it.
                _ => Err(CutupError::NotConverged),
            };
        };
        // Divided by the largest in magnitude, so that no sum below can overflow. The target
        // is strictly inside the range, so some expected return isn't 0.
        let scale = mu.amax();
        let (mu, target) = (mu / scale, target / scale);
        let budget = 1.0 - bounds.lower.sum();
        // Highest first, ties in column order, as the highest return portfolio fills them.
        let order = descending;
        // Each asset's stretch of the line, no longer than the budget.
        let mut edges = vec![0.0];
        for &i in order {
            let room = (bounds.upper[i] - bounds.lower[i]).min(budget);
            edges.push(edges[edges.len() - 1] + room);
        }
        // The window starting `s` along the line. l + (u − l) can round past u.
        let at = |s: f64| {
            let mut w = bounds.lower.clone();
            for (k, &i) in order.iter().enumerate() {
                let held = edges[k + 1].min(s + budget) - edges[k].max(s);
                w[i] = (w[i] + held.max(0.0)).min(bounds.upper[i]);
            }
            w
        };
        // Where either end of the window meets an edge, from the start of the line. The window
        // reaches the lowest return portfolio, at or below the target, at the last edge less
        // the budget, so the search stops by then.
        let mut breaks: Vec<f64> = edges
            .iter()
            .flat_map(|&edge| [edge, edge - budget])
            .filter(|&s| s >= 0.0)
            .collect();
        breaks.sort_by(f64::total_cmp);
        // Each break's return is the next pair's high, so it's computed once.
        let mut high = mu.dot(&at(breaks[0]));
        for pair in breaks.windows(2) {
            let low = mu.dot(&at(pair[1]));
            if low <= target {
                #[allow(
                    clippy::manual_clamp,
                    reason = "f64::max drops the NaN of a segment that doesn't move the return, \
                              which clamp keeps"
                )]
                let t = ((high - target) / (high - low)).max(0.0).min(1.0);
                return Ok(at(pair[0] + t * (pair[1] - pair[0])));
            }
            high = low;
        }
        Ok(self.lowest_weights.clone())
    }

    /// The bounds of the portfolios with the highest expected return, or the lowest. Without
    /// groups, assets whose expected return is above the margin's (below it, for the lowest)
    /// are held at their upper bound, those on the other side at their lower bound, and only
    /// assets tied with the margin keep their bounds. With groups, the face the simplex method
    /// found.
    fn face(&self, mu: &DVector<f64>, bounds: &qp::Bounds, highest: bool) -> qp::Bounds {
        let margins = match &self.ends {
            Ends::Filled { margins, .. } => margins,
            Ends::Faces(faces) => {
                return if highest {
                    faces.1.clone()
                } else {
                    faces.0.clone()
                };
            }
        };
        let (weights, margin) = if highest {
            (&self.highest_weights, margins.1)
        } else {
            (&self.lowest_weights, margins.0)
        };
        let mut face = bounds.clone();
        for i in (0..mu.len()).filter(|&i| margin != Some(mu[i])) {
            face.lower[i] = weights[i];
            face.upper[i] = weights[i];
        }
        face
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_weights_close, full_rank_prices, uncorrelated_returns};
    use crate::{CovarianceConfig, MeanVariance, MinVariance, Returns, WeightBounds, Weights};
    use nalgebra::{DMatrix, dvector};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_a_range_narrower_than_rounding_snaps_to_the_nearer_end() {
        // Returns of 100, 100 + 5e-11 and 100: the whole range is within rounding, and a
        // target at its top is only reached by the second asset.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let mean_variance = MeanVariance::target_return(100.0 + 5e-11)
            .expected_returns(dvector![100.0, 100.0 + 5e-11, 100.0].as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-15,
        );
    }

    #[test]
    fn test_bounded_min_variance_water_fills() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let config = CovarianceConfig::default();
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().covariance(config.clone()))
                .unwrap(),
            &Weights::from(vec![4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]),
            1e-12,
        );

        // A cap of 50% on the first asset passes the rest to the others in proportion to
        // inverse variance, 1/3 and 1/6.
        let capped = WeightBounds::per_asset(vec![0.0; 3], vec![0.5, 1.0, 1.0]);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(capped),
            )
            .unwrap();
        assert_weights_close(
            &weights,
            &Weights::from(vec![0.5, 1.0 / 3.0, 1.0 / 6.0]),
            1e-12,
        );

        // A floor of 25% on the last asset leaves 75% to split 2 : 1.
        let floored = WeightBounds::per_asset(vec![0.0, 0.0, 0.25], vec![0.5, 1.0, 1.0]);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(floored),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.25, 0.25]), 1e-12);

        // Floors and caps of 1/3 leave equal weights as the only portfolio.
        let pinned = WeightBounds::uniform(1.0 / 3.0, 1.0 / 3.0);
        let weights = allocator
            .allocate(&MinVariance::default().covariance(config).bounds(pinned))
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![1.0 / 3.0; 3]), 1e-15);
    }

    #[test]
    fn test_bounds_that_dont_bind_change_nothing() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().shrinkage(0.2);
        let loose = WeightBounds::uniform(0.0, 1.0);
        assert_eq!(
            allocator
                .allocate(
                    &MinVariance::default()
                        .covariance(config.clone())
                        .bounds(loose.clone())
                )
                .unwrap(),
            allocator
                .allocate(&MinVariance::default().covariance(config))
                .unwrap()
        );
        for mean_variance in [
            MeanVariance::risk_aversion(5.0),
            MeanVariance::target_return(0.004),
        ] {
            let mean_variance =
                mean_variance.expected_returns(dvector![0.002, 0.006, 0.003].as_slice());
            assert_eq!(
                allocator
                    .allocate(&mean_variance.clone().bounds(loose.clone()))
                    .unwrap(),
                allocator.allocate(&mean_variance).unwrap()
            );
        }
    }

    #[test]
    fn test_bounded_mean_variance_targets_the_reachable_range() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let mu = dvector![0.003, 0.003, 0.001];
        let bounds = WeightBounds::uniform(0.0, 0.6);

        // The lowest return fills the last asset to 60% first, then the first to 40%.
        let mean_variance = MeanVariance::target_return(0.004).expected_returns(mu.as_slice());
        match allocator.allocate(&mean_variance.bounds(bounds.clone())) {
            Err(CutupError::TargetReturnOutOfRange {
                target_return,
                lowest,
                highest,
            }) => {
                assert_eq!(target_return, 0.004);
                assert!((lowest - 0.0018).abs() < 1e-15, "{lowest}");
                assert_eq!(highest, 0.003);
            }
            other => panic!("{other:?}"),
        }

        // At the highest return the last asset holds nothing, and the two tied assets split
        // the weight by minimum variance, 2/3 and 1/3, until the 60% cap binds.
        let mean_variance = MeanVariance::target_return(0.003).expected_returns(mu.as_slice());
        let weights = allocator
            .allocate(&mean_variance.bounds(bounds.clone()))
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.6, 0.4, 0.0]), 1e-12);

        // In between, the target holds and the weights stay within the bounds.
        let mean_variance = MeanVariance::target_return(0.0025).expected_returns(mu.as_slice());
        let weights = allocator.allocate(&mean_variance.bounds(bounds)).unwrap();
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - 0.0025).abs() < 1e-15, "{achieved}");
        assert!(
            weights.iter().all(|&w| (0.0..=0.6 + 1e-15).contains(&w)),
            "{weights:?}"
        );
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    fn three() -> Vec<String> {
        ["A", "B", "C"].map(String::from).to_vec()
    }

    #[test]
    fn test_resolved_caps_leave_room_for_the_other_floors() {
        let resolved = WeightBounds::default().resolve(&three()).unwrap();
        assert_eq!(resolved.upper, dvector![1.0, 1.0, 1.0]);
        let resolved = WeightBounds::per_asset(vec![0.2, 0.1, 0.0], vec![f64::INFINITY; 3])
            .resolve(&three())
            .unwrap();
        assert_weights_close(
            &Weights::from(resolved.upper.as_slice().to_vec()),
            &Weights::from(vec![0.9, 0.8, 0.7]),
            1e-15,
        );
    }

    #[test]
    fn test_bounds_exactly_at_the_rounding_allowance_are_feasible() {
        // Floors summing to exactly 1 + 1e-12 are rounding, and are rescaled to sum to 1.
        let resolved = WeightBounds::per_asset(vec![1.0 + 1e-12, 0.0, 0.0], vec![2.0; 3])
            .resolve(&three())
            .unwrap();
        assert!((resolved.lower.sum() - 1.0).abs() <= f64::EPSILON);
        // Caps summing to exactly 1 − 1e-12 are rounding too.
        assert!(
            WeightBounds::per_asset(vec![0.0; 3], vec![1.0 - 1e-12, 0.0, 0.0])
                .resolve(&three())
                .is_ok()
        );
    }

    #[test]
    fn test_invalid_per_asset_bounds_are_rejected() {
        let invalid = |field, column, value| CutupError::InvalidAssetValue {
            field,
            column,
            asset: three()[column].clone(),
            value,
        };
        let resolve = |lower: Vec<f64>, upper: Vec<f64>| {
            WeightBounds::per_asset(lower, upper).resolve(&three())
        };
        assert_eq!(
            resolve(vec![0.0, -0.1, 0.0], vec![1.0; 3]).unwrap_err(),
            invalid("lower_bound", 1, -0.1)
        );
        assert_eq!(
            resolve(vec![0.0, 0.0, f64::INFINITY], vec![f64::INFINITY; 3]).unwrap_err(),
            invalid("lower_bound", 2, f64::INFINITY)
        );
        assert_eq!(
            resolve(vec![0.0, 0.3, 0.0], vec![1.0, 0.2, 1.0]).unwrap_err(),
            invalid("upper_bound", 1, 0.2)
        );
    }

    #[test]
    fn test_interior_start_spreads_the_room_above_the_floors() {
        let bounds = qp::Bounds::new(dvector![0.1, 0.0, 0.0], dvector![0.5, 1.0, 1.0]);
        // Room 2.4 for the 0.9 left above the floors: 0.375 of each asset's room.
        assert_weights_close(
            &Weights::from(interior_start(&bounds).as_slice().to_vec()),
            &Weights::from(vec![0.25, 0.375, 0.375]),
            1e-15,
        );
        // With no room, the floors.
        let fixed = qp::Bounds::new(dvector![0.5, 0.5], dvector![0.5, 0.5]);
        assert_eq!(interior_start(&fixed), dvector![0.5, 0.5]);
    }

    #[test]
    fn test_reachable_snaps_targets_within_rounding_of_an_end() {
        let unit = qp::Bounds::new(dvector![0.0, 0.0], dvector![1.0, 1.0]);
        // From 0 to 1. The top's portfolio earns 1, and the next asset 0, so its rounding is
        // 1e-12: a target exactly that far inside snaps to it. The bottom's earns exactly 0,
        // but a target just above it moves weight into the asset earning 1, so it's 1e-12 too.
        let range = ReturnRange::new(&dvector![0.0, 1.0], &unit).unwrap();
        assert_eq!(range.reachable(1.0 - 1e-12), Ok(1.0));
        assert_eq!(range.reachable(1.0 - 2e-12), Ok(1.0 - 2e-12));
        assert_eq!(range.reachable(1.0 + 1e-12), Ok(1.0));
        assert!(range.reachable(1.0 + 2e-12).is_err());
        assert_eq!(range.reachable(1e-12), Ok(0.0));
        assert_eq!(range.reachable(2e-12), Ok(2e-12));
        assert_eq!(range.reachable(-1e-12), Ok(0.0));
        assert!(range.reachable(-2e-12).is_err());
        // With no asset left to move into, as when the bounds hold every asset at its cap,
        // only the end's own terms count.
        let full = qp::Bounds::new(dvector![0.0, 0.0], dvector![0.5, 0.5]);
        let range = ReturnRange::new(&dvector![1.0, 3.0], &full).unwrap();
        assert_eq!(range.tolerances, (2e-12, 2e-12));
        // In a range within rounding, a target goes to the nearer end, and halfway to the top.
        let narrow = ReturnRange::new(&dvector![1.0, 1.0 + 4.0 * f64::EPSILON], &unit).unwrap();
        assert_eq!(narrow.reachable(1.0 + f64::EPSILON), Ok(1.0));
        assert_eq!(
            narrow.reachable(1.0 + 3.0 * f64::EPSILON),
            Ok(1.0 + 4.0 * f64::EPSILON)
        );
        assert_eq!(
            narrow.reachable(1.0 + 2.0 * f64::EPSILON),
            Ok(1.0 + 4.0 * f64::EPSILON)
        );
    }

    #[test]
    fn test_each_end_rounds_by_its_own_portfolio() {
        let unit = qp::Bounds::new(dvector![0.0, 0.0, 0.0], dvector![1.0, 1.0, 1.0]);
        // The top holds only the second asset, so the first's −1e6 doesn't widen its rounding,
        // 2e-14. The bottom holds the first, so its rounding is 1e-6.
        let mu = dvector![-1e6, 0.02, 0.02 - 5e-7];
        let range = ReturnRange::new(&mu, &unit).unwrap();
        assert_eq!(range.reachable(0.02 - 1e-7), Ok(0.02 - 1e-7));
        assert_eq!(range.reachable(0.02 - 1e-14), Ok(0.02));
        assert_eq!(range.reachable(-1e6 + 5e-7), Ok(-1e6));
        assert_eq!(range.reachable(-1e6 + 2e-6), Ok(-1e6 + 2e-6));
        assert!(range.reachable(0.02 + 1e-13).is_err());
        assert!(range.reachable(-1e6 - 2e-6).is_err());
    }

    /// The start [`ReturnRange::target_problem`] gives for `target`, checked to be within the
    /// bounds, sum to 1 and reach the target.
    fn start(mu: &DVector<f64>, bounds: &qp::Bounds, target: f64) -> DVector<f64> {
        let problem = ReturnRange::target_problem(mu, bounds.clone(), target).unwrap();
        let w = problem.start;
        assert!((w.sum() - 1.0).abs() <= 1e-15, "{w:?}");
        assert!((0..w.len()).all(|i| bounds.lower[i] <= w[i] && w[i] <= bounds.upper[i]));
        let scale = mu.amax();
        assert!(
            (mu.dot(&w) - target).abs() <= 1e-15 * scale,
            "{} vs {target}",
            mu.dot(&w)
        );
        w
    }

    #[test]
    fn test_the_start_holds_only_assets_near_the_target() {
        let unit = qp::Bounds::new(dvector![0.0, 0.0, 0.0, 0.0], dvector![1.0, 1.0, 1.0, 1.0]);
        let mu = dvector![-1e6, 0.02, 0.0199995, 0.01];
        // Near the top, only the second and third assets; lower, the third and fourth; far
        // lower, the fourth and first. The first is held only where the target needs it.
        let w = start(&mu, &unit, 0.02 - 1e-7);
        assert_eq!((w[0], w[3]), (0.0, 0.0));
        let w = start(&mu, &unit, 0.0199995 - 1e-4);
        assert_eq!((w[0], w[1]), (0.0, 0.0));
        let w = start(&mu, &unit, -5e5);
        assert_eq!((w[1], w[2]), (0.0, 0.0));
        assert!((w[0] - 0.5).abs() < 1e-6);
        // With floors and caps, and still reaching the target.
        let bounds = qp::Bounds::new(dvector![0.0, 0.1, 0.0, 0.2], dvector![0.5, 0.6, 0.15, 1.0]);
        assert_eq!(start(&mu, &bounds, 0.016)[0], 0.0);
        start(&mu, &bounds, -1e5);
        // Means near f64::MAX don't overflow it.
        let huge = dvector![-f64::MAX, f64::MAX / 2.0, f64::MAX];
        let w = start(
            &huge,
            &qp::Bounds::new(dvector![0.0, 0.0, 0.0], dvector![1.0, 1.0, 1.0]),
            f64::MAX * 0.75,
        );
        assert_eq!(w[0], 0.0);
    }

    #[test]
    fn test_the_start_slides_the_budget_along_the_returns() {
        let cap = qp::Bounds::new(DVector::zeros(4), DVector::from_element(4, 0.5));
        let start = |mu: &DVector<f64>, bounds: &qp::Bounds, target: f64| {
            start(mu, bounds, target)
                .iter()
                .map(|w| (w * 1e12).round() / 1e12)
                .collect::<Vec<_>>()
        };
        // From 1500 to 3500: the window of 1 slides from [0, 0, 0.5, 0.5] to [0.5, 0.5, 0, 0],
        // through [0, 0.5, 0.5, 0] at 2500.
        let mu = dvector![1000.0, 2000.0, 3000.0, 4000.0];
        assert_eq!(start(&mu, &cap, 3400.0), [0.0, 0.05, 0.5, 0.45]);
        assert_eq!(start(&mu, &cap, 2500.0), [0.0, 0.5, 0.5, 0.0]);
        assert_eq!(start(&mu, &cap, 1600.0), [0.45, 0.5, 0.05, 0.0]);
        // Tied expected returns are laid in column order.
        let tied = dvector![0.0, 1.0, 1.0, 2.0];
        let unit = qp::Bounds::new(DVector::zeros(4), DVector::from_element(4, 1.0));
        assert_eq!(start(&tied, &unit, 1.2), [0.0, 0.8, 0.0, 0.2]);
        assert_eq!(start(&tied, &unit, 0.5), [0.5, 0.0, 0.5, 0.0]);
        assert_eq!(start(&tied, &cap, 1.2), [0.0, 0.5, 0.3, 0.2]);
        // Uncapped, each asset's stretch is the budget left after the floors, here 0.6.
        let floors = qp::Bounds::new(
            dvector![0.1, 0.1, 0.1, 0.1],
            DVector::from_element(4, f64::INFINITY),
        );
        assert_eq!(start(&mu, &floors, 2500.0), [0.1, 0.4, 0.4, 0.1]);
    }

    #[test]
    fn test_a_target_across_several_bounded_assets_is_reached() {
        // Close to a case from the property tests, where a start that moved weight between the
        // closest pair one bound at a time fell short of the target, and the solve failed.
        let returns = DMatrix::from_vec(
            14,
            4,
            vec![
                0.0,
                0.043021496,
                0.0,
                0.039817187,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.031959524,
                0.0,
                -0.003818411,
                -0.020443345,
                -0.017913865,
                -0.034875112,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.037736578,
                0.043941819,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.021989019,
                -0.028166691,
                0.0,
                -0.030074956,
                -0.005915370,
                -0.008912545,
                -0.014654237,
                0.0,
                0.0,
                0.041957695,
                0.047556898,
                -0.018680320,
                0.0,
                0.025386184,
                0.020801107,
                -0.038463244,
                -0.049684629,
                0.006717366,
                0.036041461,
                -0.033560917,
                -0.017856673,
                -0.048719500,
                0.043808973,
                0.019405156,
                0.045918587,
                -0.045944856,
                0.027652960,
                0.009290532,
            ],
        );
        let mu = returns.row_mean().transpose();
        let data = Returns::returns_matrix(returns).unwrap();
        let bounds = WeightBounds::per_asset(
            vec![0.0, 0.068065541, 0.0, 0.185450640],
            vec![0.688408852, f64::INFINITY, 0.730141329, 0.276657586],
        );
        let target = 0.0011;
        let weights = data
            .allocate(&MeanVariance::target_return(target).bounds(bounds))
            .unwrap();
        let earned = mu.dot(&DVector::from_column_slice(weights.as_slice()));
        assert!((earned - target).abs() <= 1e-12 * mu.amax(), "{earned}");
    }

    #[test]
    fn test_a_target_just_off_an_end_earning_nothing_snaps_to_it() {
        // The lowest return portfolio earns exactly 0, but a target 1.79e-13 above it needs
        // about 3e-16 of the asset earning 616.5, below the rounding of the other weights.
        let bounds = WeightBounds::per_asset(vec![0.0, 0.2, 0.2], vec![1.0, 1.0, 1.0]);
        let mu = [616.5, 0.0, 0.0];
        for seed in [3, 5, 7] {
            let data =
                Returns::returns_matrix(crate::test_support::factor_returns(3, 30, seed)).unwrap();
            let method = MeanVariance::target_return(1.79e-13)
                .expected_returns(mu)
                .bounds(bounds.clone());
            let weights = data.allocate(&method).unwrap();
            assert_eq!(weights[0], 0.0);
        }
    }

    #[test]
    fn test_targets_near_the_top_beside_a_far_mean_are_reached() {
        // The first asset is a hundred times as volatile, so the minimum variance portfolio
        // holds none of it either.
        let mut returns = crate::test_support::factor_returns(3, 40, 7);
        returns.column_mut(0).scale_mut(100.0);
        let data = Returns::returns_matrix(returns).unwrap();
        let mu = [-1e6, 0.02, 0.02 - 5e-7];
        // Between the two highest means, which the first asset isn't needed to reach.
        for target in [0.02 - 4e-7, 0.02 - 1e-7, 0.02 - 1e-9, 0.02 - 1e-11] {
            let method = MeanVariance::target_return(target).expected_returns(mu);
            let weights = data.allocate(&method).unwrap();
            let earned: f64 = weights.iter().zip(mu).map(|(w, m)| w * m).sum();
            assert!(
                (earned - target).abs() <= 1e-12 * target,
                "{earned} vs {target}"
            );
        }
        // So the frontier stays evenly spaced on them.
        let frontier = data
            .frontier(&MeanVariance::risk_aversion(1.0).expected_returns(mu), 5)
            .unwrap();
        assert_eq!(frontier.len(), 5);
        let (lowest, highest) = (frontier[0].expected_return(), frontier[4].expected_return());
        for (k, point) in frontier.iter().enumerate() {
            let want = lowest + (highest - lowest) * k as f64 / 4.0;
            assert!(
                (point.expected_return() - want).abs() <= 1e-12 * want.abs(),
                "{k}: {}",
                point.expected_return()
            );
        }
    }

    #[test]
    fn test_a_target_at_an_end_needs_only_the_budget() {
        let unit = qp::Bounds::new(dvector![0.0, 0.0, 0.0], dvector![1.0, 1.0, 1.0]);
        let mu = dvector![0.1, 0.2, 0.3];
        for (target, columns) in [(0.1, 1), (0.3, 1), (0.2, 2)] {
            let problem = ReturnRange::target_problem(&mu, unit.clone(), target).unwrap();
            assert_eq!(problem.e.ncols(), columns, "{target}");
        }
    }

    #[test]
    fn test_groups_resolve_to_rows_narrowed_to_what_their_members_can_hold() {
        // Caps of 0.6 leave A and C at least 0.4 together, and B and A too, so floors up to
        // that are met already and dropped, as is a cap at or above the 1 they can hold. An
        // asset listed twice counts once.
        let resolved = WeightBounds::uniform(0.0, 0.6)
            .group(["A", "C"], 0.4, f64::INFINITY)
            .group(["B", "A", "B"], 0.1, 0.5)
            .group(["A", "C"], 0.4 + 1e-9, 1.0 - 1e-9)
            .resolve(&three())
            .unwrap();
        let rows = &resolved.rows;
        assert_eq!(
            rows.a,
            DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        );
        assert_eq!(
            rows.lower,
            dvector![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.4 + 1e-9]
        );
        assert_eq!(rows.upper, dvector![f64::INFINITY, 0.5, 1.0 - 1e-9]);
        // By column, with a floor on the others that leaves at most 0.7, and equal limits.
        let resolved = WeightBounds::per_asset(vec![0.0, 0.0, 0.3], vec![1.0; 3])
            .group([0, 1], 0.2, 0.7)
            .group([1], 0.3, 0.3)
            .resolve(&three())
            .unwrap();
        assert_eq!(resolved.rows.lower, dvector![0.2, 0.3]);
        assert_eq!(resolved.rows.upper, dvector![f64::INFINITY, 0.3]);
        // Without groups, none.
        assert_eq!(
            WeightBounds::default()
                .resolve(&three())
                .unwrap()
                .rows
                .a
                .ncols(),
            0
        );
    }

    #[test]
    fn test_the_range_of_a_group_is_its_members_or_what_the_others_leave() {
        let (lower, upper) = (dvector![0.1, 0.2, 0.0], dvector![0.3, 0.9, 0.5]);
        // A and C: 0.1 to 0.8 alone, but B takes 0.2 to 0.9 of the budget.
        assert_eq!(group_range(&[1.0, 0.0, 1.0], &lower, &upper), (0.1, 0.8));
        assert_eq!(group_range(&[0.0, 1.0, 0.0], &lower, &upper), (0.2, 0.9));
        let (lower, upper) = (dvector![0.0, 0.0, 0.0], dvector![0.3, 0.3, 0.3]);
        assert_eq!(group_range(&[1.0, 1.0, 0.0], &lower, &upper), (0.7, 0.6));
    }

    #[test]
    fn test_invalid_groups_are_rejected() {
        let resolve = |bounds: WeightBounds| bounds.resolve(&three());
        let invalid = |field, value| Err(CutupError::InvalidConfig { field, value });
        let bounds = WeightBounds::default;
        assert_eq!(
            resolve(bounds().group(Vec::<usize>::new(), 0.0, 1.0)),
            invalid("group", 0.0)
        );
        assert_eq!(
            resolve(bounds().group(["A"], -0.1, 1.0)),
            invalid("group_lower_bound", -0.1)
        );
        assert_eq!(
            resolve(bounds().group(["A"], f64::INFINITY, f64::INFINITY)),
            invalid("group_lower_bound", f64::INFINITY)
        );
        assert!(matches!(
            resolve(bounds().group(["A"], f64::NAN, 1.0)),
            Err(CutupError::InvalidConfig {
                field: "group_lower_bound",
                ..
            })
        ));
        assert_eq!(
            resolve(bounds().group(["A"], 0.5, 0.4)),
            invalid("group_upper_bound", 0.4)
        );
        assert!(matches!(
            resolve(bounds().group(["A"], 0.0, f64::NAN)),
            Err(CutupError::InvalidConfig {
                field: "group_upper_bound",
                ..
            })
        ));
        assert!(matches!(
            resolve(bounds().group(["A", "D"], 0.0, 1.0)),
            Err(CutupError::UnknownAsset { .. })
        ));
    }

    #[test]
    fn test_a_group_its_members_cant_meet_is_named() {
        let named = |group: &str| {
            Err(CutupError::InfeasibleGroups {
                group: Some(group.into()),
            })
        };
        let capped = || WeightBounds::uniform(0.0, 0.4);
        // A and B hold at most 0.8 under caps of 0.4, and at least 0.6 with C's cap.
        assert_eq!(
            capped().group(["B", "A"], 0.9, 1.0).resolve(&three()),
            named("A + B")
        );
        assert_eq!(
            capped().group([0, 1], 0.0, 0.1).resolve(&three()),
            named("A + B")
        );
        // Within 1e-12 is rounding, and the limit is moved onto what the members can hold.
        let resolved = capped()
            .group(["A", "B"], 0.8 + 1e-12, 1.0)
            .resolve(&three())
            .unwrap();
        assert_eq!(resolved.rows.lower, dvector![0.8]);
        assert!(
            capped()
                .group(["A", "B"], 0.8 + 2e-12, 1.0)
                .resolve(&three())
                .is_err()
        );
        let resolved = capped()
            .group(["A", "B"], 0.0, 0.6 - 1e-12)
            .resolve(&three())
            .unwrap();
        assert_eq!(resolved.rows.upper, dvector![0.6]);
        assert!(
            capped()
                .group(["A", "B"], 0.0, 0.6 - 2e-12)
                .resolve(&three())
                .is_err()
        );
    }

    /// Checks `w` sums to 1 and meets resolved `bounds`, groups included, to `tolerance`.
    fn assert_feasible(w: &DVector<f64>, bounds: &qp::Bounds, tolerance: f64) {
        assert!((w.sum() - 1.0).abs() <= tolerance, "{w:?}");
        for i in 0..w.len() {
            assert!(bounds.lower[i] - tolerance <= w[i] && w[i] <= bounds.upper[i] + tolerance);
        }
        let totals = bounds.rows.a.tr_mul(w);
        for j in 0..totals.len() {
            assert!(
                bounds.rows.lower[j] - tolerance <= totals[j]
                    && totals[j] <= bounds.rows.upper[j] + tolerance,
                "group {j}: {}",
                totals[j]
            );
        }
    }

    #[test]
    fn test_the_start_meets_the_groups_or_finds_they_cant_all_be_met() {
        // Equal weights, the start without groups, break both groups.
        let bounds = WeightBounds::uniform(0.0, 0.6)
            .group(["A", "B"], 0.0, 0.5)
            .group(["B", "C"], 0.0, 0.5)
            .resolve(&three())
            .unwrap();
        assert_feasible(&feasible_start(&bounds).unwrap(), &bounds, 1e-15);
        // Without groups, the start strictly inside the bounds.
        let plain = WeightBounds::uniform(0.0, 0.6).resolve(&three()).unwrap();
        assert_eq!(feasible_start(&plain).unwrap(), interior_start(&plain));
        // Each pair at least 0.7 needs 1.05 in all.
        let pairs = WeightBounds::default()
            .group(["A", "B"], 0.7, 1.0)
            .group(["B", "C"], 0.7, 1.0)
            .group(["A", "C"], 0.7, 1.0)
            .resolve(&three())
            .unwrap();
        assert_eq!(
            feasible_start(&pairs),
            Err(CutupError::InfeasibleGroups { group: None })
        );
        // Caps that leave the budget 1e-10 short, which the simplex method's tolerance takes
        // as met, can't all be met.
        let short = WeightBounds::default()
            .group(["A"], 0.0, 0.3)
            .group(["B"], 0.0, 0.3)
            .group(["C"], 0.0, 0.4 - 1e-10)
            .resolve(&three())
            .unwrap();
        assert_eq!(
            feasible_start(&short),
            Err(CutupError::InfeasibleGroups { group: None })
        );
    }

    #[test]
    fn test_a_start_meets_the_bounds_and_groups_to_within_rounding() {
        // Rounding of 0.25, so every value is exact: bounds of 0 to 0.5, and the first two
        // from 0.25 to 0.75 together.
        let bounds = qp::Bounds {
            lower: DVector::zeros(3),
            upper: DVector::from_element(3, 0.5),
            rows: qp::Rows {
                a: DMatrix::from_column_slice(3, 1, &[1.0, 1.0, 0.0]),
                lower: dvector![0.25],
                upper: dvector![0.75],
            },
        };
        let meets = |w: [f64; 3]| meets(&bounds, &DVector::from_row_slice(&w), 0.25);
        assert!(meets([0.25, 0.25, 0.5]));
        // At each limit's rounding, and past it.
        assert!(meets([-0.25, 0.75, 0.5]));
        assert!(!meets([-0.375, 0.875, 0.5]));
        assert!(meets([0.5, 0.5, 0.25]));
        assert!(!meets([0.625, 0.5, 0.25]));
        // Only the group missed.
        assert!(!meets([0.75, 0.375, -0.125]));
        assert!(meets([0.0, 0.0, 0.75]));
        assert!(!meets([0.0, -0.125, 0.75]));
        assert!(meets([0.25, 0.25, 0.75]));
        assert!(!meets([0.25, 0.25, 0.875]));
        assert!(meets([0.25, 0.25, 0.25]));
        assert!(!meets([0.25, 0.25, 0.125]));
    }

    #[test]
    fn test_a_mistake_in_a_group_isnt_hidden_by_one_that_cant_be_met() {
        let bounds = WeightBounds::uniform(0.0, 0.4)
            .group(["A", "B"], 0.9, 1.0)
            .group(["A", "D"], 0.0, 1.0);
        assert!(matches!(
            bounds.resolve(&three()),
            Err(CutupError::UnknownAsset { .. })
        ));
    }

    #[test]
    fn test_grouped_min_variance_water_fills_within_the_groups() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let solve = |bounds: WeightBounds| {
            let method = MinVariance::default()
                .covariance(CovarianceConfig::default())
                .bounds(bounds);
            allocator.allocate(&method).unwrap()
        };
        // Unbounded, 4/7, 2/7 and 1/7 by inverse variance. Capping the first two at 50%
        // together splits that 2 : 1 between them, and leaves the rest to the third.
        assert_weights_close(
            &solve(WeightBounds::default().group([0, 1], 0.0, 0.5)),
            &Weights::from(vec![1.0 / 3.0, 1.0 / 6.0, 0.5]),
            1e-12,
        );
        // A floor of 60% on the last two leaves 40% to the first, and splits the rest 2 : 1.
        assert_weights_close(
            &solve(WeightBounds::default().group([1, 2], 0.6, 1.0)),
            &Weights::from(vec![0.4, 0.4, 0.2]),
            1e-12,
        );
        // Groups that don't bind change nothing.
        assert_weights_close(
            &solve(WeightBounds::default().group([0, 2], 0.0, 0.9)),
            &Weights::from(vec![4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]),
            1e-12,
        );
        // Each pair at least 0.7 can't hold at once.
        let pairs = WeightBounds::default()
            .group([0, 1], 0.7, 1.0)
            .group([1, 2], 0.7, 1.0)
            .group([0, 2], 0.7, 1.0);
        assert_eq!(
            allocator.allocate(&MinVariance::default().bounds(pairs)),
            Err(CutupError::InfeasibleGroups { group: None })
        );
    }

    #[test]
    fn test_grouped_mean_variance_shares_the_groups_multiplier() {
        // Risk aversion 1 on uncorrelated assets, variances 4, 8 and 16 in units of 1/30000:
        // each free weight is (μᵢ − ν)/σᵢ², ν the budget's multiplier. Long-only, the third
        // asset holds nothing. With the first two capped at 50% together, they share ν + η,
        // so with equal means they split it 2 : 1, and the third takes the rest.
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let method = |bounds: WeightBounds| {
            MeanVariance::risk_aversion(1.0)
                .expected_returns([0.0002, 0.0002, 0.0001])
                .covariance(CovarianceConfig::default())
                .bounds(bounds)
        };
        let free = allocator
            .allocate(&method(WeightBounds::default()))
            .unwrap();
        assert_eq!(free[2], 0.0);
        assert_weights_close(
            &allocator
                .allocate(&method(WeightBounds::default().group([0, 1], 0.0, 0.5)))
                .unwrap(),
            &Weights::from(vec![1.0 / 3.0, 1.0 / 6.0, 0.5]),
            1e-12,
        );
    }

    #[test]
    fn test_a_group_of_assets_at_their_floors_changes_nothing() {
        // A cap that doesn't bind on the assets mean-variance holds nothing of. Narrowed onto
        // the 0 its members' floors already force, the group's floor let the solver hold its
        // total there as well as every member, and the step had no unique solution.
        let mut failures = Vec::new();
        for seed in 0..10_u64 {
            let mut state = seed;
            let mut uniform = || {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
            };
            let n = 6 + (uniform() * 10.0) as usize;
            let rows: Vec<Vec<f64>> = (0..3 * n)
                .map(|_| {
                    let f = uniform() - 0.5;
                    (0..n)
                        .map(|i| {
                            let scale = 1.0 + i as f64 / n as f64;
                            0.0003 * i as f64 + 0.01 * f * scale + 0.01 * (uniform() - 0.5)
                        })
                        .collect()
                })
                .collect();
            let data = Returns::new(&rows).unwrap();
            let cap = (1.5 + uniform()) / n as f64;
            let gamma = 1.0 + 9.0 * uniform();
            for rebalance in [false, true] {
                let method = |bounds: WeightBounds| {
                    let method = MeanVariance::risk_aversion(gamma).bounds(bounds);
                    match rebalance {
                        true => method
                            .current_weights(vec![1.0 / n as f64; n])
                            .turnover_penalty(1e-4),
                        false => method,
                    }
                };
                let plain = WeightBounds::uniform(0.0, cap);
                let w = data.allocate(&method(plain.clone())).unwrap();
                let zeros: Vec<usize> = (0..n).filter(|&i| w[i] == 0.0).collect();
                match data.allocate(&method(plain.group(zeros, 0.0, 0.3))) {
                    Ok(grouped) => assert_weights_close(&grouped, &w, 1e-9),
                    Err(error) => failures.push((seed, rebalance, error)),
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn test_the_range_with_groups_comes_from_the_simplex_method() {
        // C at most 0.5: the top holds 0.5 of C and 0.5 of B, 2.5, and the bottom all of A, 1.
        let bounds = WeightBounds::default()
            .group(["C"], 0.0, 0.5)
            .resolve(&three())
            .unwrap();
        let mu = dvector![1.0, 2.0, 3.0];
        let range = ReturnRange::new(&mu, &bounds).unwrap();
        assert_eq!((range.lowest, range.highest), (1.0, 2.5));
        // 1e-12 of each end's Σ|μᵢwᵢ| and of the next asset a target inside moves into: A
        // from the top, and from the bottom B, whose reduced cost is the least.
        let (low, high) = range.tolerances;
        assert!(
            (low - 3e-12).abs() < 1e-24 && (high - 3.5e-12).abs() < 1e-24,
            "{low} {high}"
        );
        // The top holds A at 0 and the group at its cap, which leaves B and C one portfolio.
        let top = range.face(&mu, &bounds, true);
        assert_eq!((top.lower[0], top.upper[0]), (0.0, 0.0));
        assert_eq!((top.lower[1], top.upper[1]), (0.0, 1.0));
        assert_eq!((top.rows.lower[0], top.rows.upper[0]), (0.5, 0.5));
        // The bottom is all of A, whichever multipliers the end has. C is held, so the group's
        // total isn't, and the group keeps its limits.
        let bottom = range.face(&mu, &bounds, false);
        assert_eq!(feasible_start(&bottom).unwrap(), dvector![1.0, 0.0, 0.0]);
        assert_eq!((bottom.lower[2], bottom.upper[2]), (0.0, 0.0));
        assert_eq!(bottom.rows.upper, dvector![0.5]);
        assert_eq!(bottom.rows.lower[0], f64::NEG_INFINITY);
    }

    #[test]
    fn test_a_face_with_groups_leaves_tied_assets_to_the_solver() {
        // A and B tie at the top, so the group on A isn't held there, and minimum variance
        // splits them 2 : 1 up to its cap of 0.6.
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let mu = [0.0002, 0.0002, 0.0001];
        let grouped = WeightBounds::default().group([0], 0.0, 0.6);
        let bounds = grouped.resolve(allocator.labels()).unwrap();
        let range = ReturnRange::new(&DVector::from_row_slice(&mu), &bounds).unwrap();
        let top = range.face(&DVector::from_row_slice(&mu), &bounds, true);
        assert_eq!(
            (top.rows.lower[0], top.rows.upper[0]),
            (f64::NEG_INFINITY, 0.6)
        );
        let method = MeanVariance::target_return(0.0002)
            .expected_returns(mu)
            .covariance(CovarianceConfig::default())
            .bounds(grouped);
        assert_weights_close(
            &allocator.allocate(&method).unwrap(),
            &Weights::from(vec![0.6, 0.4, 0.0]),
            1e-12,
        );
    }

    #[test]
    fn test_the_next_asset_from_an_end_with_groups_has_room() {
        // Caps of 0.4, D fixed at 0.4, and A and B at least 0.3 together: the top holds C at
        // 0.3 and B at 0.3, 2.9. D's reduced cost, 0.5, is less than A's, 1, but D has no room
        // above its lower bound, so a target inside moves into A.
        let four: Vec<String> = ["A", "B", "C", "D"].map(String::from).to_vec();
        let bounds = WeightBounds::per_asset(vec![0.0, 0.0, 0.0, 0.4], vec![0.4; 4])
            .group(["A", "B"], 0.3, 1.0)
            .resolve(&four)
            .unwrap();
        let range = ReturnRange::new(&dvector![1.0, 2.0, 3.0, 3.5], &bounds).unwrap();
        assert!((range.highest - 2.9).abs() < 1e-15);
        assert!(
            (range.tolerances.1 - 3.9e-12).abs() < 1e-24,
            "{}",
            range.tolerances.1
        );
    }

    #[test]
    fn test_a_start_inside_the_range_with_groups_reaches_the_target() {
        let bounds = WeightBounds::uniform(0.0, 0.6)
            .group(["A", "B"], 0.0, 0.7)
            .resolve(&three())
            .unwrap();
        let mu = dvector![1.0, 2.0, 3.0];
        let problem = ReturnRange::target_problem(&mu, bounds.clone(), 2.1).unwrap();
        assert_eq!(problem.e.ncols(), 2);
        assert!((mu.dot(&problem.start) - 2.1).abs() < 1e-15);
        assert!(meets(&bounds, &problem.start, 1e-15));
        // At the top, the face, with only the budget.
        let top = ReturnRange::target_problem(&mu, bounds, 2.6).unwrap();
        assert_eq!(top.e.ncols(), 1);
    }

    #[test]
    fn test_a_range_with_groups_only_the_simplex_tolerance_meets_is_infeasible() {
        // A and B at least 0.6, and C and D at least 0.4 + 1e-10: the simplex method accepts
        // both, but no portfolio summing to 1 meets them, as feasible_start finds.
        let four: Vec<String> = ["A", "B", "C", "D"].map(String::from).to_vec();
        let bounds = WeightBounds::default()
            .group(["A", "B"], 0.6, 1.0)
            .group(["C", "D"], 0.4 + 1e-10, 1.0)
            .resolve(&four)
            .unwrap();
        let infeasible = CutupError::InfeasibleGroups { group: None };
        assert_eq!(feasible_start(&bounds).unwrap_err(), infeasible);
        let mu = dvector![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            ReturnRange::target_problem(&mu, bounds, 2.5).err(),
            Some(infeasible)
        );
    }

    #[test]
    fn test_a_reduced_cost_counts_beyond_rounding_of_its_terms() {
        // The first column's reduced cost, 1.5e-9, is within 1e-9 of its terms, 1 and 1, but
        // beyond 1e-9 of their difference. The second's, 3, counts, and its terms, 3, set the
        // floor at 3e-14, which the third's, 2e-14, is within.
        let a = DMatrix::from_row_slice(1, 3, &[1.0, -1.0, 0.0]);
        let y = dvector![1.0 - 1.5e-9];
        let cost = dvector![1.0, 2.0 - 1.5e-9, 2e-14];
        let (reduced, nonzero) = reduced_costs(&cost, &a, &y);
        assert!((reduced[0] - 1.5e-9).abs() < 1e-15);
        assert_eq!(nonzero, [false, true, false]);
        // A cost of 0 with no terms can't count.
        let (_, nonzero) = reduced_costs(&dvector![0.0], &DMatrix::zeros(1, 1), &dvector![0.0]);
        assert_eq!(nonzero, [false]);
    }

    #[test]
    fn test_a_group_whose_members_end_at_zero_changes_nothing() {
        // From a random search: a target return where a cap that doesn't bind is on assets
        // that end at 0. The solver's slack for the group came back at 4e-33 of rounding,
        // which the final check, measured against the row's own terms, took as all residual.
        let mut state = 129_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 5 + (uniform() * 8.0) as usize;
        let rows: Vec<Vec<f64>> = (0..3 * n)
            .map(|_| {
                let f = uniform() - 0.5;
                (0..n)
                    .map(|i| {
                        let scale = 1.0 + i as f64 / n as f64;
                        0.0003 * i as f64 + 0.01 * f * scale + 0.01 * (uniform() - 0.5)
                    })
                    .collect()
            })
            .collect();
        let data = Returns::new(&rows).unwrap();
        let cap = (1.5 + uniform()) / n as f64;
        let mu: Vec<f64> = (0..n)
            .map(|i| 0.0003 * i as f64 + 0.0002 * uniform())
            .collect();
        let method = |bounds: WeightBounds| {
            MeanVariance::target_return(0.0007182175668002149)
                .expected_returns(mu.clone())
                .bounds(bounds)
        };
        let plain = WeightBounds::uniform(0.0, cap);
        let w = data.allocate(&method(plain.clone())).unwrap();
        let zeros: Vec<usize> = (0..n).filter(|&i| w[i] == 0.0).collect();
        assert_eq!(zeros, [3, 5, 6, 7]);
        let grouped = data
            .allocate(&method(plain.group(zeros, 0.0, 0.3)))
            .unwrap();
        assert_weights_close(&grouped, &w, 1e-12);
    }

    #[test]
    fn test_min_cvar_without_groups_yet_says_so() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let grouped = WeightBounds::default().group([0, 1], 0.0, 0.5);
        assert_eq!(
            allocator.allocate(&crate::MinCvar::default().bounds(grouped)),
            Err(CutupError::NotSupported {
                setting: "group",
                with: "MinCvar",
            })
        );
    }

    #[test]
    fn test_invalid_bounds_are_rejected() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let n = full_rank_prices().ncols();
        let config = CovarianceConfig::default();
        let solve = |bounds: WeightBounds| {
            allocator.allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(bounds),
            )
        };
        let infeasible = |lower_sum: f64, upper_sum: f64| {
            Err(CutupError::InfeasibleBounds {
                lower_sum,
                upper_sum,
            })
        };

        let caps = 0.9 / n as f64;
        assert_eq!(
            solve(WeightBounds::uniform(0.0, caps)),
            infeasible(0.0, caps * n as f64)
        );
        let floors = 1.1 / n as f64;
        assert_eq!(
            solve(WeightBounds::uniform(floors, 1.0)),
            infeasible(floors * n as f64, n as f64)
        );
        assert_eq!(
            solve(WeightBounds::uniform(-0.1, 1.0)),
            Err(CutupError::InvalidConfig {
                field: "lower_bound",
                value: -0.1
            })
        );
        assert_eq!(
            solve(WeightBounds::uniform(0.3, 0.2)),
            Err(CutupError::InvalidConfig {
                field: "upper_bound",
                value: 0.2
            })
        );
        assert!(matches!(
            solve(WeightBounds::uniform(0.0, f64::NAN)),
            Err(CutupError::InvalidConfig {
                field: "upper_bound",
                ..
            })
        ));
        assert!(matches!(
            solve(WeightBounds::uniform(f64::INFINITY, f64::INFINITY)),
            Err(CutupError::InvalidConfig {
                field: "lower_bound",
                ..
            })
        ));
        assert_eq!(
            solve(WeightBounds::per_asset(vec![0.0; n + 1], vec![1.0; n + 1])),
            Err(CutupError::DimensionMismatch {
                field: "lower_bound",
                expected: n,
                actual: n + 1
            })
        );

        // Caps or floors short of the budget by less than 1e-12 are rescaled onto it, giving
        // equal weights within 1e-12 of each bound.
        for (lower, upper) in [
            (0.0, (1.0 - 5e-13) / n as f64),
            (0.0, (1.0 - 9e-13) / n as f64),
            ((1.0 + 9e-13) / n as f64, 1.0),
        ] {
            let weights = solve(WeightBounds::uniform(lower, upper)).unwrap();
            assert_weights_close(&weights, &Weights::from(vec![1.0 / n as f64; n]), 1e-15);
            for &w in &weights {
                assert!(
                    w >= lower * (1.0 - 1e-12) && w <= upper * (1.0 + 1e-12),
                    "{w}"
                );
            }
        }

        // Within rounding of a budget of 1 is feasible: ten caps of 0.1 sum to just under it.
        let ten = Returns::returns_matrix(DMatrix::from_fn(12, 10, |i, j| {
            ((i * 7 + j * 3) % 11) as f64 * 0.001 - 0.005 + (i == j) as u8 as f64 * 0.01
        }))
        .unwrap();
        let weights = ten
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(WeightBounds::uniform(0.0, 0.1)),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.1; 10]), 1e-15);
    }
}
