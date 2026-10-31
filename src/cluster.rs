//! Agglomerative clustering for HRP and HERC.
//!
//! Reproduces SciPy's `scipy.cluster.hierarchy.linkage` followed by
//! `to_tree().pre_order()` step for step, including how ties are broken, so the leaf order
//! matches SciPy and PyPortfolioOpt's `HRPOpt` exactly. `cut` and `cluster_count` follow
//! SciPy's `cut_tree` and riskfolio-lib's gap statistic for HERC.

use nalgebra::DMatrix;

/// How the distance between two clusters is measured when HRP or HERC builds its hierarchy.
/// See [`Hrp::linkage`](crate::Hrp::linkage) and [`Herc::linkage`](crate::Herc::linkage).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Linkage {
    /// The distance between the two closest members. The default, as in López de Prado (2016)
    /// and PyPortfolioOpt. Tends to chain assets together one at a time.
    #[default]
    Single,
    /// The distance between the two farthest members. Gives compact clusters of similar size.
    Complete,
    /// The average distance between all pairs of members (UPGMA).
    Average,
    /// Ward's method: merge the pair of clusters that least increases the within-cluster sum
    /// of squares. Applied to the correlation distances, as SciPy and PyPortfolioOpt do.
    Ward,
}

/// One merge: the two clusters joined and the distance between them.
type Merge = (usize, usize, f64);

/// A clustering hierarchy over `n` assets. Merge `k` joins two clusters into cluster `n + k`,
/// and merges are ordered by increasing distance, so the last merge creates the root.
pub(crate) struct Hierarchy {
    /// The two clusters each merge joins, the one with the smaller id first, as SciPy labels
    /// them.
    children: Vec<(usize, usize)>,
}

/// Builds the hierarchy of `distance` with `linkage`.
///
/// `distance` must be symmetric, finite and non-negative.
pub(crate) fn hierarchy(distance: &DMatrix<f64>, linkage: Linkage) -> Hierarchy {
    let n = distance.nrows();
    let mut merges = match linkage {
        Linkage::Single => minimum_spanning_tree(distance),
        _ => nearest_neighbor_chain(distance, linkage),
    };
    // Stable, like SciPy's mergesort, so equal distances keep the order they were found in.
    merges.sort_by(|a, b| a.2.total_cmp(&b.2));

    let mut parent: Vec<usize> = (0..2 * n - 1).collect();
    let find = |parent: &mut Vec<usize>, mut node: usize| {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    };
    let mut children = Vec::with_capacity(n - 1);
    for (k, &(a, b, _)) in merges.iter().enumerate() {
        let (root_a, root_b) = (find(&mut parent, a), find(&mut parent, b));
        parent[root_a] = n + k;
        parent[root_b] = n + k;
        children.push((root_a.min(root_b), root_a.max(root_b)));
    }
    Hierarchy { children }
}

impl Hierarchy {
    /// The number of assets clustered.
    fn assets(&self) -> usize {
        self.children.len() + 1
    }

    /// The two clusters merged into `node`, which must not be a leaf.
    pub(crate) fn children(&self, node: usize) -> (usize, usize) {
        self.children[node - self.assets()]
    }

    /// The assets in cluster `node`, left to right, as SciPy's `pre_order` lists them.
    pub(crate) fn leaves(&self, node: usize) -> Vec<usize> {
        let n = self.assets();
        let mut order = Vec::new();
        let mut stack = vec![node];
        while let Some(node) = stack.pop() {
            if node < n {
                order.push(node);
            } else {
                let (left, right) = self.children[node - n];
                stack.push(right);
                stack.push(left);
            }
        }
        order
    }

    /// Every asset, left to right.
    pub(crate) fn leaf_order(&self) -> Vec<usize> {
        self.leaves(2 * self.assets() - 2)
    }

    /// The `k` clusters left after the first `n − k` merges, as SciPy's `cut_tree` gives them
    /// for a monotonic hierarchy, each with the id of the node that formed it. `k` must be
    /// between 1 and `n`.
    pub(crate) fn cut(&self, k: usize) -> Vec<(usize, Vec<usize>)> {
        self.merge_down_to(k, |_, _| {})
            .into_iter()
            .enumerate()
            .filter_map(|(node, members)| members.map(|members| (node, members)))
            .collect()
    }

    /// Replays the merges until `k` clusters remain, calling `visit` with the number of
    /// clusters and the clusters, indexed by node, at each level from `n` down to `k`.
    fn merge_down_to(
        &self,
        k: usize,
        mut visit: impl FnMut(usize, &[Option<Vec<usize>>]),
    ) -> Vec<Option<Vec<usize>>> {
        let n = self.assets();
        let mut clusters: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
        clusters.resize(2 * n - 1, None);
        visit(n, &clusters);
        for (m, &(a, b)) in self.children[..n - k].iter().enumerate() {
            let mut merged = clusters[a].take().expect("each cluster merges once");
            merged.extend(clusters[b].take().expect("each cluster merges once"));
            clusters[n + m] = Some(merged);
            visit(n - m - 1, &clusters);
        }
        clusters
    }

    /// Chooses the number of clusters with the two-difference gap statistic (Yue et al.,
    /// 2008), exactly as riskfolio-lib's `two_diff_gap_stat` does.
    ///
    /// For each `k` up to `min(max_clusters, √n) + 2`, the dispersion `W_k` sums, over the
    /// `k` clusters, the population standard deviation of the distances within each cluster.
    /// The chosen `k` maximizes `W_{k+2} + W_k − 2 W_{k+1}`, with `W_1 = −∞`, and is at most
    /// `min(max_clusters, √n)`. The smallest `k` wins a tie, and 1 is returned when there are
    /// too few assets to compare.
    pub(crate) fn cluster_count(&self, distance: &DMatrix<f64>, max_clusters: usize) -> usize {
        let n = self.assets();
        let most = (max_clusters as f64).min((n as f64).sqrt()) + 2.0;
        // Levels 1..=levels; the cuts are nested, so one replay of the merges visits them all.
        let levels = (1..=n).take_while(|&k| k as f64 <= most).count();
        let mut dispersion = vec![f64::NEG_INFINITY; levels];
        self.merge_down_to(2.min(n), |k, clusters| {
            if (2..=levels).contains(&k) {
                dispersion[k - 1] = clusters
                    .iter()
                    .flatten()
                    .filter(|cluster| cluster.len() > 1)
                    .map(|cluster| spread(distance, cluster))
                    .sum();
            }
        });

        let (mut best_k, mut best_gap) = (1, f64::NEG_INFINITY);
        for (i, w) in dispersion.windows(3).enumerate() {
            let gap = w[2] + w[0] - 2.0 * w[1];
            if gap > best_gap {
                (best_k, best_gap) = (i + 1, gap);
            }
        }
        best_k
    }
}

/// The population standard deviation of the distances between each pair in `cluster`.
fn spread(distance: &DMatrix<f64>, cluster: &[usize]) -> f64 {
    let pairs: Vec<f64> = cluster
        .iter()
        .enumerate()
        .flat_map(|(a, &i)| cluster[a + 1..].iter().map(move |&j| distance[(i, j)]))
        .collect();
    let mean = pairs.iter().sum::<f64>() / pairs.len() as f64;
    let variance = pairs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / pairs.len() as f64;
    variance.sqrt()
}

/// Single linkage through Prim's minimum spanning tree, as in SciPy's `mst_single_linkage`:
/// start from point 0 and, among equally near points, take the lowest index.
fn minimum_spanning_tree(distance: &DMatrix<f64>) -> Vec<Merge> {
    let n = distance.nrows();
    let mut merges = Vec::with_capacity(n - 1);
    let mut distance_to_tree = vec![f64::INFINITY; n];
    let mut in_tree = vec![false; n];
    let mut x = 0;
    for _ in 1..n {
        in_tree[x] = true;
        let (mut y, mut nearest) = (0, f64::INFINITY);
        for (i, to_tree) in distance_to_tree.iter_mut().enumerate() {
            if in_tree[i] {
                continue;
            }
            *to_tree = to_tree.min(distance[(x, i)]);
            if *to_tree < nearest {
                (y, nearest) = (i, *to_tree);
            }
        }
        merges.push((x, y, nearest));
        x = y;
    }
    merges
}

/// Complete, average and Ward linkage through the nearest-neighbor chain algorithm, as in
/// SciPy's `nn_chain`.
///
/// The chain follows nearest neighbors until two clusters are each other's nearest, then
/// merges them. The merged cluster takes the slot of the higher-indexed one, and its
/// distances are updated with the Lance-Williams formula for the linkage.
fn nearest_neighbor_chain(distance: &DMatrix<f64>, linkage: Linkage) -> Vec<Merge> {
    let n = distance.nrows();
    let mut d = distance.clone();
    let mut size = vec![1usize; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut merges = Vec::with_capacity(n - 1);

    for _ in 1..n {
        if chain.is_empty() {
            chain.push(size.iter().position(|&s| s > 0).expect("a cluster remains"));
        }
        let (x, y, nearest) = loop {
            let x = chain[chain.len() - 1];
            // Prefer the previous link on a tie, so the chain can't cycle.
            let (mut y, mut nearest) = match chain.len() {
                1 => (x, f64::INFINITY),
                len => (chain[len - 2], d[(x, chain[len - 2])]),
            };
            for i in 0..n {
                if size[i] > 0 && i != x && d[(x, i)] < nearest {
                    (y, nearest) = (i, d[(x, i)]);
                }
            }
            if chain.len() > 1 && y == chain[chain.len() - 2] {
                break (x, y, nearest);
            }
            chain.push(y);
        };
        chain.truncate(chain.len() - 2);

        let (x, y) = (x.min(y), x.max(y));
        let (nx, ny) = (size[x] as f64, size[y] as f64);
        merges.push((x, y, nearest));
        // The merged cluster lives on in slot y; slot x is retired.
        size[y] += size[x];
        size[x] = 0;

        for i in 0..n {
            if size[i] == 0 || i == y {
                continue;
            }
            let ni = size[i] as f64;
            let (dxi, dyi) = (d[(i, x)], d[(i, y)]);
            // Written in the same order of operations as SciPy's update functions, so the
            // rounding, and with it every near-tie, comes out the same.
            let updated = match linkage {
                Linkage::Complete => dxi.max(dyi),
                Linkage::Average => (nx * dxi + ny * dyi) / (nx + ny),
                Linkage::Ward => {
                    let t = 1.0 / (nx + ny + ni);
                    ((ni + nx) * t * dxi * dxi + (ni + ny) * t * dyi * dyi
                        - ni * t * nearest * nearest)
                        .sqrt()
                }
                Linkage::Single => unreachable!("single linkage uses the minimum spanning tree"),
            };
            d[(i, y)] = updated;
            d[(y, i)] = updated;
        }
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distances for points on a line at 0, 1, 10, 11, 30, so single linkage merges (0, 1),
    /// (2, 3), then those two pairs, then 4.
    fn line() -> DMatrix<f64> {
        let x = [0.0, 1.0, 10.0, 11.0, 30.0];
        DMatrix::from_fn(5, 5, |i, j| f64::abs(x[i] - x[j]))
    }

    fn sorted(clusters: Vec<(usize, Vec<usize>)>) -> Vec<Vec<usize>> {
        let mut clusters: Vec<Vec<usize>> = clusters.into_iter().map(|(_, c)| c).collect();
        clusters.iter_mut().for_each(|c| c.sort_unstable());
        clusters.sort();
        clusters
    }

    #[test]
    fn cut_gives_k_clusters() {
        let h = hierarchy(&line(), Linkage::Single);
        assert_eq!(
            sorted(h.cut(5)),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
        assert_eq!(sorted(h.cut(3)), vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(sorted(h.cut(2)), vec![vec![0, 1, 2, 3], vec![4]]);
        assert_eq!(sorted(h.cut(1)), vec![vec![0, 1, 2, 3, 4]]);
        assert_eq!(h.cut(2)[1].0, 7);
        assert_eq!(h.children(8), (4, 7));
        assert_eq!(h.leaves(7), vec![0, 1, 2, 3]);
        assert_eq!(h.leaf_order(), vec![4, 0, 1, 2, 3]);
    }

    #[test]
    fn cluster_count_follows_the_gap_statistic() {
        let distance = line();
        let h = hierarchy(&distance, Linkage::Single);
        // W₁..W₄ are computed, up to min(10, √5) + 2 ≈ 4.2. The k = 1 gap is −∞, so k = 2
        // wins with W₄ + W₂ − 2W₃, which is finite.
        assert_eq!(h.cluster_count(&distance, 10), 2);
        // With max_clusters = 1 only W₁..W₃ are computed, leaving the k = 1 gap of −∞.
        assert_eq!(h.cluster_count(&distance, 1), 1);
    }

    #[test]
    fn cluster_count_with_too_few_assets_is_one() {
        for n in [2, 3] {
            let distance = DMatrix::from_fn(n, n, |i, j| if i == j { 0.0 } else { 0.5 });
            assert_eq!(
                hierarchy(&distance, Linkage::Average).cluster_count(&distance, 10),
                1
            );
        }
    }

    #[test]
    fn spread_is_the_population_standard_deviation() {
        // The distances between 0, 1 and 10 are 1, 10 and 9.
        let s = spread(&line(), &[0, 1, 2]);
        let mean = (1.0 + 10.0 + 9.0) / 3.0;
        let expected =
            (((1.0f64 - mean).powi(2) + (10.0f64 - mean).powi(2) + (9.0f64 - mean).powi(2)) / 3.0)
                .sqrt();
        assert!((s - expected).abs() < 1e-15);
    }
}
