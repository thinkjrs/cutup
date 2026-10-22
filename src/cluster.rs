//! Agglomerative clustering for HRP.
//!
//! Reproduces SciPy's `scipy.cluster.hierarchy.linkage` followed by
//! `to_tree().pre_order()` step for step, including how ties are broken, so the leaf order
//! matches SciPy and PyPortfolioOpt's `HRPOpt` exactly.

use nalgebra::DMatrix;

/// How the distance between two clusters is measured when HRP builds its hierarchy. See
/// [`HrpConfig`](crate::HrpConfig).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

/// Returns the leaves of the hierarchy built from `distance`, left to right.
///
/// `distance` must be symmetric, finite and non-negative. Merge `k` creates cluster `n + k`.
/// Its children are the two merged clusters, the one with the smaller id on the left, as
/// SciPy labels them.
pub(crate) fn leaf_order(distance: &DMatrix<f64>, linkage: Linkage) -> Vec<usize> {
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

    let mut order = Vec::with_capacity(n);
    let mut stack = vec![2 * n - 2];
    while let Some(node) = stack.pop() {
        if node < n {
            order.push(node);
        } else {
            let (left, right) = children[node - n];
            stack.push(right);
            stack.push(left);
        }
    }
    order
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
