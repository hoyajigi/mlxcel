// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The shared candidate pool of the GLiNER2.5 boundary head, upstream
//! `DocumentCandidatePool` in `mlx_vlm/models/gliner2_5/boundary.py`.
//!
//! The top boundaries by start and by end score (over all labels) are paired,
//! every valid pair (`end > start`) is ranked by a global score and every
//! label additionally reserves its own best `min_pool_per_query` pairs, and
//! the first `pool_size` survivors form the pool every label is then scored
//! against. Upstream does this ranking on the host after one device readback,
//! and so does this port: the selection order (stable sorts, the pair grid
//! read start-major, dictionary insertion order on ties) is reproduced
//! exactly, because it decides which spans can be returned at all.

use anyhow::{Result, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::gliner2_5_layers::{BoundaryHeadSettings, MASK_LOGIT, linear, scalar};

/// The pool: `pool_size` boundary pairs, the first `count` of them real.
pub(crate) struct PooledCandidates {
    /// `(start, end)` boundary indices, `(0, 0)` past `count`; the length is
    /// `min(pool_size, top_k * top_k)`.
    pub pairs: Vec<(usize, usize)>,
    pub count: usize,
    /// `[P]` int32 starts.
    pub starts: UniquePtr<MlxArray>,
    /// `[P]` int32 ends.
    pub ends: UniquePtr<MlxArray>,
    /// `[P]` bool, true for the real entries.
    pub mask: UniquePtr<MlxArray>,
    /// `[P]` start / end compatibility, zero on the padding.
    pub compat_logits: UniquePtr<MlxArray>,
}

pub(crate) struct DocumentCandidatePool {
    start_projection: UnifiedLinear,
    end_projection: UnifiedLinear,
    pool_boundary_top_k: usize,
    pool_size: usize,
    min_pool_per_query: usize,
    sqrt_dim: f32,
}

/// Indices of `values` by descending value; equal values keep index order,
/// as `mx.argsort(-values)` does for the ties that matter here.
fn argsort_descending(values: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| (-values[a]).total_cmp(&-values[b]));
    order
}

impl DocumentCandidatePool {
    pub fn from_weights(
        weights: &WeightMap,
        settings: &BoundaryHeadSettings,
        q: (i32, i32),
    ) -> Result<Self> {
        let p = "boundary_head.shared_pool_builder";
        Ok(Self {
            start_projection: linear(weights, &format!("{p}.start_projection"), q.0, q.1)?,
            end_projection: linear(weights, &format!("{p}.end_projection"), q.0, q.1)?,
            pool_boundary_top_k: settings.pool_boundary_top_k,
            pool_size: settings.pool_size,
            min_pool_per_query: settings.min_pool_per_query,
            sqrt_dim: (settings.boundary_dim as f32).sqrt(),
        })
    }

    /// `sum(projected_start[starts] * projected_end[ends], -1) / sqrt(dim)`.
    fn compatibility(
        &self,
        projected_start: &MlxArray,
        projected_end: &MlxArray,
        starts: &MlxArray,
        ends: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let product = mlxcel_core::multiply(
            &mlxcel_core::take(projected_start, starts, 0),
            &mlxcel_core::take(projected_end, ends, 0),
        );
        let summed = mlxcel_core::sum_axis(&product, -1, false);
        // Divide, as upstream does: `x * (1 / s)` can differ by an ulp, and
        // the host ranking below compares these values.
        mlxcel_core::divide(&summed, &scalar(self.sqrt_dim, &summed))
    }

    /// Build the pool for `boundary_states` (`[T + 1, dim]`) from the
    /// `[Q, T + 1]` start and end logits.
    pub fn build(
        &self,
        boundary_states: &MlxArray,
        start_logits: &MlxArray,
        end_logits: &MlxArray,
    ) -> Result<PooledCandidates> {
        let boundaries = mlxcel_core::array_shape(boundary_states)[0] as usize;
        let shape = mlxcel_core::array_shape(start_logits);
        let queries = shape[0] as usize;
        if shape[1] as usize != boundaries || mlxcel_core::array_shape(end_logits) != shape {
            bail!(
                "gliner2_5 pool: start / end logits {shape:?} do not cover {boundaries} boundaries"
            );
        }
        let projected_start = self.start_projection.forward(boundary_states);
        let projected_end = self.end_projection.forward(boundary_states);
        // The readbacks below evaluate unconditionally; surface an MLX error
        // (a checkpoint whose tensors disagree with its config, or memory) as
        // an `Err` instead of an abort.
        for logits in [start_logits, end_logits] {
            mlxcel_core::try_eval(logits)
                .map_err(|e| anyhow::anyhow!("gliner2_5 pool: evaluation failed: {e}"))?;
        }
        let start_host = array_to_vec_f32(start_logits);
        let end_host = array_to_vec_f32(end_logits);
        let union = |plane: &[f32]| -> Vec<f32> {
            (0..boundaries)
                .map(|b| {
                    (0..queries)
                        .map(|q| plane[q * boundaries + b])
                        .fold(MASK_LOGIT, f32::max)
                })
                .collect()
        };
        let union_start = union(&start_host);
        let union_end = union(&end_host);

        let top_k = self.pool_boundary_top_k.min(boundaries);
        let starts: Vec<usize> = argsort_descending(&union_start)
            .into_iter()
            .take(top_k)
            .collect();
        let ends: Vec<usize> = argsort_descending(&union_end)
            .into_iter()
            .take(top_k)
            .collect();
        // The pair grid, start-major: pair `i` is `(starts[i / k], ends[i % k])`.
        let pair_start: Vec<usize> = starts
            .iter()
            .flat_map(|&s| std::iter::repeat_n(s, top_k))
            .collect();
        let pair_end: Vec<usize> = (0..top_k).flat_map(|_| ends.iter().copied()).collect();
        let as_array = |indices: &[usize]| {
            let values: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
            mlxcel_core::from_slice_i32(&values, &[values.len() as i32])
        };
        let compatibility = array_to_vec_f32(&self.compatibility(
            &projected_start,
            &projected_end,
            &as_array(&pair_start),
            &as_array(&pair_end),
        ));

        // Valid pairs in grid order, with their global priority. `starts` and
        // `ends` are permutations, so every `(s, e)` is unique and
        // `priorities[row]` is upstream's `priorities[(s, e)]`.
        let pair_rows: Vec<(usize, usize, usize)> = (0..pair_start.len())
            .filter(|&i| pair_end[i] > pair_start[i])
            .map(|i| (pair_start[i], pair_end[i], i))
            .collect();
        let mut priorities: Vec<((usize, usize), f32)> = pair_rows
            .iter()
            .map(|&(s, e, i)| ((s, e), compatibility[i] + union_start[s] + union_end[e]))
            .collect();
        let quota = self.min_pool_per_query.min(pair_rows.len());
        if quota > 0 {
            for q in 0..queries {
                let scores: Vec<f32> = pair_rows
                    .iter()
                    .map(|&(s, e, i)| {
                        start_host[q * boundaries + s]
                            + end_host[q * boundaries + e]
                            + compatibility[i]
                    })
                    .collect();
                for (rank, row) in argsort_descending(&scores)
                    .into_iter()
                    .take(quota)
                    .enumerate()
                {
                    let entry = &mut priorities[row];
                    entry.1 = entry.1.max(5000.0 + (quota - rank) as f32);
                }
            }
        }
        let ranked: Vec<f32> = priorities.iter().map(|(_, priority)| *priority).collect();
        let mut pairs: Vec<(usize, usize)> = argsort_descending(&ranked)
            .into_iter()
            .take(self.pool_size)
            .map(|index| priorities[index].0)
            .collect();
        let count = pairs.len();
        // Upstream pads to `pool_size`; every scorer op is per row and the
        // padding is masked, so padding only to what the grid can hold gives
        // the same logits for the real rows without a `pool_size`-sized pool
        // for a short text.
        let width = self.pool_size.min(top_k * top_k).max(1);
        pairs.resize(width, (0, 0));

        let starts = as_array(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let ends = as_array(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
        let mask_host: Vec<i32> = (0..width).map(|i| i32::from(i < count)).collect();
        let mask = mlxcel_core::astype(
            &mlxcel_core::from_slice_i32(&mask_host, &[width as i32]),
            dtype::BOOL,
        );
        let compat = self.compatibility(&projected_start, &projected_end, &starts, &ends);
        let compat_logits = mlxcel_core::where_cond(&mask, &compat, &scalar(0.0, &compat));
        Ok(PooledCandidates {
            pairs,
            count,
            starts,
            ends,
            mask,
            compat_logits,
        })
    }
}
