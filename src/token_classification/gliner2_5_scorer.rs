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

//! `SharedPoolScorer` of the GLiNER2.5 boundary head (upstream
//! `boundary.py`): one logit per pooled span and label, from the boundary
//! representations, span length features, the pool's compatibility prior, the
//! pooled span content, a FiLM-conditioned query interaction and the
//! start / end / inside marginals.
//!
//! Includes upstream's fix for the candidate / query transpose
//! (Blaizzy/mlx-vlm#2020, second commit): the gathers over the `[Q,
//! positions]` marginal planes are transposed into the `[P, Q]` layout of the
//! score before they are added.

use anyhow::Result;
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::gliner2_5_head::Marginals;
use super::gliner2_5_layers::{
    BoundaryHeadSettings, MASK_LOGIT, layer_norm, linear, scalar, slice_last,
};
use super::gliner2_5_pool::PooledCandidates;

/// Mean of the projected word states inside each span, LayerNormed.
struct SpanContentPooler {
    value_projection: UnifiedLinear,
    layer_norm: LayerNorm,
}

impl SpanContentPooler {
    fn from_weights(weights: &WeightMap, prefix: &str, q: (i32, i32)) -> Result<Self> {
        Ok(Self {
            value_projection: linear(weights, &format!("{prefix}.value_projection"), q.0, q.1)?,
            layer_norm: layer_norm(weights, &format!("{prefix}.layer_norm"))?,
        })
    }

    /// `[P, content_dim]` for the pooled spans.
    fn forward(
        &self,
        text_states: &MlxArray,
        starts: &MlxArray,
        ends: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let values = self.value_projection.forward(text_states);
        let width = mlxcel_core::array_shape(&values)[1];
        let prefix = mlxcel_core::utils::concatenate(
            &mlxcel_core::zeros(&[1, width], dtype::FLOAT32),
            &mlxcel_core::cumsum(
                &mlxcel_core::astype(&values, dtype::FLOAT32),
                0,
                false,
                true,
            ),
            0,
        );
        let span_sum = mlxcel_core::subtract(
            &mlxcel_core::take(&prefix, ends, 0),
            &mlxcel_core::take(&prefix, starts, 0),
        );
        let length = mlxcel_core::maximum(
            &mlxcel_core::subtract(ends, starts),
            &mlxcel_core::from_slice_i32(&[1], &[1]),
        );
        let length = mlxcel_core::expand_dims(&mlxcel_core::astype(&length, dtype::FLOAT32), -1);
        let mean = mlxcel_core::astype(
            &mlxcel_core::divide(&span_sum, &length),
            mlxcel_core::array_dtype(&values),
        );
        self.layer_norm.forward(&mean)
    }
}

/// One logit per `(pooled span, label)`.
pub(crate) struct SharedPoolScorer {
    start_projection: UnifiedLinear,
    end_projection: UnifiedLinear,
    length_projection: UnifiedLinear,
    prior_projection: UnifiedLinear,
    content_pooler: SpanContentPooler,
    content_projection: UnifiedLinear,
    candidate_norm: LayerNorm,
    query_projection: UnifiedLinear,
    film: UnifiedLinear,
    film_hidden: UnifiedLinear,
    film_output: UnifiedLinear,
    pair_dim: i32,
}

impl SharedPoolScorer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        settings: &BoundaryHeadSettings,
        q: (i32, i32),
    ) -> Result<Self> {
        let p = "boundary_head.shared_pool_scorer";
        let load = |name: &str| linear(weights, &format!("{p}.{name}"), q.0, q.1);
        Ok(Self {
            start_projection: load("start_projection")?,
            end_projection: load("end_projection")?,
            length_projection: load("length_projection")?,
            prior_projection: load("prior_projection")?,
            content_pooler: SpanContentPooler::from_weights(
                weights,
                &format!("{p}.content_pooler"),
                q,
            )?,
            content_projection: load("content_projection")?,
            candidate_norm: layer_norm(weights, &format!("{p}.candidate_norm"))?,
            query_projection: load("query_projection")?,
            film: load("film")?,
            film_hidden: load("film_output.0")?,
            film_output: load("film_output.3")?,
            pair_dim: settings.pair_dim as i32,
        })
    }

    /// `[P, Q]` logits, `MASK_LOGIT` on the padded pool entries.
    pub(crate) fn forward(
        &self,
        boundaries: &MlxArray,
        queries: &MlxArray,
        pooled: &PooledCandidates,
        marginals: &Marginals,
        text_states: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let (starts, ends) = (&pooled.starts, &pooled.ends);
        let words = mlxcel_core::array_shape(text_states)[0];
        let start_rep = mlxcel_core::take(&self.start_projection.forward(boundaries), starts, 0);
        let end_rep = mlxcel_core::take(&self.end_projection.forward(boundaries), ends, 0);
        let one = mlxcel_core::from_slice_i32(&[1], &[1]);
        let lengths = mlxcel_core::astype(
            &mlxcel_core::maximum(&mlxcel_core::subtract(ends, starts), &one),
            mlxcel_core::array_dtype(&start_rep),
        );
        let text_length = scalar(words.max(1) as f32, &lengths);
        let features = mlxcel_core::utils::stack_arrays(
            &[
                mlxcel_core::log1p(&lengths),
                mlxcel_core::divide(&lengths, &text_length),
                mlxcel_core::rsqrt(&lengths),
            ],
            -1,
        );
        let content = self.content_pooler.forward(text_states, starts, ends);
        let mut candidate = mlxcel_core::add(&start_rep, &end_rep);
        candidate = mlxcel_core::add(&candidate, &self.length_projection.forward(&features));
        candidate = mlxcel_core::add(
            &candidate,
            &self
                .prior_projection
                .forward(&mlxcel_core::expand_dims(&pooled.compat_logits, -1)),
        );
        candidate = mlxcel_core::add(&candidate, &self.content_projection.forward(&content));
        let pool_mask = mlxcel_core::expand_dims(&pooled.mask, -1);
        let candidate = mlxcel_core::multiply(
            &self.candidate_norm.forward(&candidate),
            &mlxcel_core::astype(&pool_mask, mlxcel_core::array_dtype(&candidate)),
        );

        let query = self.query_projection.forward(queries);
        let mut score = mlxcel_core::matmul(&candidate, &mlxcel_core::swap_axes(&query, -1, -2));
        score = mlxcel_core::divide(&score, &scalar((self.pair_dim as f32).sqrt(), &score));
        let film = self.film.forward(&query);
        let gamma = mlxcel_core::expand_dims(&slice_last(&film, 0, self.pair_dim), 0);
        let beta =
            mlxcel_core::expand_dims(&slice_last(&film, self.pair_dim, 2 * self.pair_dim), 0);
        let conditioned = mlxcel_core::add(
            &mlxcel_core::multiply(
                &mlxcel_core::expand_dims(&candidate, 1),
                &mlxcel_core::add(&scalar(1.0, &gamma), &gamma),
            ),
            &beta,
        );
        let film = self
            .film_output
            .forward(&mlxcel_core::gelu(&self.film_hidden.forward(&conditioned)));
        score = mlxcel_core::add(&score, &mlxcel_core::squeeze_axis(&film, -1));

        // Gathers over the `[Q, positions]` marginals come back `[Q, P]`;
        // transpose them into the candidate-major layout of `score`.
        let gather_t = |plane: &MlxArray, index: &MlxArray| {
            mlxcel_core::swap_axes(&mlxcel_core::take(plane, index, 1), -1, -2)
        };
        score = mlxcel_core::add(&score, &gather_t(&marginals.start_logits, starts));
        score = mlxcel_core::add(&score, &gather_t(&marginals.end_logits, ends));
        let inside = mlxcel_core::subtract(
            &gather_t(&marginals.inside_prefix, ends),
            &gather_t(&marginals.inside_prefix, starts),
        );
        let lengths_f32 = mlxcel_core::astype(&lengths, dtype::FLOAT32);
        let interval = mlxcel_core::add(
            &inside,
            &mlxcel_core::swap_axes(
                &mlxcel_core::multiply(
                    &marginals.inside_mean,
                    &mlxcel_core::expand_dims(&lengths_f32, 0),
                ),
                -1,
                -2,
            ),
        );
        let interval = mlxcel_core::divide(
            &interval,
            &mlxcel_core::expand_dims(&mlxcel_core::sqrt(&lengths_f32), -1),
        );
        score = mlxcel_core::add(
            &score,
            &mlxcel_core::astype(&interval, mlxcel_core::array_dtype(&score)),
        );
        mlxcel_core::where_cond(&pool_mask, &score, &scalar(MASK_LOGIT, &score))
    }
}
