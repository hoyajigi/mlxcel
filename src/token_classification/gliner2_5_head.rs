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

//! The GLiNER2.5 boundary head, ported from
//! `mlx_vlm/models/gliner2_5/boundary.py`: `BoundaryEncoder` (boundary
//! states between words), `BoundaryQueryHead` (per-label start / end / inside
//! marginals), the shared candidate pool
//! ([`super::gliner2_5_pool::DocumentCandidatePool`]) and
//! [`super::gliner2_5_scorer::SharedPoolScorer`] (one logit per pooled span
//! and label); the settings and loading helpers are in
//! [`super::gliner2_5_layers`].
//!
//! Everything runs on one sequence (`batch 1`), which is all `_prepare`
//! produces, so the batch axis is dropped after the encoder: text states are
//! `[T, H]`, boundaries `[T + 1, dim]`, queries `[Q, H]`. The text and query
//! masks of the upstream port are therefore all true and folded away.

use anyhow::{Result, bail};
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::gliner2_5_layers::{
    BoundaryHeadSettings, MASK_LOGIT, layer_norm, linear, scalar, slice_last, tensor,
};
use super::gliner2_5_pool::{DocumentCandidatePool, PooledCandidates};
use super::gliner2_5_scorer::SharedPoolScorer;

/// Pre-norm SwiGLU block with a residual: `x + out(value * silu(gate))`.
struct ResidualSwiGlu {
    norm: LayerNorm,
    input_projection: UnifiedLinear,
    output_projection: UnifiedLinear,
    hidden: i32,
}

impl ResidualSwiGlu {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dim: usize,
        multiplier: f64,
        q: (i32, i32),
    ) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(weights, &format!("{prefix}.norm"))?,
            input_projection: linear(weights, &format!("{prefix}.input_projection"), q.0, q.1)?,
            output_projection: linear(weights, &format!("{prefix}.output_projection"), q.0, q.1)?,
            hidden: ((dim as f64 * multiplier) as i32).max(1),
        })
    }

    fn forward(&self, states: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.input_projection.forward(&self.norm.forward(states));
        let value = slice_last(&projected, 0, self.hidden);
        let gate = slice_last(&projected, self.hidden, 2 * self.hidden);
        let activated = mlxcel_core::multiply(&value, &mlxcel_core::silu(&gate));
        mlxcel_core::add(states, &self.output_projection.forward(&activated))
    }
}

/// Pre-norm self-attention over the boundary states with an optional local
/// window; every position may attend to itself.
struct BoundaryAttentionBlock {
    norm: LayerNorm,
    qkv_projection: UnifiedLinear,
    output_projection: UnifiedLinear,
    heads: i32,
    window: i32,
}

impl BoundaryAttentionBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        settings: &BoundaryHeadSettings,
        q: (i32, i32),
    ) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(weights, &format!("{prefix}.norm"))?,
            qkv_projection: linear(weights, &format!("{prefix}.qkv_projection"), q.0, q.1)?,
            output_projection: linear(weights, &format!("{prefix}.output_projection"), q.0, q.1)?,
            heads: settings.boundary_attention_heads as i32,
            window: settings.boundary_attention_window as i32,
        })
    }

    /// `[1, 1, L, L]` additive mask: 0 inside the window (or everywhere when
    /// there is none) and on the diagonal, `MASK_LOGIT` elsewhere.
    fn additive_mask(&self, length: i32, like: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        if self.window <= 0 {
            return None;
        }
        let positions = mlxcel_core::arange_i32(0, length, 1);
        let rows = mlxcel_core::reshape(&positions, &[length, 1]);
        let cols = mlxcel_core::reshape(&positions, &[1, length]);
        let distance = mlxcel_core::abs(&mlxcel_core::subtract(&rows, &cols));
        let allowed = mlxcel_core::less_equal(
            &distance,
            &mlxcel_core::from_slice_i32(&[self.window], &[1]),
        );
        let additive =
            mlxcel_core::where_cond(&allowed, &scalar(0.0, like), &scalar(MASK_LOGIT, like));
        Some(mlxcel_core::reshape(&additive, &[1, 1, length, length]))
    }

    fn forward(&self, states: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(states);
        let (length, dim) = (shape[0], shape[1]);
        let head_dim = dim / self.heads;
        let qkv = self.qkv_projection.forward(&self.norm.forward(states));
        let qkv = mlxcel_core::reshape(&qkv, &[length, 3, self.heads, head_dim]);
        // `[1, heads, L, head_dim]` each.
        let part = |index: i32| {
            let taken = mlxcel_core::slice(
                &qkv,
                &[0, index, 0, 0],
                &[length, index + 1, self.heads, head_dim],
            );
            mlxcel_core::transpose_axes(&taken, &[1, 2, 0, 3])
        };
        let (q, k, v) = (part(0), part(1), part(2));
        let mut scores = mlxcel_core::matmul(&q, &mlxcel_core::swap_axes(&k, -1, -2));
        scores = mlxcel_core::multiply(&scores, &scalar(1.0 / (head_dim as f32).sqrt(), &scores));
        if let Some(mask) = self.additive_mask(length, &scores) {
            scores = mlxcel_core::add(&scores, &mask);
        }
        let attended = mlxcel_core::matmul(&mlxcel_core::softmax(&scores, -1), &v);
        let attended = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]),
            &[length, dim],
        );
        mlxcel_core::add(states, &self.output_projection.forward(&attended))
    }
}

/// Boundary states: one per gap between words, including the two ends.
struct BoundaryEncoder {
    left_projection: UnifiedLinear,
    right_projection: UnifiedLinear,
    output_projection: UnifiedLinear,
    layer_norm: LayerNorm,
    attention_blocks: Vec<BoundaryAttentionBlock>,
    refinement_blocks: Vec<ResidualSwiGlu>,
    bos_state: UniquePtr<MlxArray>,
    eos_state: UniquePtr<MlxArray>,
}

impl BoundaryEncoder {
    fn from_weights(
        weights: &WeightMap,
        settings: &BoundaryHeadSettings,
        hidden_size: usize,
        q: (i32, i32),
    ) -> Result<Self> {
        let p = "boundary_head.boundary_encoder";
        let state = |name: &str| -> Result<UniquePtr<MlxArray>> {
            let key = format!("{p}.{name}");
            let array = tensor(weights, &key)?;
            let shape = mlxcel_core::array_shape(&array);
            if shape != [hidden_size as i32] {
                bail!(
                    "gliner2_5: {key} is {shape:?}, expected [{hidden_size}] (the encoder hidden size)"
                );
            }
            Ok(array)
        };
        Ok(Self {
            left_projection: linear(weights, &format!("{p}.left_projection"), q.0, q.1)?,
            right_projection: linear(weights, &format!("{p}.right_projection"), q.0, q.1)?,
            output_projection: linear(weights, &format!("{p}.output_projection"), q.0, q.1)?,
            layer_norm: layer_norm(weights, &format!("{p}.layer_norm"))?,
            attention_blocks: (0..settings.boundary_attention_layers)
                .map(|i| {
                    BoundaryAttentionBlock::from_weights(
                        weights,
                        &format!("{p}.attention_blocks.{i}"),
                        settings,
                        q,
                    )
                })
                .collect::<Result<_>>()?,
            refinement_blocks: (0..settings.boundary_refinement_layers)
                .map(|i| {
                    ResidualSwiGlu::from_weights(
                        weights,
                        &format!("{p}.refinement_blocks.{i}"),
                        settings.boundary_dim,
                        settings.boundary_ffn_multiplier,
                        q,
                    )
                })
                .collect::<Result<_>>()?,
            bos_state: state("bos_state")?,
            eos_state: state("eos_state")?,
        })
    }

    /// `[T, H]` word states to `[T + 1, dim]` boundary states.
    fn forward(&self, text_states: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = mlxcel_core::array_shape(text_states)[1];
        let bos = mlxcel_core::reshape(&self.bos_state, &[1, hidden]);
        let eos = mlxcel_core::reshape(&self.eos_state, &[1, hidden]);
        let left = mlxcel_core::utils::concatenate(&bos, text_states, 0);
        let right = mlxcel_core::utils::concatenate(text_states, &eos, 0);
        let joined = mlxcel_core::utils::concatenate(
            &self.left_projection.forward(&left),
            &self.right_projection.forward(&right),
            -1,
        );
        let mut states = self
            .layer_norm
            .forward(&self.output_projection.forward(&joined));
        for block in &self.attention_blocks {
            states = block.forward(&states);
        }
        for block in &self.refinement_blocks {
            states = block.forward(&states);
        }
        states
    }
}

/// Per-label start / end logits over the boundaries and the centered inside
/// evidence over the words.
pub(crate) struct Marginals {
    /// `[Q, T + 1]`
    pub start_logits: UniquePtr<MlxArray>,
    /// `[Q, T + 1]`
    pub end_logits: UniquePtr<MlxArray>,
    /// `[Q, T + 1]` prefix sums of the centered inside scores (f32).
    pub inside_prefix: UniquePtr<MlxArray>,
    /// `[Q, 1]` mean inside score per label (f32).
    pub inside_mean: UniquePtr<MlxArray>,
}

struct BoundaryQueryHead {
    start_boundary_projection: UnifiedLinear,
    start_query_projection: UnifiedLinear,
    end_boundary_projection: UnifiedLinear,
    end_query_projection: UnifiedLinear,
    inside_text_projection: UnifiedLinear,
    inside_query_projection: UnifiedLinear,
    scale: f32,
}

impl BoundaryQueryHead {
    fn from_weights(
        weights: &WeightMap,
        settings: &BoundaryHeadSettings,
        q: (i32, i32),
    ) -> Result<Self> {
        let p = "boundary_head.boundary_query_head";
        let load = |name: &str| linear(weights, &format!("{p}.{name}"), q.0, q.1);
        Ok(Self {
            start_boundary_projection: load("start_boundary_projection")?,
            start_query_projection: load("start_query_projection")?,
            end_boundary_projection: load("end_boundary_projection")?,
            end_query_projection: load("end_query_projection")?,
            inside_text_projection: load("inside_text_projection")?,
            inside_query_projection: load("inside_query_projection")?,
            scale: 1.0 / (settings.boundary_dim as f32).sqrt(),
        })
    }

    fn scored(&self, queries: &MlxArray, keys: &MlxArray) -> UniquePtr<MlxArray> {
        let product = mlxcel_core::matmul(queries, &mlxcel_core::swap_axes(keys, -1, -2));
        mlxcel_core::multiply(&product, &scalar(self.scale, &product))
    }

    fn forward(
        &self,
        boundaries: &MlxArray,
        text_states: &MlxArray,
        queries: &MlxArray,
    ) -> Marginals {
        let start_logits = self.scored(
            &self.start_query_projection.forward(queries),
            &self.start_boundary_projection.forward(boundaries),
        );
        let end_logits = self.scored(
            &self.end_query_projection.forward(queries),
            &self.end_boundary_projection.forward(boundaries),
        );
        let inside = self.scored(
            &self.inside_query_projection.forward(queries),
            &self.inside_text_projection.forward(text_states),
        );
        let inside = mlxcel_core::astype(&inside, dtype::FLOAT32);
        let shape = mlxcel_core::array_shape(&inside);
        let (labels, words) = (shape[0], shape[1]);
        let count = mlxcel_core::full_f32(&[1], words.max(1) as f32, dtype::FLOAT32);
        let inside_mean = mlxcel_core::divide(&mlxcel_core::sum_axis(&inside, -1, true), &count);
        let centered = mlxcel_core::subtract(&inside, &inside_mean);
        let zeros = mlxcel_core::zeros(&[labels, 1], dtype::FLOAT32);
        let inside_prefix = mlxcel_core::utils::concatenate(
            &zeros,
            &mlxcel_core::cumsum(&centered, -1, false, true),
            -1,
        );
        Marginals {
            start_logits,
            end_logits,
            inside_prefix,
            inside_mean,
        }
    }
}

/// Outputs of one head pass.
pub(crate) struct HeadOutput {
    pub pooled: PooledCandidates,
    /// `[P, Q]` span logits.
    pub logits: UniquePtr<MlxArray>,
    /// `[Q]` abstention logits.
    pub null_logits: UniquePtr<MlxArray>,
}

/// The whole head: boundary encoder, query head, candidate pool and scorer.
pub(crate) struct BoundaryHead {
    boundary_encoder: BoundaryEncoder,
    boundary_query_head: BoundaryQueryHead,
    shared_pool_builder: DocumentCandidatePool,
    shared_pool_scorer: SharedPoolScorer,
    null_projection: UnifiedLinear,
}

impl BoundaryHead {
    /// `hidden_size` is the encoder width the head's query and text
    /// projections read.
    pub fn from_weights(
        weights: &WeightMap,
        settings: &BoundaryHeadSettings,
        hidden_size: usize,
        q: (i32, i32),
    ) -> Result<Self> {
        Ok(Self {
            boundary_encoder: BoundaryEncoder::from_weights(weights, settings, hidden_size, q)?,
            boundary_query_head: BoundaryQueryHead::from_weights(weights, settings, q)?,
            shared_pool_builder: DocumentCandidatePool::from_weights(weights, settings, q)?,
            shared_pool_scorer: SharedPoolScorer::from_weights(weights, settings, q)?,
            null_projection: linear(weights, "boundary_head.null_projection", q.0, q.1)?,
        })
    }

    /// Score every pooled span of `text_states` (`[T, H]`) against every
    /// label query (`[Q, H]`).
    pub fn forward(&self, text_states: &MlxArray, query_states: &MlxArray) -> Result<HeadOutput> {
        let boundaries = self.boundary_encoder.forward(text_states);
        let marginals = self
            .boundary_query_head
            .forward(&boundaries, text_states, query_states);
        let pooled = self.shared_pool_builder.build(
            &boundaries,
            &marginals.start_logits,
            &marginals.end_logits,
        )?;
        let logits = self.shared_pool_scorer.forward(
            &boundaries,
            query_states,
            &pooled,
            &marginals,
            text_states,
        );
        let null_logits =
            mlxcel_core::squeeze_axis(&self.null_projection.forward(query_states), -1);
        Ok(HeadOutput {
            pooled,
            logits,
            null_logits,
        })
    }
}
