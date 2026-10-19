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

//! Shared pieces of the GLiNER2.5 head: the `boundary_head` settings and
//! the small weight-loading and array helpers every head module uses.

use anyhow::{Result, bail};
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Score of a masked position before the softmax, as upstream `MASK_LOGIT`.
pub(crate) const MASK_LOGIT: f32 = -1e4;

/// `nn.LayerNorm` default epsilon in the head (the encoder has its own).
const HEAD_LAYER_NORM_EPS: f32 = 1e-5;

/// Ceiling on `pool_boundary_top_k`: the pool scores every pair of the top
/// starts and ends, so the grid is its square (upstream ships 32).
pub const MAX_POOL_BOUNDARY_TOP_K: usize = 1024;

/// The `boundary_head` settings this port reads, with upstream's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryHeadSettings {
    pub boundary_dim: usize,
    pub content_dim: usize,
    pub pair_dim: usize,
    pub boundary_attention_heads: usize,
    pub boundary_attention_layers: usize,
    pub boundary_attention_window: usize,
    pub boundary_ffn_multiplier: f64,
    pub boundary_refinement_layers: usize,
    pub pool_boundary_top_k: usize,
    pub pool_size: usize,
    pub min_pool_per_query: usize,
}

impl BoundaryHeadSettings {
    /// Read the settings object, refusing the head variants upstream refuses.
    pub fn from_config(settings: &serde_json::Value) -> Result<Self> {
        // Every size becomes an `i32` array dimension, so it must fit one.
        let int = |key: &str, default: u64| -> Result<usize> {
            match settings.get(key) {
                None | Some(serde_json::Value::Null) => Ok(default as usize),
                Some(value) => value
                    .as_u64()
                    .and_then(|v| i32::try_from(v).ok())
                    .map(|v| v as usize)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "boundary_head.{key} must be an integer in 0..={}, got {value}",
                            i32::MAX
                        )
                    }),
            }
        };
        let pool = settings
            .get("candidate_pool")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("shared");
        if pool != "shared" {
            bail!(
                "GLiNER2.5 MLX currently supports shared candidate pools (candidate_pool is '{pool}')"
            );
        }
        if int("candidate_attention_layers", 0)? != 0 {
            bail!("candidate attention layers are not supported yet");
        }
        if int("query_attention_layers", 0)? != 0 {
            bail!("query attention layers are not supported yet");
        }
        let boundary_dim = int("boundary_dim", 128)?;
        let heads = int("boundary_attention_heads", 4)?;
        if boundary_dim == 0 || heads == 0 || !boundary_dim.is_multiple_of(heads) {
            bail!(
                "boundary_head.boundary_dim {boundary_dim} is not divisible by boundary_attention_heads {heads}"
            );
        }
        let pool_size = match settings.get("pool_size") {
            Some(value) if !value.is_null() => int("pool_size", 192)?,
            _ => int("candidate_budget", 192)?,
        };
        let pool_boundary_top_k = int("pool_boundary_top_k", 32)?;
        if pool_boundary_top_k > MAX_POOL_BOUNDARY_TOP_K {
            bail!(
                "boundary_head.pool_boundary_top_k {pool_boundary_top_k} exceeds {MAX_POOL_BOUNDARY_TOP_K}; the candidate grid is its square"
            );
        }
        let boundary_ffn_multiplier = settings
            .get("boundary_ffn_multiplier")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(2.0);
        if !boundary_ffn_multiplier.is_finite() || boundary_ffn_multiplier <= 0.0 {
            bail!(
                "boundary_head.boundary_ffn_multiplier must be a positive number, got {boundary_ffn_multiplier}"
            );
        }
        Ok(Self {
            boundary_dim,
            content_dim: int("content_dim", 64)?,
            pair_dim: int("pair_dim", 128)?,
            boundary_attention_heads: heads,
            boundary_attention_layers: int("boundary_attention_layers", 0)?,
            boundary_attention_window: int("boundary_attention_window", 0)?,
            boundary_ffn_multiplier,
            boundary_refinement_layers: int("boundary_refinement_layers", 1)?,
            pool_boundary_top_k,
            pool_size,
            min_pool_per_query: int("min_pool_per_query", 8)?,
        })
    }
}

pub(crate) fn linear(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<UnifiedLinear> {
    UnifiedLinear::from_weights(weights, prefix, group_size, bits)
        .map_err(|e| anyhow::anyhow!("gliner2_5: {e}"))
}

pub(crate) fn layer_norm(weights: &WeightMap, prefix: &str) -> Result<LayerNorm> {
    let weight_key = format!("{prefix}.weight");
    let Some(weight) = weights.get(&weight_key).map(|w| mlxcel_core::copy(w)) else {
        bail!("gliner2_5 weight not found: {weight_key}");
    };
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, HEAD_LAYER_NORM_EPS))
}

pub(crate) fn tensor(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("gliner2_5 weight not found: {key}"))
}

pub(crate) fn scalar(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(like))
}

/// `x[..., start..end]` along the last axis; a rank-0 array is returned as is.
pub(crate) fn slice_last(x: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let Some(last) = shape.len().checked_sub(1) else {
        return mlxcel_core::copy(x);
    };
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[last] = start;
    stops[last] = end;
    mlxcel_core::slice(x, &starts, &stops)
}
