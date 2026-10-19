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

//! Config resolution and relative-position bucketing for the DeBERTa-v2
//! encoder.
//!
//! Split out of [`super::deberta_v2`] so the encoder module stays inside the
//! 500-line file guideline; both items are re-exported there, so callers keep
//! using `crate::models::deberta_v2::{DebertaV2Args, relative_position_buckets}`.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
struct RawDebertaV2Args {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_minus_one")]
    max_relative_positions: i64,
    #[serde(default = "default_minus_one")]
    position_buckets: i64,
    #[serde(default = "default_relative_attention")]
    relative_attention: bool,
    #[serde(default)]
    pos_att_type: PosAttType,
    /// Hugging Face's default is `false`; a config that omits the key uses
    /// separate position projections, which are not ported.
    #[serde(default)]
    share_att_key: bool,
    #[serde(default = "default_true")]
    position_biased_input: bool,
    #[serde(default)]
    type_vocab_size: usize,
    #[serde(default = "default_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_gelu")]
    hidden_act: String,
    #[serde(default = "default_norm_rel_ebd")]
    norm_rel_ebd: String,
    #[serde(default)]
    conv_kernel_size: usize,
    #[serde(default)]
    embedding_size: Option<usize>,
}

/// `pos_att_type` as a list (`["p2c", "c2p"]`) or Hugging Face's legacy
/// `"p2c|c2p"` string.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(untagged)]
enum PosAttType {
    #[default]
    Unset,
    List(Vec<String>),
    Joined(String),
}

impl PosAttType {
    fn terms(&self) -> Vec<String> {
        match self {
            Self::Unset => Vec::new(),
            Self::List(terms) => terms
                .iter()
                .map(|t| t.trim().to_ascii_lowercase())
                .collect(),
            Self::Joined(joined) => joined
                .split('|')
                .map(|t| t.trim().to_ascii_lowercase())
                .filter(|t| !t.is_empty())
                .collect(),
        }
    }
}

fn default_minus_one() -> i64 {
    -1
}
fn default_relative_attention() -> bool {
    false
}
fn default_true() -> bool {
    true
}
fn default_eps() -> f32 {
    1e-7
}
fn default_gelu() -> String {
    "gelu".to_string()
}
fn default_norm_rel_ebd() -> String {
    "none".to_string()
}

/// Resolved encoder geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct DebertaV2Args {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    /// Largest distance the log buckets reach; `max_position_embeddings` when
    /// the config sets it below 1, as in Hugging Face.
    pub max_relative_positions: usize,
    /// Rows of the relative table are `2 * position_buckets`.
    pub position_buckets: usize,
    /// Whether the `c2p` term is on.
    pub c2p: bool,
    /// Whether the `p2c` term is on.
    pub p2c: bool,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    /// Whether the relative table is LayerNormed before use
    /// (`norm_rel_ebd: layer_norm`).
    pub norm_rel_ebd: bool,
}

impl DebertaV2Args {
    /// Read the encoder geometry, refusing the DeBERTa options this port does
    /// not implement rather than running them wrong.
    pub fn from_config(config: &Value) -> Result<Self> {
        let raw: RawDebertaV2Args =
            serde_json::from_value(config.clone()).context("deberta-v2 config")?;
        if raw.hidden_size == 0 || raw.num_attention_heads == 0 {
            bail!("deberta-v2 config: hidden_size and num_attention_heads must be non-zero");
        }
        if !raw.hidden_size.is_multiple_of(raw.num_attention_heads) {
            bail!(
                "deberta-v2 config: hidden_size {} is not divisible by num_attention_heads {}",
                raw.hidden_size,
                raw.num_attention_heads
            );
        }
        if !raw.relative_attention {
            bail!(
                "deberta-v2 config: relative_attention must be true (absolute positions are not ported)"
            );
        }
        if raw.position_biased_input {
            bail!(
                "deberta-v2 config: position_biased_input must be false (absolute positions are not ported)"
            );
        }
        if raw.type_vocab_size != 0 {
            bail!("deberta-v2 config: type_vocab_size must be 0 (token types are not ported)");
        }
        if !raw.share_att_key {
            bail!(
                "deberta-v2 config: share_att_key must be true (separate position projections are not ported)"
            );
        }
        if raw.conv_kernel_size != 0 {
            bail!(
                "deberta-v2 config: conv_kernel_size must be 0 (the convolution layer is not ported)"
            );
        }
        if raw
            .embedding_size
            .is_some_and(|size| size != raw.hidden_size)
        {
            bail!(
                "deberta-v2 config: embedding_size must equal hidden_size (the embedding projection is not ported)"
            );
        }
        if raw.position_buckets < 1 {
            bail!(
                "deberta-v2 config: position_buckets must be positive (unbucketed relative positions are not ported)"
            );
        }
        let norm_rel_ebd = match raw.norm_rel_ebd.trim().to_ascii_lowercase().as_str() {
            "layer_norm" => true,
            "none" | "" => false,
            other => bail!("deberta-v2 config: norm_rel_ebd `{other}` is not supported"),
        };
        let pos_att_type = raw.pos_att_type.terms();
        for term in &pos_att_type {
            if term != "c2p" && term != "p2c" {
                bail!(
                    "deberta-v2 config: pos_att_type `{term}` is not supported (expected c2p / p2c)"
                );
            }
        }
        let max_relative_positions = if raw.max_relative_positions < 1 {
            raw.max_position_embeddings
        } else {
            raw.max_relative_positions as usize
        };
        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            intermediate_size: raw.intermediate_size,
            max_position_embeddings: raw.max_position_embeddings,
            max_relative_positions,
            position_buckets: raw.position_buckets as usize,
            c2p: pos_att_type.iter().any(|t| t == "c2p"),
            p2c: pos_att_type.iter().any(|t| t == "p2c"),
            layer_norm_eps: raw.layer_norm_eps,
            hidden_act: raw.hidden_act,
            norm_rel_ebd,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `sqrt(head_dim * (1 + number of position terms))`, the shared divisor
    /// of the content and position scores.
    pub fn attention_scale(&self) -> f32 {
        let terms = 1 + usize::from(self.c2p) + usize::from(self.p2c);
        ((self.head_dim() * terms) as f32).sqrt()
    }
}

/// Hugging Face `make_log_bucket_position` for the distances
/// `-(length - 1) ..= length - 1`, in that order.
///
/// Distances within half the bucket count map to themselves; larger ones are
/// log-spaced up to `max_position`, so the result lies in
/// `(-bucket_size / 2 - ..., bucket_size / 2 + ...)` and indexes a table of
/// `2 * bucket_size` rows after `+ bucket_size`. The arithmetic is done in
/// f32, as torch does it, so a `ceil` at an exact boundary lands on the same
/// side.
pub fn relative_position_buckets(
    length: usize,
    bucket_size: usize,
    max_position: usize,
) -> Vec<i32> {
    let mid = (bucket_size / 2) as i64;
    let log_base = ((max_position as f32 - 1.0) / mid as f32).ln();
    let span = length as i64 - 1;
    (-span..=span)
        .map(|relative| {
            let abs_pos = if relative < mid && relative > -mid {
                mid - 1
            } else {
                relative.abs()
            };
            if abs_pos <= mid {
                relative as i32
            } else {
                let log_pos = ((abs_pos as f32 / mid as f32).ln() / log_base * (mid - 1) as f32)
                    .ceil()
                    + mid as f32;
                (log_pos * relative.signum() as f32) as i32
            }
        })
        .collect()
}
