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

//! GLiNER2.5 unit tests that need no weights: detection, config handling,
//! schema tokens, word splitting, overlap resolution and the generation
//! refusal.

use serde_json::{Value, json};

use super::*;
use crate::models::get_model_type;
use crate::token_classification::gliner2_5_layers::BoundaryHeadSettings;
use crate::token_classification::gliner2_5_prepare::{
    WordSplitter, resolve_flat_overlaps, schema_tokens, split_words,
};
use crate::token_classification::loader::{
    TokenModelKind, TokenOutputShape, detect_token_model_kind, load_span_extractor,
    load_token_classifier,
};

/// The live `fastino/gliner2.5-small-v1` `config.json`, as the mlx-vlm
/// fixture generator read it.
pub(crate) fn live_config() -> Value {
    json!({
        "architecture": "boundary",
        "architecture_version": 1,
        "architectures": ["Gliner2ForSchemaExtraction"],
        "boundary_head": {
            "boundary_attention_heads": 4,
            "boundary_attention_layers": 2,
            "boundary_attention_window": 128,
            "boundary_dim": 128,
            "boundary_ffn_multiplier": 2.0,
            "boundary_refinement_layers": 1,
            "candidate_attention_layers": 0,
            "candidate_budget": 192,
            "candidate_pool": "shared",
            "content_dim": 64,
            "end_top_k": 24,
            "ends_per_start": 12,
            "min_pool_per_query": 8,
            "pair_dim": 128,
            "pool_boundary_top_k": 32,
            "pool_size": 192,
            "query_attention_layers": 0,
            "start_top_k": 24,
            "starts_per_end": 12,
        },
        "max_len": 4096,
        "model_type": "gliner2",
        "token_pooling": "first",
        "encoder_config": {
            "model_type": "deberta-v2",
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "vocab_size": 128011,
            "max_position_embeddings": 512,
            "max_relative_positions": -1,
            "position_buckets": 256,
            "pos_att_type": ["p2c", "c2p"],
            "share_att_key": true,
            "relative_attention": true,
            "norm_rel_ebd": "layer_norm",
            "position_biased_input": false,
            "type_vocab_size": 0,
            "layer_norm_eps": 1e-7,
        },
    })
}

fn checkpoint_dir(config: &Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("config written");
    dir
}

#[test]
fn live_config_detects_as_gliner2_5() {
    let kind = detect_token_model_kind(&live_config()).unwrap();
    assert_eq!(kind, TokenModelKind::Gliner2_5);
    assert_eq!(kind.output_shape(), TokenOutputShape::SpanExtraction);
    assert_eq!(kind.as_str(), "gliner2_5");
    for model_type in ["gliner2_5", "GLiNER2"] {
        let mut config = live_config();
        config["model_type"] = json!(model_type);
        assert_eq!(
            detect_token_model_kind(&config).unwrap(),
            TokenModelKind::Gliner2_5
        );
    }
}

#[test]
fn boundary_extractor_export_detects_and_reads_the_sidecar_encoder_config() {
    let mut config = live_config();
    config["model_type"] = json!("deberta-v2");
    config["architectures"] = json!(["BoundaryExtractor"]);
    let encoder_config = config
        .as_object_mut()
        .unwrap()
        .remove("encoder_config")
        .unwrap();
    config.as_object_mut().unwrap().remove("architecture");
    assert_eq!(
        detect_token_model_kind(&config).unwrap(),
        TokenModelKind::Gliner2_5
    );

    let dir = checkpoint_dir(&config);
    let err = resolve_encoder_config(dir.path(), &config).err().unwrap();
    assert!(
        err.to_string().contains("encoder_config") && err.to_string().contains("config.json"),
        "{err}"
    );
    std::fs::create_dir(dir.path().join("encoder_config")).unwrap();
    std::fs::write(
        dir.path().join("encoder_config").join("config.json"),
        encoder_config.to_string(),
    )
    .unwrap();
    assert_eq!(
        resolve_encoder_config(dir.path(), &config).unwrap(),
        encoder_config
    );
    assert_eq!(
        resolve_encoder_config(dir.path(), &live_config()).unwrap(),
        live_config()["encoder_config"],
        "an inline block wins over the sidecar"
    );
}

#[test]
fn non_boundary_architectures_are_refused() {
    let mut config = live_config();
    config["architecture"] = json!("span");
    let err = detect_token_model_kind(&config).err().unwrap();
    assert_eq!(
        err.to_string(),
        "GLiNER2 architecture 'span' is not supported; only the boundary extractor is"
    );
    let mut absent = live_config();
    absent.as_object_mut().unwrap().remove("architecture");
    let err = detect_token_model_kind(&absent).err().unwrap();
    assert!(err.to_string().contains("'<missing>'"), "{err}");
}

#[test]
fn unsupported_head_settings_are_refused() {
    for (key, value, needle) in [
        ("pool_boundary_top_k", json!(4096), "exceeds 1024"),
        ("pool_size", json!(1u64 << 40), "in 0..="),
        ("boundary_dim", json!(-1), "in 0..="),
        ("boundary_ffn_multiplier", json!(0.0), "positive number"),
        ("boundary_attention_heads", json!(3), "not divisible"),
        (
            "candidate_pool",
            json!("per_query"),
            "shared candidate pools",
        ),
        (
            "candidate_attention_layers",
            json!(1),
            "candidate attention layers",
        ),
        ("query_attention_layers", json!(2), "query attention layers"),
    ] {
        let mut config = live_config();
        config["boundary_head"][key] = value;
        let err = BoundaryHeadSettings::from_config(&config["boundary_head"])
            .err()
            .unwrap();
        assert!(err.to_string().contains(needle), "{key}: {err}");
    }
    let settings = BoundaryHeadSettings::from_config(&live_config()["boundary_head"]).unwrap();
    assert_eq!(settings.boundary_dim, 128);
    assert_eq!(settings.pool_size, 192);
    assert_eq!(settings.boundary_attention_window, 128);
    let defaults = BoundaryHeadSettings::from_config(&json!({})).unwrap();
    assert_eq!(defaults.pool_size, 192);
    assert_eq!(defaults.boundary_attention_layers, 0);
}

#[test]
fn sanitize_splits_encoder_from_head_and_drops_unsupported_tensors() {
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for key in [
        "encoder.embeddings.word_embeddings.weight",
        "encoder.encoder.layer.0.attention.self.query_proj.weight",
        "boundary_head.null_projection.weight",
        "classifier.0.weight",
        "boundary_head.boundary_proposer.end_key_projection.weight",
        "boundary_head.pair_scorer.compat_mix.weight",
        "boundary_head.candidate_encoder.weight",
        "record_decoder.cand_proj.weight",
        "relation_scorer.mlp.0.weight",
    ] {
        weights.insert(
            key.to_string(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        );
    }
    let (encoder, head) = sanitize(weights);
    let mut encoder_keys: Vec<_> = encoder.keys().cloned().collect();
    encoder_keys.sort();
    assert_eq!(
        encoder_keys,
        [
            "embeddings.word_embeddings.weight",
            "encoder.layer.0.attention.self.query_proj.weight"
        ]
    );
    let mut head_keys: Vec<_> = head.keys().cloned().collect();
    head_keys.sort();
    assert_eq!(
        head_keys,
        [
            "boundary_head.null_projection.weight",
            "classifier.0.weight"
        ]
    );
}

#[test]
fn sanitize_maps_an_mlx_vlm_export_back_to_hugging_face_names() {
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for key in [
        "encoder.embeddings.layer_norm.weight",
        "encoder.encoder.layer_norm.bias",
        "encoder.encoder.layers.3.attention.self_attn.key_proj.scales",
        "encoder.encoder.layers.3.attention.output.layer_norm.weight",
        "encoder.encoder.layers.3.output.layer_norm.bias",
        "boundary_head.boundary_encoder.layer_norm.weight",
        "boundary_head.shared_pool_scorer.content_pooler.layer_norm.bias",
    ] {
        weights.insert(
            key.to_string(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        );
    }
    let (encoder, head) = sanitize(weights);
    let mut encoder_keys: Vec<_> = encoder.keys().cloned().collect();
    encoder_keys.sort();
    assert_eq!(
        encoder_keys,
        [
            "embeddings.LayerNorm.weight",
            "encoder.LayerNorm.bias",
            "encoder.layer.3.attention.output.LayerNorm.weight",
            "encoder.layer.3.attention.self.key_proj.scales",
            "encoder.layer.3.output.LayerNorm.bias",
        ]
    );
    let mut head_keys: Vec<_> = head.keys().cloned().collect();
    head_keys.sort();
    assert_eq!(
        head_keys,
        [
            "boundary_head.boundary_encoder.layer_norm.weight",
            "boundary_head.shared_pool_scorer.content_pooler.layer_norm.bias",
        ],
        "the head keeps its own layer_norm names"
    );
}

#[test]
fn schema_tokens_match_upstream() {
    let labels = ["company", "person"].map(str::to_string);
    assert_eq!(
        schema_tokens("entities", &labels, "[E]", None, |_| None),
        [
            "(", "[P]", "entities", "(", "[E]", "company", "[E]", "person", ")", ")"
        ]
    );
    assert_eq!(
        schema_tokens("entities", &labels, "[E]", Some(""), |_| None)[2],
        "entities",
        "an empty prompt falls back to the parent, as upstream's `if prompt`"
    );
    let described = schema_tokens("sentiment", &labels, "[L]", Some("the tone"), |label| {
        (label == "person").then_some("a human")
    });
    assert_eq!(
        described,
        [
            "(",
            "[P]",
            "sentiment: the tone [DESCRIPTION] person: a human",
            "(",
            "[L]",
            "company",
            "[L]",
            "person",
            ")",
            ")"
        ]
    );
}

#[test]
fn word_splitting_keeps_urls_emails_and_handles_whole() {
    let words = split_words(
        "Contact ada.lovelace@example.com or visit https://example.com/team today",
        WordSplitter::Whitespace,
    );
    let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(
        texts,
        [
            "contact",
            "ada.lovelace@example.com",
            "or",
            "visit",
            "https://example.com/team",
            "today"
        ]
    );
    assert_eq!((words[1].start, words[1].end), (8, 32));
    assert_eq!((words[4].start, words[4].end), (42, 66));

    let joined = split_words("New-York, @Sam!", WordSplitter::Whitespace);
    let texts: Vec<&str> = joined.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["new-york", ",", "@sam", "!"]);

    // Offsets are scalar indices even after multi-byte characters.
    let cjk = split_words("东京大学的田中教授访问了北京。", WordSplitter::Char);
    assert_eq!(cjk.len(), 15);
    assert_eq!(
        (cjk[5].text.as_str(), cjk[5].start, cjk[5].end),
        ("田", 5, 6)
    );
    let whitespace_cjk = split_words("东京大学 Tokyo", WordSplitter::Whitespace);
    let texts: Vec<&str> = whitespace_cjk.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["东京大学", "tokyo"]);
}

/// Python's `\w` is `isalnum() + _`: combining marks and joiners are not
/// word characters (each becomes its own `\S` word) and `No` digits are. The
/// expected values are Python 3 `re.finditer` outputs on upstream's pattern.
#[test]
fn word_splitting_matches_python_word_characters() {
    let words = |text: &str| -> Vec<(String, usize, usize)> {
        split_words(text, WordSplitter::Whitespace)
            .into_iter()
            .map(|w| (w.text, w.start, w.end))
            .collect()
    };
    let devanagari = words("हिन्दी भाषा");
    assert_eq!(devanagari.len(), 10, "{devanagari:?}");
    assert_eq!(devanagari[0], ("ह".to_string(), 0, 1));
    assert_eq!(devanagari[1], ("ि".to_string(), 1, 2));
    assert_eq!(devanagari[6], ("भ".to_string(), 7, 8));
    let persian = words("می\u{200c}خواهم");
    assert_eq!(
        persian,
        [
            ("می".to_string(), 0, 2),
            ("\u{200c}".to_string(), 2, 3),
            ("خواهم".to_string(), 3, 8)
        ]
    );
    assert_eq!(words("x²"), [("x²".to_string(), 0, 2)]);
    assert_eq!(
        words("e\u{301}"),
        [("e".to_string(), 0, 1), ("\u{301}".to_string(), 1, 2)]
    );
    assert_eq!(words("café"), [("café".to_string(), 0, 4)]);
    // The bounded e-mail branch no longer rescans a long dotted run on every
    // word; the quadratic form took seconds here.
    let long_run = "a.".repeat(5000);
    let started = std::time::Instant::now();
    let run = split_words(&long_run, WordSplitter::Whitespace);
    assert_eq!(run.len(), 10000);
    assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
}

#[test]
fn flat_overlaps_keep_the_heaviest_disjoint_set() {
    // (score, start, end): the two short spans outweigh the long one that
    // covers them, and a tie prefers the span seen later in (end, start) order.
    let spans = [
        (0.9, 0, 3),
        (0.6, 0, 1),
        (0.5, 1, 3),
        (0.95, 3, 4),
        (0.4, 2, 4),
    ];
    let kept = resolve_flat_overlaps(&spans);
    assert_eq!(kept, [(0.95, 3, 4), (0.6, 0, 1), (0.5, 1, 3)]);
    assert!(resolve_flat_overlaps(&[]).is_empty());

    // Equal totals: `>` keeps the skipped branch, as upstream.
    let tie = [(0.5, 0, 2), (0.25, 0, 1), (0.25, 1, 2)];
    assert_eq!(resolve_flat_overlaps(&tie), [(0.5, 0, 2)]);
}

#[test]
fn generation_refuses_the_checkpoint_and_tag_names_extract() {
    let dir = checkpoint_dir(&live_config());
    let err = get_model_type(dir.path()).err().unwrap();
    assert!(
        err.to_string()
            .contains("GLiNER2.5 is a span extractor; use `mlxcel extract`"),
        "{err}"
    );
    let err = format!("{:#}", load_token_classifier(dir.path()).err().unwrap());
    assert!(
        err.contains("span-extraction checkpoint (gliner2_5)"),
        "{err}"
    );
    assert!(
        err.contains("use `mlxcel extract` instead of `mlxcel tag`"),
        "{err}"
    );
}

#[test]
fn extract_loader_refuses_an_out_of_range_max_len_before_reading_weights() {
    for value in [json!(0), json!(32769), json!(-1), json!("x")] {
        let mut config = live_config();
        config["max_len"] = value.clone();
        let dir = checkpoint_dir(&config);
        let err = format!("{:#}", load_span_extractor(dir.path()).err().unwrap());
        assert!(
            err.contains("max_len must be an integer in 1..=32768"),
            "max_len {value}: {err}"
        );
    }
    // The ceiling itself and an absent value pass the check and fail only
    // on the missing weights.
    for config in [
        {
            let mut config = live_config();
            config["max_len"] = json!(32768);
            config
        },
        {
            let mut config = live_config();
            config.as_object_mut().unwrap().remove("max_len");
            config
        },
    ] {
        let dir = checkpoint_dir(&config);
        let err = format!("{:#}", load_span_extractor(dir.path()).err().unwrap());
        assert!(!err.contains("max_len"), "{err}");
    }
}

#[test]
fn extract_loader_fails_on_a_missing_sidecar_before_reading_weights() {
    let mut config = live_config();
    config.as_object_mut().unwrap().remove("encoder_config");
    let dir = checkpoint_dir(&config);
    let err = format!("{:#}", load_span_extractor(dir.path()).err().unwrap());
    assert!(err.contains("GLiNER2.5 encoder config not found"), "{err}");
}
