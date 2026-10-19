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

//! Real-checkpoint gates for `mlxcel extract` on `fastino/gliner2.5-small-v1`.
//!
//! Soft-skip when the checkpoint is absent (`models/mlx/gliner2.5-small-v1`
//! through the repository's `models` link, or `mlxcel download
//! fastino/gliner2.5-small-v1`). The fixture
//! `tests/fixtures/token_classification/gliner2_5_mlx_vlm.json` was written
//! by `generate_gliner2_5_fixture.py` with the mlx-vlm revision its `oracle`
//! field records; every case is under 128 tokens, where mlx-vlm's relative
//! positions agree with Hugging Face's. Input ids, word and marker positions
//! must match exactly, spans must match exactly, and confidences within 1e-3.

use serde_json::{Value, json};

use super::*;
use crate::models::deberta_v2::deberta_v2_tests::gliner2_5_small_dir;
use crate::models::embedding_test_support::mlx_test_guard;
use crate::token_classification::loader::{TokenModelKind, load_span_extractor};

const CONFIDENCE_TOLERANCE: f64 = 1e-3;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/gliner2_5_mlx_vlm.json"
    ))
    .expect("fixture parses")
}

fn positions(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn labels_of(case: &Value) -> Vec<(String, Option<String>)> {
    let descriptions = case["descriptions"].as_object();
    case["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|label| {
            let label = label.as_str().unwrap().to_string();
            let description = descriptions
                .and_then(|d| d.get(&label))
                .and_then(Value::as_str)
                .map(str::to_string);
            (label, description)
        })
        .collect()
}

fn assert_prepared(prepared: &PreparedInput, expected: &Value, name: &str) {
    assert_eq!(
        prepared.input_ids,
        ids(&expected["input_ids"]),
        "{name}: input ids"
    );
    assert_eq!(
        prepared.word_positions,
        positions(&expected["word_positions"]),
        "{name}: word positions"
    );
    assert_eq!(
        prepared.marker_positions,
        positions(&expected["marker_positions"]),
        "{name}: marker positions"
    );
    let offsets: Vec<(usize, usize)> = expected["offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let pair = positions(pair);
            (pair[0], pair[1])
        })
        .collect();
    assert_eq!(prepared.offsets, offsets, "{name}: offsets");
    assert_eq!(
        prepared.text,
        expected["text"].as_str().unwrap(),
        "{name}: text"
    );
}

/// Compare `{"entities": ...}` objects: identical labels, texts and offsets,
/// confidences within tolerance.
fn assert_entities(got: &Value, want: &Value, name: &str) {
    let (got, want) = (&got["entities"], &want["entities"]);
    let got_map = got.as_object().unwrap();
    let want_map = want.as_object().unwrap();
    let got_labels: Vec<&String> = got_map.keys().collect();
    let want_labels: Vec<&String> = want_map.keys().collect();
    assert_eq!(got_labels, want_labels, "{name}: label order");
    for (label, want_items) in want_map {
        let got_items = got_map[label].as_array().unwrap();
        let want_items = want_items.as_array().unwrap();
        assert_eq!(
            got_items.len(),
            want_items.len(),
            "{name}/{label}: span count\n  mlxcel:  {got_items:?}\n  mlx-vlm: {want_items:?}"
        );
        for (got_item, want_item) in got_items.iter().zip(want_items) {
            match (got_item, want_item) {
                (Value::String(a), Value::String(b)) => assert_eq!(a, b, "{name}/{label}"),
                (Value::Object(a), Value::Object(b)) => {
                    for key in ["text", "start", "end"] {
                        assert_eq!(a.get(key), b.get(key), "{name}/{label}: {key}");
                    }
                    if let Some(want) = b.get("confidence").and_then(Value::as_f64) {
                        let got = a["confidence"].as_f64().expect("confidence present");
                        assert!(
                            (got - want).abs() < CONFIDENCE_TOLERANCE,
                            "{name}/{label}: confidence {got} vs mlx-vlm {want}"
                        );
                    }
                }
                other => panic!("{name}/{label}: item shapes differ: {other:?}"),
            }
        }
    }
}

#[test]
fn gliner2_5_small_matches_the_mlx_vlm_fixture() {
    let Some(dir) = gliner2_5_small_dir() else {
        eprintln!("gliner2.5-small-v1 not found; skipping the mlx-vlm parity gate");
        return;
    };
    let _guard = mlx_test_guard();
    let fixture = fixture();
    eprintln!("oracle: {}", fixture["oracle"]);
    let mut model = Gliner2_5Model::load_dir(&dir).expect("gliner2.5-small-v1 loads");
    assert_eq!(model.max_len(), 4096);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        match case["kind"].as_str().unwrap() {
            "extract" => {
                let labels = labels_of(case);
                let names: Vec<String> = labels.iter().map(|(l, _)| l.clone()).collect();
                let splitter = match case["word_splitter"].as_str().unwrap() {
                    "char" => WordSplitter::Char,
                    _ => WordSplitter::Whitespace,
                };
                model.set_word_splitter(splitter);
                let schema = schema_tokens("entities", &names, "[E]", None, |label| {
                    labels
                        .iter()
                        .find(|(l, _)| l == label)
                        .and_then(|(_, d)| d.as_deref())
                });
                let prepared = model.prepare(text, &schema).unwrap();
                assert_prepared(&prepared, &case["prepared"], name);
                let options = ExtractOptions {
                    include_confidence: case["options"]["include_confidence"]
                        .as_bool()
                        .unwrap_or(false),
                    include_spans: case["options"]["include_spans"].as_bool().unwrap_or(false),
                };
                let result = model.extract_entities(text, &labels, 0.5, options).unwrap();
                let got = serde_json::to_value(&result).unwrap();
                assert_entities(&got, &case["result"], name);
                eprintln!(
                    "parity ok: {name} ({} tokens): {got}",
                    prepared.input_ids.len()
                );
            }
            "classify" => {
                let tasks: Vec<(String, ClassifyTask)> = case["tasks"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(task, spec)| {
                        (task.clone(), serde_json::from_value(spec.clone()).unwrap())
                    })
                    .collect();
                model.set_word_splitter(WordSplitter::Whitespace);
                for (task, spec) in &tasks {
                    let schema =
                        schema_tokens(task, spec.labels(), "[L]", spec.prompt(), |label| {
                            spec.label_description(label)
                        });
                    let prepared = model.prepare(text, &schema).unwrap();
                    assert_prepared(
                        &prepared,
                        &case["prepared"][task],
                        &format!("{name}/{task}"),
                    );
                }
                let got = model.classify_text(text, &tasks, 0.5).unwrap();
                assert_eq!(got, case["result"], "{name}");
                eprintln!("parity ok: {name}: {got}");
            }
            other => panic!("unknown case kind {other}"),
        }
    }
}

#[test]
fn gliner2_5_small_runs_the_readme_examples_through_the_loader() {
    let Some(dir) = gliner2_5_small_dir() else {
        eprintln!("gliner2.5-small-v1 not found; skipping the loader gate");
        return;
    };
    let _guard = mlx_test_guard();
    let mut loaded = load_span_extractor(&dir).expect("loads through the detector");
    assert_eq!(loaded.kind, TokenModelKind::Gliner2_5);
    assert_eq!(loaded.model_type, "gliner2");
    let labels = ["company", "person", "location"].map(|l| (l.to_string(), None));
    let result = loaded
        .extractor
        .extract_entities(
            "Apple hired Sam in New York.",
            &labels,
            0.5,
            ExtractOptions::default(),
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        json!({"entities": {"company": ["Apple"], "person": ["Sam"], "location": ["New York"]}})
    );
    let tasks = vec![(
        "sentiment".to_string(),
        ClassifyTask::Labels(
            ["positive", "neutral", "negative"]
                .map(str::to_string)
                .to_vec(),
        ),
    )];
    let classified = loaded
        .extractor
        .classify_text("The launch was excellent.", &tasks, 0.5)
        .unwrap();
    assert_eq!(classified, json!({"sentiment": "positive"}));
    eprintln!(
        "loader gate ok: {} / {classified}",
        serde_json::to_string(&result).unwrap()
    );

    // Every label abstains or scores below threshold: one key per label, no spans.
    let unrelated = loaded
        .extractor
        .extract_entities(
            "The sky is blue.",
            &[("chemical compound".to_string(), None)],
            0.99,
            ExtractOptions::default(),
        )
        .unwrap();
    assert_eq!(unrelated.entities.len(), 1);
    assert_eq!(unrelated.entities[0].0, "chemical compound");
}

#[test]
fn gliner2_5_small_refuses_text_past_max_len_gracefully() {
    let Some(dir) = gliner2_5_small_dir() else {
        return;
    };
    let _guard = mlx_test_guard();
    let model = Gliner2_5Model::load_dir(&dir).expect("gliner2.5-small-v1 loads");
    let schema: Vec<String> = (0..3000).map(|i| format!("label{i}")).collect();
    let err = model.prepare("Some text.", &schema).err().unwrap();
    assert!(
        err.to_string().contains("schema alone exceeds max_len"),
        "{err}"
    );
    // A text longer than max_len is truncated at a word boundary, never cut
    // inside a word.
    let long = "word ".repeat(5000);
    let schema = schema_tokens("entities", &["thing".to_string()], "[E]", None, |_| None);
    let prepared = model.prepare(&long, &schema).unwrap();
    assert!(prepared.input_ids.len() <= 4096);
    assert!(prepared.offsets.len() < 5000);
    assert_eq!(prepared.offsets.len(), prepared.word_positions.len());
}

/// About 300 words, past the 128 tokens where the parity fixture stops.
const LONG_TEXT: &str = "Ada Lovelace worked with Charles Babbage in London on the Analytical Engine. \
Grace Hopper joined Harvard in 1944 and later founded a consulting company in Philadelphia. \
Alan Turing moved from Cambridge to Manchester after the war and wrote about machine intelligence. \
Margaret Hamilton led the software team at MIT that put Apollo on the Moon in 1969. \
Linus Torvalds released Linux from Helsinki in 1991 and now lives near Portland. \
Barbara Liskov received the Turing Award in 2008 for her work at MIT on data abstraction. \
Tim Berners-Lee built the first web server at CERN near Geneva in 1990 and later founded the W3C in Boston. \
Katherine Johnson computed trajectories for NASA at Langley in Virginia for more than thirty years. \
Dennis Ritchie and Ken Thompson designed Unix at Bell Labs in Murray Hill, New Jersey, in the early seventies. \
Radia Perlman wrote the spanning tree protocol at Digital Equipment Corporation in Massachusetts. \
Guido van Rossum started Python in Amsterdam in 1989 and later worked at Google and Dropbox in California. \
Frances Allen joined IBM in New York in 1957 and became the first woman to win the Turing Award in 2006.";

#[test]
fn gliner2_5_small_is_deterministic_on_a_long_text() {
    let Some(dir) = gliner2_5_small_dir() else {
        return;
    };
    let _guard = mlx_test_guard();
    let mut model = Gliner2_5Model::load_dir(&dir).expect("gliner2.5-small-v1 loads");
    let labels = ["person", "organization", "location", "date"].map(|l| (l.to_string(), None));
    let schema = schema_tokens("entities", &["person".to_string()], "[E]", None, |_| None);
    let tokens = model.prepare(LONG_TEXT, &schema).unwrap().input_ids.len();
    assert!(
        tokens > 200,
        "the text must exceed the fixture range, got {tokens} tokens"
    );
    let options = ExtractOptions {
        include_confidence: true,
        include_spans: true,
    };
    let first = model
        .extract_entities(LONG_TEXT, &labels, 0.5, options)
        .unwrap();
    assert!(
        first.entities.iter().any(|(_, items)| !items.is_empty()),
        "the long text yields some span"
    );
    for round in 1..4 {
        let again = model
            .extract_entities(LONG_TEXT, &labels, 0.5, options)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&again).unwrap(),
            serde_json::to_value(&first).unwrap(),
            "round {round} differs from the first extraction on {tokens} tokens"
        );
    }
    eprintln!(
        "determinism ok on {tokens} tokens: {}",
        serde_json::to_string(&first).unwrap()
    );
}
