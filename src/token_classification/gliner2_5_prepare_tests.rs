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

//! `prepare` on an in-process word-level tokenizer: sequence layout, the two
//! `max_len` failures and word-boundary truncation, without a checkpoint.

use serde_json::json;
use tokenizers::Tokenizer;

use super::*;

const UNK: u32 = 0;

/// Lower-case words of the README example plus the schema punctuation as a
/// word-level `tokenizer.json`; the five GLiNER2.5 special tokens are added
/// tokens, as in the checkpoint.
fn tokenizer() -> Tokenizer {
    let words = [
        "[UNK]", "(", ")", "entities", "apple", "hired", "sam", "in", "new", "york", "-",
        "company", "person", ".",
    ];
    let vocab: serde_json::Map<String, serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), json!(id)))
        .collect();
    let added_tokens: Vec<serde_json::Value> = SPECIAL_TOKENS
        .iter()
        .enumerate()
        .map(|(offset, token)| {
            json!({
                "id": words.len() + offset,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let definition = json!({
        "version": "1.0",
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    });
    Tokenizer::from_bytes(definition.to_string().as_bytes()).expect("word-level tokenizer parses")
}

fn schema() -> Vec<String> {
    schema_tokens(
        "entities",
        &["company".to_string(), "person".to_string()],
        "[E]",
        None,
        |_| None,
    )
}

#[test]
fn prepare_lays_out_schema_separator_and_words() {
    let tokenizer = tokenizer();
    let prepared = prepare(
        &tokenizer,
        "Apple hired Sam",
        &schema(),
        4096,
        WordSplitter::Whitespace,
        UNK,
    )
    .unwrap();
    // `( [P] entities ( [E] company [E] person ) ) [SEP_TEXT]` is 11 tokens.
    assert_eq!(prepared.marker_positions, [1, 4, 6]);
    assert_eq!(prepared.word_positions, [11, 12, 13]);
    assert_eq!(prepared.input_ids.len(), 14);
    assert_eq!(prepared.offsets, [(0, 5), (6, 11), (12, 15)]);
    assert_eq!(
        prepared.text, "Apple hired Sam.",
        "a terminal `.` is appended"
    );
    let id = |token: &str| tokenizer.token_to_id(token).unwrap();
    assert_eq!(prepared.input_ids[1], id("[P]"));
    assert_eq!(prepared.input_ids[4], id("[E]"));
    assert_eq!(prepared.input_ids[10], id("[SEP_TEXT]"));
    assert_eq!(prepared.input_ids[11], id("apple"));
}

#[test]
fn prepare_refuses_a_schema_at_or_over_max_len() {
    let err = prepare(
        &tokenizer(),
        "Apple",
        &schema(),
        11,
        WordSplitter::Whitespace,
        UNK,
    )
    .err()
    .unwrap();
    assert_eq!(err.to_string(), "schema alone exceeds max_len");
}

#[test]
fn prepare_refuses_a_first_word_that_does_not_fit() {
    // `new-york` is one word to the splitter and three pieces to this
    // tokenizer, so with one slot left nothing fits.
    let err = prepare(
        &tokenizer(),
        "New-York",
        &schema(),
        12,
        WordSplitter::Whitespace,
        UNK,
    )
    .err()
    .unwrap();
    assert_eq!(err.to_string(), "no text tokens fit within max_len");
}

#[test]
fn prepare_truncates_at_a_word_boundary() {
    let prepared = prepare(
        &tokenizer(),
        "Apple hired Sam in New York",
        &schema(),
        13,
        WordSplitter::Whitespace,
        UNK,
    )
    .unwrap();
    assert_eq!(prepared.input_ids.len(), 13);
    assert_eq!(prepared.word_positions, [11, 12]);
    assert_eq!(
        prepared.offsets,
        [(0, 5), (6, 11)],
        "four of six words dropped"
    );
}

#[test]
fn prepare_maps_an_unknown_word_to_its_pieces_not_to_nothing() {
    let prepared = prepare(
        &tokenizer(),
        "Zebra hired Sam",
        &schema(),
        4096,
        WordSplitter::Whitespace,
        UNK,
    )
    .unwrap();
    assert_eq!(prepared.input_ids[11], UNK);
    assert_eq!(prepared.offsets.len(), 3);
}
