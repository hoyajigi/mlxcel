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

//! GLiNER2.5 input preparation and span bookkeeping, ported from
//! `mlx_vlm/models/gliner2_5/gliner2_5.py` (`_split_words`,
//! `_schema_tokens`, `Model._prepare`, `_resolve_flat_overlaps`).
//!
//! One request becomes one sequence: the schema (prompt and label markers),
//! `[SEP_TEXT]`, then the sub-tokens of every word of the text. Nothing here
//! touches MLX; the model module gathers the encoder rows at the word and
//! marker positions this module reports.

use std::sync::LazyLock;

use anyhow::{Result, bail};
use fancy_regex::Regex;
use tokenizers::Tokenizer;

/// Tokens a GLiNER2.5 tokenizer must already define.
pub const SPECIAL_TOKENS: &[&str] = &["[SEP_TEXT]", "[P]", "[E]", "[L]", "[DESCRIPTION]"];

/// How the text is cut into words before sub-tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WordSplitter {
    /// Upstream's default: URLs, e-mail addresses and handles stay one word,
    /// `\w+` runs may be joined by `-` / `_`, any other non-space is a word.
    #[default]
    Whitespace,
    /// ASCII runs are words and every other non-space character is its own
    /// word, for scripts without word spacing.
    Char,
}

// Upstream's pattern with two deliberate edits. Python's `\w` is
// `str.isalnum()` plus `_` (letters and numbers, no combining marks, no
// joiners), which is `[\p{L}\p{N}_]`; the regex crate's `\w` also matches
// marks and join controls, so Devanagari, Persian ZWNJ and `x²` would split
// differently from the model's training. The e-mail branch is bounded to the
// RFC 5321 lengths (64 / 253) because its unbounded run before the `@` rescans
// the rest of a long `a.a.a...` run on every word, which is quadratic in the
// text (upstream has the same cost); only addresses that are invalid anyway
// match differently. Both patterns are literals, so a compile failure is a
// programming error, not an input error.
static WHITESPACE_WORDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]{1,64}@[a-z0-9.-]{1,253}\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S",
    )
    .expect("whitespace word pattern compiles")
});

static CHAR_WORDS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z0-9@._\-+]+|\S").expect("char word pattern compiles"));

/// One word of the input: lower-cased text and its Unicode scalar offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Cut `text` into words the way upstream `_split_words` does. Offsets are
/// Unicode scalar indices, the indices Python reports.
pub fn split_words(text: &str, splitter: WordSplitter) -> Vec<Word> {
    let pattern = match splitter {
        WordSplitter::Whitespace => &*WHITESPACE_WORDS,
        WordSplitter::Char => &*CHAR_WORDS,
    };
    // Byte offset -> scalar index, for every char boundary.
    let mut char_at_byte = vec![0usize; text.len() + 1];
    for (chars, (byte, _)) in text.char_indices().enumerate() {
        char_at_byte[byte] = chars;
    }
    char_at_byte[text.len()] = text.chars().count();
    pattern
        .find_iter(text)
        .filter_map(|found| found.ok())
        .map(|found| Word {
            text: found.as_str().to_lowercase(),
            start: char_at_byte[found.start()],
            end: char_at_byte[found.end()],
        })
        .collect()
}

/// The schema tokens of one request:
/// `( [P] <prompt> ( <marker> <label> ... ) )`, where the prompt is the
/// parent name, optionally `"<parent>: <prompt>"`, followed by
/// ` [DESCRIPTION] <label>: <description>` for every described label.
pub fn schema_tokens<'a>(
    parent: &str,
    labels: &[String],
    marker: &str,
    prompt: Option<&str>,
    description: impl Fn(&str) -> Option<&'a str>,
) -> Vec<String> {
    // An empty prompt falls back to the bare parent, as upstream's `if prompt`.
    let mut prompt_text = match prompt.filter(|prompt| !prompt.is_empty()) {
        Some(prompt) => format!("{parent}: {prompt}"),
        None => parent.to_string(),
    };
    for label in labels {
        if let Some(text) = description(label) {
            prompt_text.push_str(&format!(" [DESCRIPTION] {label}: {text}"));
        }
    }
    let mut output = vec![
        "(".to_string(),
        "[P]".to_string(),
        prompt_text,
        "(".to_string(),
    ];
    for label in labels {
        output.push(marker.to_string());
        output.push(label.clone());
    }
    output.push(")".to_string());
    output.push(")".to_string());
    output
}

/// Schema indices whose first sub-token is a marker position: the `[P]`
/// token at 1 and every label marker at `4, 6, ..., len - 3`.
fn is_marker_slot(index: usize, schema_len: usize) -> bool {
    index == 1 || (index >= 4 && index + 2 < schema_len && (index - 4).is_multiple_of(2))
}

/// One encoded request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedInput {
    /// The whole sequence: schema, `[SEP_TEXT]`, text sub-tokens.
    pub input_ids: Vec<u32>,
    /// Position of each word's first sub-token.
    pub word_positions: Vec<usize>,
    /// Position of the `[P]` token and of each label marker.
    pub marker_positions: Vec<usize>,
    /// Unicode scalar `(start, end)` of each word that fit.
    pub offsets: Vec<(usize, usize)>,
    /// The text the offsets index: the input, plus a `.` when it did not end
    /// in `.`, `!` or `?`.
    pub text: String,
}

/// Tokenize `text` without special tokens, as `PreTrainedTokenizerFast.tokenize`.
fn pieces(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer.encode(text, false).map_err(|e| {
        anyhow::anyhow!(
            "GLiNER2.5 tokenization of a {}-char piece failed: {e}",
            text.chars().count()
        )
    })?;
    Ok(encoding.get_ids().to_vec())
}

/// Build the sequence for `schema` over `text`, upstream `Model._prepare`.
///
/// `unk_id` stands in for a word the tokenizer maps to no piece. Words are
/// appended until the next one would push the sequence past `max_len`.
pub fn prepare(
    tokenizer: &Tokenizer,
    text: &str,
    schema: &[String],
    max_len: usize,
    splitter: WordSplitter,
    unk_id: u32,
) -> Result<PreparedInput> {
    let words = split_words(text, splitter);
    let mut text = text.to_string();
    if !text.trim_end().ends_with(['.', '!', '?']) {
        text.push('.');
    }
    let mut input_ids: Vec<u32> = Vec::new();
    let mut marker_positions = Vec::new();
    for (index, token) in schema.iter().enumerate() {
        if is_marker_slot(index, schema.len()) {
            marker_positions.push(input_ids.len());
        }
        input_ids.extend(pieces(tokenizer, token)?);
    }
    input_ids.extend(pieces(tokenizer, "[SEP_TEXT]")?);
    if input_ids.len() >= max_len {
        bail!("schema alone exceeds max_len");
    }
    let mut word_positions = Vec::new();
    let mut offsets = Vec::new();
    for word in &words {
        let mut ids = pieces(tokenizer, &word.text)?;
        if ids.is_empty() {
            ids.push(unk_id);
        }
        if input_ids.len() + ids.len() > max_len {
            break;
        }
        word_positions.push(input_ids.len());
        offsets.push((word.start, word.end));
        input_ids.extend(ids);
    }
    if offsets.is_empty() {
        bail!("no text tokens fit within max_len");
    }
    if offsets.len() < words.len() {
        tracing::warn!(
            "GLiNER2.5: only the first {} of {} words fit within max_len {max_len}; the rest of the text is not scored",
            offsets.len(),
            words.len()
        );
    }
    Ok(PreparedInput {
        input_ids,
        word_positions,
        marker_positions,
        offsets,
        text,
    })
}

/// A candidate span: `(score, start word, end word)` with `end` exclusive.
pub type ScoredSpan = (f32, usize, usize);

/// Keep the non-overlapping subset of `spans` with the largest total score
/// (weighted interval scheduling), upstream `_resolve_flat_overlaps`.
///
/// Candidates are ordered by `(end, start, -score)`; the dynamic program
/// prefers skipping a span on an exact tie, as upstream's strict `>` does.
/// The survivors come back by descending score, then start, then end.
pub fn resolve_flat_overlaps(spans: &[ScoredSpan]) -> Vec<ScoredSpan> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut ordered: Vec<ScoredSpan> = spans.to_vec();
    ordered.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)).then(b.0.total_cmp(&a.0)));
    let ends: Vec<usize> = ordered.iter().map(|span| span.2).collect();
    // `best[i]` is the best total over the first `i` candidates and which
    // ones it keeps. Totals accumulate in f64, as Python floats do, so a
    // near-tie resolves the same way as upstream.
    let mut best: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
    for (index, span) in ordered.iter().enumerate() {
        // Rightmost earlier candidate ending at or before this start.
        let predecessor = ends[..index].partition_point(|&end| end <= span.1);
        let (previous_score, previous) = &best[predecessor];
        let selected_score = previous_score + f64::from(span.0);
        let skipped = &best[index];
        if selected_score > skipped.0 {
            let mut kept = previous.clone();
            kept.push(index);
            best.push((selected_score, kept));
        } else {
            let skipped = skipped.clone();
            best.push(skipped);
        }
    }
    let mut kept: Vec<ScoredSpan> = best
        .last()
        .map(|(_, indices)| indices.iter().map(|&i| ordered[i]).collect())
        .unwrap_or_default();
    kept.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    kept
}

#[cfg(test)]
#[path = "gliner2_5_prepare_tests.rs"]
mod gliner2_5_prepare_tests;

/// `text[start..end]` by Unicode scalar index.
pub fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}
