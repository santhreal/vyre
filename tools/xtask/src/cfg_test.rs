//! Which spans of a Rust source are gated behind `#[cfg(test)]`.
//!
//! A rule that judges production code has to see the tree the production build
//! sees. A fixture registration, a test-only module declaration and a doubled
//! helper all live behind a test gate, and counting them reports code that no
//! shipped binary holds. The spans are found by scanning text, so a crate that
//! does not compile is still judged.

use std::borrow::Cow;

use crate::source_scan::opaque_span;

/// Remove every `#[cfg(test)]`-gated item before a production-code scan.
///
/// A test module registers fixture operations - `test::reference_echo`,
/// `test::call_u32` and friends - that exist in no production build. Counting
/// them as registry members reported four test doubles as misplaced production
/// operations, and pointed Phase 2 at code that was already correct.
///
/// The predicate is tokenized with string literals removed first, so
/// `#[cfg(feature = "test-utils")]` is not mistaken for a test gate.
pub fn strip_cfg_test_items(text: &str) -> Cow<'_, str> {
    let spans = cfg_test_spans(text);
    if spans.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut kept = String::with_capacity(text.len());
    let mut kept_from = 0usize;
    for span in spans {
        kept.push_str(&text[kept_from..span.0]);
        kept_from = span.1;
    }
    kept.push_str(&text[kept_from..]);
    Cow::Owned(kept)
}

/// Every `#[cfg(test)]`-gated item of one file, concatenated in source order.
///
/// The complement of [`strip_cfg_test_items`], for a caller that judges test
/// code rather than production code. Both read the same spans, so a scanner
/// improvement lands on both views at once: the coverage corpus that used
/// "everything after the first `#[cfg(test)]` marker" instead counted a crate's
/// production re-export list as test text, and 174 runtime symbols were
/// "covered" by a `pub use` block that names them.
#[must_use]
pub fn cfg_test_items(text: &str) -> String {
    let mut out = String::new();
    for (start, end) in cfg_test_spans(text) {
        out.push_str(&text[start..end]);
        out.push('\n');
    }
    out
}

/// Names of the modules one file declares behind a `#[cfg(test)]` gate.
///
/// A test module written in its own file - `mod tests;` beside `tests/mod.rs`,
/// or `mod core_tests;` beside `core_tests.rs` - carries no gating attribute of
/// its own, so [`cfg_test_items`] over that file returns nothing and a caller
/// judging test text would read the file as production code. The declaration is
/// the only place the gate is written, so it is read here rather than inferred
/// from a file name: `tests.rs` is a test module because a `#[cfg(test)] mod
/// tests;` says so, and a crate that ships a production module of that name is
/// not misread.
#[must_use]
pub fn cfg_test_module_declarations(text: &str) -> Vec<String> {
    cfg_test_spans_detailed(text)
        .into_iter()
        .filter_map(|span| declared_module_name(&text[span.start..span.end]))
        .collect()
}

/// Name of the module a gated declaration names, if the item is one.
fn declared_module_name(span: &str) -> Option<String> {
    let body = span.trim_end().strip_suffix(';')?;
    let name = body.rsplit_once("mod ")?.1.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(name.to_string())
}

/// For each line of `text`, whether it sits inside a `#[cfg(test)]`-gated item.
///
/// A third view of the same spans [`strip_cfg_test_items`] and [`cfg_test_items`]
/// read, for a caller that reports a line number. Stripping the text first would
/// shift every line after the first test module, so a finding would name a line
/// the reader cannot open, and rebuilding the mapping from the stripped text is
/// how the two views drift.
///
/// A line the spans cover only partly counts as test text: the attribute and the
/// item's opening line hold no production code, and the closing line holds a
/// brace.
#[must_use]
pub fn cfg_test_line_mask(text: &str) -> Vec<bool> {
    let mut mask = vec![false; text.lines().count()];
    for (start, end) in cfg_test_spans(text) {
        let first = line_index(text, start);
        let last = line_index(text, end.saturating_sub(1).max(start));
        for line in first..=last {
            if let Some(flag) = mask.get_mut(line) {
                *flag = true;
            }
        }
    }
    mask
}

/// The zero-based line `offset` falls on, counted from the newlines before it.
fn line_index(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())].matches('\n').count()
}

/// Byte spans of every test-gated item, attribute included, in source order.
///
/// Text not yet accounted for and where the next attribute is looked for are
/// separate cursors: a non-test `#[cfg(...)]` moves the search past its
/// predicate but keeps every byte, and sharing one cursor for both deleted the
/// whole file up to the last non-test attribute. That silently dropped the
/// `const` an id resolved through, so a real registration became no registration
/// and the rules below judged a registry they could not see.
fn cfg_test_spans(text: &str) -> Vec<(usize, usize)> {
    cfg_test_spans_detailed(text)
        .into_iter()
        .map(|span| (span.start, span.end))
        .collect()
}

/// One test-gated item: its byte span and whether the gate needs `test` on.
struct CfgTestSpan {
    start: usize,
    end: usize,
}

fn cfg_test_spans_detailed(text: &str) -> Vec<CfgTestSpan> {
    const ATTR: &str = "#[cfg(";
    let mut spans = Vec::new();
    let mut search = 0usize;
    while search < text.len() {
        if let Some(span) = opaque_span(text, search) {
            search += span.get();
            continue;
        }
        if !text.as_bytes()[search..].starts_with(ATTR.as_bytes()) {
            search += 1;
            continue;
        }
        let attr_start = search;
        let predicate_start = attr_start + ATTR.len() - 1;
        let Some(predicate_end) = match_delimited(text, predicate_start, b'(', b')') else {
            search += ATTR.len();
            continue;
        };
        let predicate = &text[predicate_start + 1..predicate_end];
        if !mentions_test(predicate) || !compiles_with_test(predicate) {
            search = predicate_end + 1;
            continue;
        }
        let Some(attr_end) = text[predicate_end..].find(']').map(|at| predicate_end + at) else {
            search = predicate_end + 1;
            continue;
        };
        let Some(item_end) = end_of_item(text, attr_end + 1) else {
            search = attr_end + 1;
            continue;
        };
        spans.push(CfgTestSpan {
            start: attr_start,
            end: item_end,
        });
        search = item_end;
    }
    spans
}

/// True when some configuration with `test` on compiles the item.
///
/// `#[cfg(not(test))]` names `test` and is the opposite of a test gate: it
/// compiles only into a build without `test`, so its item is production code
/// and a scan of production text has to keep reading it. Atoms other than
/// `test` are free: a `feature` may be on or off, so a predicate naming one is
/// satisfiable unless `test` itself forbids it.
fn compiles_with_test(predicate: &str) -> bool {
    let predicate = predicate.trim();
    if let Some(rest) = predicate.strip_prefix("all(") {
        return arguments(rest).iter().all(|part| compiles_with_test(part));
    }
    if let Some(rest) = predicate.strip_prefix("any(") {
        let parts = arguments(rest);
        return parts.is_empty() || parts.iter().any(|part| compiles_with_test(part));
    }
    if let Some(rest) = predicate.strip_prefix("not(") {
        return arguments(rest)
            .iter()
            .all(|part| compiles_without_test(part));
    }
    true
}

/// True when some configuration with `test` on leaves the predicate false.
///
/// The dual of [`compiles_with_test`], for reading a `not(..)`. `test` itself
/// is on, so it cannot be false; every other atom can be.
fn compiles_without_test(predicate: &str) -> bool {
    let predicate = predicate.trim();
    if let Some(rest) = predicate.strip_prefix("all(") {
        return arguments(rest)
            .iter()
            .any(|part| compiles_without_test(part));
    }
    if let Some(rest) = predicate.strip_prefix("any(") {
        let parts = arguments(rest);
        return parts.is_empty() || parts.iter().all(|part| compiles_without_test(part));
    }
    if let Some(rest) = predicate.strip_prefix("not(") {
        return arguments(rest).iter().all(|part| compiles_with_test(part));
    }
    predicate != "test"
}

/// Top-level comma-separated arguments of a predicate list, closing `)` dropped.
fn arguments(rest: &str) -> Vec<&str> {
    let Some(end) = closing_paren(rest) else {
        return Vec::new();
    };
    let inner = &rest[..end];
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut offset = 0usize;
    let bytes = inner.as_bytes();
    while offset < bytes.len() {
        if let Some(span) = opaque_span(inner, offset) {
            offset += span.get();
            continue;
        }
        match bytes[offset] {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                parts.push(inner[start..offset].trim());
                start = offset + 1;
            }
            _ => {}
        }
        offset += 1;
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts
}

/// Byte index of the `)` closing the list `rest` starts inside.
fn closing_paren(rest: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut offset = 0usize;
    let bytes = rest.as_bytes();
    while offset < bytes.len() {
        if let Some(span) = opaque_span(rest, offset) {
            offset += span.get();
            continue;
        }
        match bytes[offset] {
            b'(' => depth += 1,
            b')' if depth == 0 => return Some(offset),
            b')' => depth -= 1,
            _ => {}
        }
        offset += 1;
    }
    None
}

/// Byte index of the delimiter closing the one that opens at `open`.
///
/// Delimiters inside a string, char literal, raw string or comment are text:
/// counting them ends a `#[cfg(test)] mod tests { .. }` at a `}` written inside
/// a string and leaves the rest of the test module in the scanned text.
fn match_delimited(text: &str, open: usize, opener: u8, closer: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&opener) {
        return None;
    }
    let mut depth = 0usize;
    let mut index = open;
    while index < bytes.len() {
        if let Some(span) = opaque_span(text, index) {
            index += span.get();
            continue;
        }
        if bytes[index] == opener {
            depth += 1;
        } else if bytes[index] == closer {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

/// True when the cfg predicate names `test` as a bare token.
fn mentions_test(predicate: &str) -> bool {
    let mut outside = String::with_capacity(predicate.len());
    let mut offset = 0usize;
    while offset < predicate.len() {
        if let Some(span) = opaque_span(predicate, offset) {
            offset += span.get();
            continue;
        }
        if let Some(character) = predicate[offset..].chars().next() {
            outside.push(character);
            offset += character.len_utf8();
        } else {
            break;
        }
    }
    outside
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .any(|token| token == "test")
}

/// End of the item a gating attribute applies to, past `from`.
///
/// A braced item ends at its matching `}`; a declaration such as
/// `#[cfg(test)] mod tests;` ends at its `;`. Further attributes stacked on the
/// same item are skipped so the whole item is removed, not just the tail.
fn end_of_item(text: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = from;
    while index < bytes.len() {
        if let Some(span) = opaque_span(text, index) {
            index += span.get();
            continue;
        }
        match bytes[index] {
            b'{' => return match_delimited(text, index, b'{', b'}').map(|close| close + 1),
            b';' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}
