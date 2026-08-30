use std::borrow::Cow;
use std::io;
use std::path::Path;

use super::classification::is_xtask_source_path;
use super::records::MAX_HYGIENE_SCAN_FILE_BYTES;

/// True when a `#[cfg(...)]` attribute gates the item to test builds only.
///
/// Any predicate mentioning the `test` cfg compiles only in a test build, so the item
/// behind it is test code no matter how the predicate is spelled. The previous version
/// listed four exact spellings and missed `#[cfg(all(test, feature = "..."))]`, which is
/// how the regex scan suites gate themselves: four `mod tests` blocks were scanned as
/// production source and their test helpers reported as release blockers. `not(test)` is
/// the opposite gate and stays in scope.
pub(crate) fn is_non_release_cfg_attr(trimmed: &str) -> bool {
    if !trimmed.starts_with("#[cfg(") || trimmed.contains("not(test)") {
        return false;
    }
    let predicate = trimmed
        .trim_start_matches("#[cfg(")
        .trim_end_matches(")]")
        .trim_end_matches(')');
    predicate
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|token| token == "test")
}

/// True when `line` calls a filesystem read or reads a stream to the end.
///
/// The `fs::` forms are matched as a path segment, not as a substring. A plain
/// `line.contains("fs::read(")` also matched `BufferRefs::read(count_buffer)`,
/// whose type name happens to end in `fs`; that call reads a GPU buffer
/// reference and has no file, no length, and nothing to bound. A false positive
/// here is not harmless: it is a permanent release blocker on correct code, and
/// the only way to clear it would have been to rename the type.
pub(crate) fn line_contains_read_call(line: &str) -> bool {
    calls_path_function(line, "fs::read_to_string")
        || calls_path_function(line, "fs::read")
        || line.contains(".read_to_end(")
        || line.contains(".read_to_string(")
}

/// True when `line` calls `name` as a whole path segment rather than as a suffix.
pub(crate) fn calls_path_function(line: &str, name: &str) -> bool {
    line.match_indices(name)
        .any(|(index, _)| is_word_start(line, index) && line[index + name.len()..].starts_with('('))
}

pub(crate) fn line_contains_unbounded_read(path: &Path, line: &str) -> bool {
    let normalized = path.to_string_lossy();
    if is_xtask_source_path(&normalized.replace('\\', "/")) {
        return false;
    }
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || is_release_rule_text(trimmed) {
        return false;
    }
    if trimmed.contains(".take(") {
        return false;
    }
    line_contains_read_call(trimmed)
}

/// The `Duration` readers that answer a `u128`.
const U128_DURATION_READERS: [&str; 3] = ["as_nanos()", "as_micros()", "as_millis()"];

/// The `as` casts that discard high bits of a `u128`.
const NARROWING_CASTS: [&str; 10] = [
    "asu8", "asu16", "asu32", "asu64", "asusize", "asi8", "asi16", "asi32", "asi64", "asisize",
];

/// The 1-based lines of `text` that narrow a `u128` duration count with an `as`
/// cast.
///
/// `Duration::as_nanos`, `as_micros` and `as_millis` all answer a `u128`, and an
/// `as` cast to any integer discards the high bits instead of clamping, so a
/// count past the target maximum is reported as an unrelated small number. Three
/// spellings of the narrowing coexisted in `vyre-bench`, and the crate-local gate
/// that closed them there cannot see the same cast in `vyre-foundation` or
/// `vyre-runtime`, which is where two of them survived a year. The saturating
/// spelling is `u64::try_from(count).unwrap_or(u64::MAX)`.
///
/// Comment lines and rule text are skipped, and the match runs over code with the
/// whitespace removed, so a cast rustfmt split across two lines is caught the
/// same as a cast written on one. `as_secs` is not read: it already answers a
/// `u64`, so casting it to `u64` narrows nothing.
pub(crate) fn truncating_duration_cast_lines(path: &Path, text: &str) -> Vec<usize> {
    let normalized = path.to_string_lossy().replace('\\', "/");
    if is_xtask_source_path(&normalized) {
        return Vec::new();
    }
    let mut dense = String::with_capacity(text.len());
    let mut source_line = Vec::with_capacity(text.len());
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || is_release_rule_text(trimmed) {
            continue;
        }
        for character in trimmed
            .chars()
            .filter(|character| !character.is_whitespace())
        {
            dense.push(character);
            source_line.push(index + 1);
        }
    }
    let mut lines = Vec::new();
    for reader in U128_DURATION_READERS {
        for (at, _) in dense.match_indices(reader) {
            let after = at + reader.len();
            if !NARROWING_CASTS
                .iter()
                .any(|cast| dense[after..].starts_with(cast))
            {
                continue;
            }
            if let Some(line) = source_line.get(after) {
                lines.push(*line);
            }
        }
    }
    lines.sort_unstable();
    lines.dedup();
    lines
}

/// True when the panicking call at `line_index` sits in a function whose docs declare a
/// `# Panics` section.
///
/// A panic in production code is acceptable only when failing closed IS the contract and
/// the contract is written down. Vyre's panicking functions are infallible wrappers over
/// `try_*` twins, because the quiet alternative (return an empty match set, an empty
/// table, no offsets) reports a dirty input as clean and is a total recall-loss silent
/// fallback (Law 10). Rust already has one place to record that: the `# Panics` doc
/// section, which `clippy::missing_panics_doc` enforces the same way. The gate reads the
/// docs instead of keeping a second allowlist file that would drift out of date, and an
/// undocumented panic stays a release blocker.
pub(crate) fn has_documented_panic_contract(text: &str, line_index: usize) -> bool {
    let lines = text.lines().collect::<Vec<_>>();
    let Some(site) = lines.get(line_index) else {
        return false;
    };
    let site_indent = site.len() - site.trim_start().len();
    let mut cursor = line_index;
    while cursor > 0 {
        cursor -= 1;
        let line = lines[cursor];
        let trimmed = line.trim_start();
        // Only an enclosing item counts: a signature at or past the call's own indent
        // belongs to a sibling, not to the function the call sits in.
        if line.len() - trimmed.len() >= site_indent || !is_fn_signature_line(trimmed) {
            continue;
        }
        let mut doc = cursor;
        while doc > 0 {
            doc -= 1;
            let previous = lines[doc].trim();
            if previous.starts_with("///") || previous.starts_with("//!") {
                if previous.contains("# Panics") {
                    return true;
                }
                continue;
            }
            // Attributes and plain `//` notes sit between a doc block and its signature
            // (`// INTENTIONAL: ...` above `#[allow(clippy::expect_used)]` is the house
            // style for a deliberate panic), so walking up must step over them or the doc
            // block is never reached.
            if previous.is_empty() || previous.starts_with("#[") || previous.starts_with("//") {
                continue;
            }
            break;
        }
        return false;
    }
    false
}

/// True when `trimmed` opens a function signature, whatever the leading keywords.
pub(crate) fn is_fn_signature_line(trimmed: &str) -> bool {
    let mut rest = trimmed;
    loop {
        if rest.starts_with("fn ") {
            return true;
        }
        if let Some(restricted) = rest.strip_prefix("pub(") {
            let Some(close) = restricted.find(')') else {
                return false;
            };
            rest = restricted[close + 1..].trim_start();
            continue;
        }
        let Some((head, tail)) = rest.split_once(' ') else {
            return false;
        };
        let is_signature_keyword = head.starts_with("pub")
            || head.starts_with("extern")
            || head.starts_with('"')
            || matches!(head, "const" | "async" | "unsafe" | "default");
        if !is_signature_keyword {
            return false;
        }
        rest = tail.trim_start();
    }
}

pub(crate) fn line_contains_blocked_pattern(
    path: &Path,
    name: &str,
    pattern: &str,
    line: &str,
    lower: &str,
) -> bool {
    let trimmed = line.trim();
    if is_code_call_blocker(name)
        && (is_rust_doc_comment_line(trimmed) || pattern_only_inside_literal(pattern, line))
    {
        return false;
    }
    if is_hygiene_rule_source(path) {
        return false;
    }
    if is_hidden_fallback_pattern(name) && is_hidden_fallback_guard_source(path) {
        return false;
    }
    if is_hidden_fallback_pattern(name) && is_negated_hidden_fallback_statement(lower) {
        return false;
    }
    if name == "cfg_not_gpu" && !line_cfg_not_gpu_hides_work(lower) {
        return false;
    }
    if is_release_rule_text(trimmed) {
        return false;
    }
    match name {
        "placeholder_text" => contains_word(lower, pattern),
        "stub_text" => contains_word(lower, pattern),
        "not_implemented_text" => lower.contains(pattern),
        "TODO" | "FIXME" => line.contains(pattern),
        _ => line.contains(pattern) || lower.contains(pattern),
    }
}

pub(crate) fn is_rust_doc_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("///") || trimmed.starts_with("//!")
}

pub(crate) fn is_code_call_blocker(name: &str) -> bool {
    matches!(
        name,
        "panic_macro"
            | "unwrap_call"
            | "expect_call"
            | "todo_macro"
            | "unimplemented_macro"
            | "not_implemented_text"
    )
}

/// Whether a code-call pattern appears only inside string literals on `line`.
///
/// A gate that detects `todo!(` has to spell `todo!(` to detect it, and a
/// pattern table row reading `text: "todo!(",` is that spelling, not a stub.
/// The rule already exempted a doc comment for the same reason. This is the
/// other half: a string literal names a call, it does not make one. It applies
/// to the code-call family only, because the hidden-fallback family is meant to
/// read prose and printed excuses, where the literal IS the evidence.
pub(crate) fn pattern_only_inside_literal(pattern: &str, line: &str) -> bool {
    let masked = crate::gates::scan::mask_literals(line);
    line.contains(pattern) && !masked.contains(pattern)
}

pub(crate) fn is_hidden_fallback_pattern(name: &str) -> bool {
    matches!(
        name,
        "silent_gpu_skip"
            | "silent_gpu_skipped"
            | "gpu_unavailable_skip"
            | "cfg_not_gpu"
            | "cpu_fallback"
            | "software_fallback"
            | "fallback_dispatch"
            | "falling_back_to_cpu"
            | "fallback_to_cpu"
            | "synthetic_gpu_timing"
            | "fake_gpu_timing_formula"
    )
}

/// Whether `name` states unfinished work rather than a call or an excuse.
///
/// The family covers an unfinished item, a placeholder, and the macros and
/// prose that stand in for an implementation. It overlaps the code-call
/// family, which decides whether a literal-only occurrence is exempt: the two
/// answer different questions about the same row.
pub(crate) fn is_unresolved_marker_pattern(name: &str) -> bool {
    matches!(
        name,
        "TODO"
            | "FIXME"
            | "placeholder_text"
            | "stub_text"
            | "not_implemented_text"
            | "todo_macro"
            | "unimplemented_macro"
    )
}

pub(crate) fn is_negated_hidden_fallback_statement(lower: &str) -> bool {
    lower.contains("no cpu fallback")
        || lower.contains("no hidden fallback")
        || lower.contains("no software fallback")
        || lower.contains("never hides")
        || lower.contains("must not hide")
}

pub(crate) fn line_cfg_not_gpu_hides_work(lower: &str) -> bool {
    lower.contains("fallback")
        || lower.contains("skip")
        || lower.contains("return ok")
        || lower.contains("success")
}

/// The workspace commands a reader is told to run through the wrapper.
pub(crate) const RAW_CARGO_COMMANDS: [&str; 14] = [
    "cargo build",
    "cargo check",
    "cargo test",
    "cargo clippy",
    "cargo doc",
    "cargo fmt",
    "cargo run",
    "cargo xtask",
    "cargo bench",
    "cargo publish",
    "cargo machete",
    "cargo udeps",
    "cargo fuzz",
    "cargo public-api",
];

/// Whether a comment tells a reader to run the command it names.
///
/// A comment that says what cargo does with a member, or which build a rule is
/// about, is a description: the sentence is true and there is nothing to fix in
/// it. A comment that tells a maintainer to run something is an instruction,
/// and an instruction in this workspace names the wrapper.
///
/// Two signals have to agree. The verb comes before the command, because the
/// command itself contains the word run and matching the whole line read every
/// sentence that mentioned `cargo run` as an order to run it. And the command
/// is delimited as code, because prose says a full cargo build while an
/// instruction quotes what to type: a first attempt on the verb alone read
/// `the gates that run a full cargo build` as an order.
pub(crate) fn comment_instructs_a_run(before_command: &str) -> bool {
    let quoted_as_code = before_command.ends_with('`')
        || before_command.ends_with('"')
        || before_command.ends_with("`./")
        || before_command.ends_with("\"./");
    if !quoted_as_code {
        return false;
    }
    let lower = before_command.to_ascii_lowercase();
    [
        "run ",
        "runs ",
        "running ",
        "invoke",
        "rebuild",
        "regenerate",
        "reproduce",
        "re-run",
        "rerun",
        "reverify",
        "re-verify",
        "via ",
        "with ",
    ]
    .iter()
    .any(|verb| lower.contains(verb))
}

/// Whether a line is a comment rather than code or an emitted string.
pub(crate) fn is_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("//") || trimmed.starts_with("* ") || trimmed == "*"
}

/// A line with every `cargo +<toolchain>` selector reduced to plain `cargo`.
///
/// The selector picks a compiler, not a different program: `cargo +nightly
/// install` installs a tool exactly as `cargo install` does, and `cargo
/// +stable build` is the workspace build the wrapper owns. Reading the
/// selector as part of the command name left the pinned-toolchain form
/// outside every exemption and inside a blanket `cargo +` fallback, so the
/// one line that installs a gate's own dependency was reported as a release
/// blocker while `cargo +stable build` was caught only by accident.
fn without_toolchain_selector(line: &str) -> Cow<'_, str> {
    if !line.contains("cargo +") {
        return Cow::Borrowed(line);
    }
    let mut normalized = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("cargo +") {
        let after = &rest[start + "cargo +".len()..];
        let Some(space) = after.find(' ') else {
            break;
        };
        normalized.push_str(&rest[..start]);
        normalized.push_str("cargo ");
        rest = &after[space + 1..];
    }
    normalized.push_str(rest);
    Cow::Owned(normalized)
}

pub(crate) fn line_contains_raw_workspace_cargo(line: &str) -> bool {
    let normalized = without_toolchain_selector(line.trim());
    let trimmed = normalized.trim();
    if trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("name:")
        || is_release_rule_text(trimmed)
        || trimmed.starts_with("echo ")
        || trimmed.contains("cargo install")
        || trimmed.contains("cargo_full")
        || trimmed.contains("CARGO_RUNNER")
        || trimmed.contains("./cargo_full")
        || trimmed.contains("VYRE_CARGO_RUNNER")
    {
        return false;
    }
    let Some(offset) = RAW_CARGO_COMMANDS
        .iter()
        .filter_map(|needle| trimmed.find(needle))
        .min()
    else {
        return trimmed.starts_with("cargo +");
    };
    if is_emitted_sentence(trimmed, offset) {
        return comment_instructs_a_run(&trimmed[..offset]);
    }
    !is_comment_line(trimmed) || comment_instructs_a_run(&trimmed[..offset])
}

/// Whether the command at `offset` is inside a sentence the code emits.
///
/// A gate that spawns cargo through the one resolver still has to say which
/// command failed, and that sentence names `cargo test`. Reading it as an
/// invocation reported three gates that call the wrapper correctly. An emitted
/// sentence is then judged by [`comment_instructs_a_run`], the same question
/// asked of a comment: a message telling a maintainer to run something must
/// name the wrapper, and one saying what failed is a description.
///
/// A spawn names its program to `Command::new`, so a line that does stays
/// reported however it is quoted, and text printed for a reader to copy is not
/// one of the message shapes.
pub(crate) fn is_emitted_sentence(trimmed: &str, offset: usize) -> bool {
    if trimmed.contains("Command::new") {
        return false;
    }
    let bytes = trimmed.as_bytes();
    let mut quotes = 0usize;
    let mut at = 0usize;
    while at < offset {
        match bytes[at] {
            b'\\' => at += 1,
            b'"' => quotes += 1,
            _ => {}
        }
        at += 1;
    }
    if quotes % 2 == 0 {
        return false;
    }
    let before = &trimmed[..offset];
    [
        "format!(",
        "panic!(",
        "expect(",
        "unwrap_or_else(",
        "GateError::new(",
        "assert",
    ]
    .iter()
    .any(|shape| before.contains(shape))
}

pub(crate) fn line_contains_invalid_cargo_full_xtask(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || is_release_rule_text(trimmed) {
        return false;
    }
    let plain = ["cargo_full", " xtask"].concat();
    let dotted = ["./cargo_full", " xtask"].concat();
    trimmed.contains(&plain) || trimmed.contains(&dotted)
}

pub(crate) fn line_contains_heredoc(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return false;
    }
    trimmed.contains("<<") && !trimmed.contains("<<<")
}

/// Whether a line is a data row spelling text rather than a marker using it.
///
/// A row that spells what a rule forbids is the rule's own vocabulary or a
/// scanner corpus entry. The byte-string form is that same row with a `b`
/// prefix, and a marker comment never takes either shape.
pub(crate) fn is_release_rule_text(trimmed: &str) -> bool {
    trimmed.starts_with('"')
        || trimmed.starts_with("b\"")
        || trimmed.starts_with("(\"")
        || trimmed.starts_with("&[")
        || trimmed.contains("no-stubs")
        || trimmed.contains("unresolved marker")
        || trimmed.contains("No shipped stubs")
}

/// The files that own a hygiene rule and therefore spell what it forbids.
///
/// Two rows named generator scripts that were deleted with the ticket tree they
/// belonged to, so the list exempted files that do not exist. The test below
/// reads this array and requires every row to resolve, because an exemption that
/// names nothing reads as a decision while doing nothing.
pub(crate) const HYGIENE_RULE_SOURCES: [&str; 6] = [
    "xtask/src/gates/lint_hygiene.rs",
    "xtask/src/release/feature_matrix.rs",
    "xtask/src/gates/hygiene_matrix/mod.rs",
    "xtask-evidence/src/release/backend_matrix.rs",
    "xtask-evidence/src/release/vyre_release_gate/mod.rs",
    "xtask-registry/src/release/optimization_matrix.rs",
];

/// The files that own the hidden-fallback rule and spell the prose it catches.
pub(crate) const HIDDEN_FALLBACK_GUARD_SOURCES: [&str; 7] = [
    "xtask/src/gates/gpu_loudness.rs",
    "vyre-lints/src/production_cpu_fallbacks.rs",
    "vyre-lints/src/gpu_skip_guards.rs",
    "vyre-lints/src/lib.rs",
    "vyre-lints/src/main.rs",
    "vyre-lints/tests/production_cpu_fallbacks.rs",
    "vyre-lints/tests/gpu_skip_guards.rs",
];

pub(crate) fn is_hygiene_rule_source(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    HYGIENE_RULE_SOURCES
        .iter()
        .any(|suffix| normalized.ends_with(suffix))
}

pub(crate) fn is_hidden_fallback_guard_source(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    HIDDEN_FALLBACK_GUARD_SOURCES
        .iter()
        .any(|suffix| normalized.ends_with(suffix))
}

pub(crate) fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(index, _)| {
        is_word_start(haystack, index) && is_word_end(haystack, index + needle.len())
    })
}

pub(crate) fn is_word_start(text: &str, index: usize) -> bool {
    text.get(..index)
        .and_then(|prefix| prefix.chars().next_back())
        .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
}

pub(crate) fn is_word_end(text: &str, index: usize) -> bool {
    text.get(index..)
        .and_then(|suffix| suffix.chars().next())
        .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
}

#[derive(Default)]
pub(crate) struct BraceDepthState {
    pub(crate) depth: usize,
    pub(crate) block_comment_depth: usize,
    pub(crate) raw_string_hashes: Option<usize>,
}

impl BraceDepthState {
    pub(crate) fn with_depth(depth: usize) -> Self {
        Self {
            depth,
            ..Self::default()
        }
    }

    pub(crate) fn update(&mut self, line: &str) {
        let bytes = line.as_bytes();
        let mut index = 0usize;
        let mut in_string = false;
        let mut in_char = false;
        let mut escaped = false;

        while index < bytes.len() {
            if let Some(hashes) = self.raw_string_hashes {
                if raw_string_end_at(bytes, index, hashes) {
                    self.raw_string_hashes = None;
                    index += hashes + 1;
                } else {
                    index += 1;
                }
                continue;
            }
            if self.block_comment_depth > 0 {
                if bytes[index..].starts_with(b"/*") {
                    self.block_comment_depth = self.block_comment_depth.saturating_add(1);
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    self.block_comment_depth = self.block_comment_depth.saturating_sub(1);
                    index += 2;
                } else {
                    index += 1;
                }
                continue;
            }
            if in_string {
                match bytes[index] {
                    _ if escaped => escaped = false,
                    b'\\' => escaped = true,
                    b'"' => in_string = false,
                    _ => {}
                }
                index += 1;
                continue;
            }
            if in_char {
                match bytes[index] {
                    _ if escaped => escaped = false,
                    b'\\' => escaped = true,
                    b'\'' => in_char = false,
                    _ => {}
                }
                index += 1;
                continue;
            }

            if bytes[index..].starts_with(b"//") {
                break;
            }
            if bytes[index..].starts_with(b"/*") {
                self.block_comment_depth = 1;
                index += 2;
                continue;
            }
            if let Some((hashes, consumed)) = raw_string_start(bytes, index) {
                self.raw_string_hashes = Some(hashes);
                index += consumed;
                continue;
            }

            match bytes[index] {
                b'"' => in_string = true,
                b'\'' if bytes[index + 1..].contains(&b'\'') => in_char = true,
                b'{' => self.depth = self.depth.saturating_add(1),
                b'}' => self.depth = self.depth.saturating_sub(1),
                _ => {}
            }
            index += 1;
        }
    }
}

pub(crate) fn raw_string_start(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    let mut cursor = index;
    if bytes.get(cursor) == Some(&b'b') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hash_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    let hashes = cursor - hash_start;
    Some((hashes, cursor - index + 1))
}

pub(crate) fn raw_string_end_at(bytes: &[u8], index: usize, hashes: usize) -> bool {
    bytes.get(index) == Some(&b'"')
        && bytes
            .get(index + 1..index + 1 + hashes)
            .is_some_and(|suffix| suffix.iter().all(|byte| *byte == b'#'))
}

pub(crate) fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(path, MAX_HYGIENE_SCAN_FILE_BYTES, "hygiene scan")
}

pub(crate) fn update_brace_depth(current: usize, line: &str) -> usize {
    let mut state = BraceDepthState::with_depth(current);
    state.update(line);
    state.depth
}
