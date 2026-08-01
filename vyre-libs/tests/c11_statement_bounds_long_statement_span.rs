//! Regression witness suite for silent statement truncation in
//! `c11_statement_bounds`.
//!
//! This suite locks out the following defect. It is not a description of
//! current behaviour, it is the fence around behaviour that must never come
//! back.
//!
//! # The defect this suite locks out
//!
//! `c11_statement_bounds` used to scan for a statement terminator in a FIXED
//! 256 token window off each candidate's own start index, and to record
//! whatever it had when that window ran out, with no signal that the scan was
//! truncated.
//!
//! Mechanism. The builder bound `t` to `Expr::InvocationId { axis: 0 }` and ran
//! `loop_for("stmt_scan", t, min(t + 256, buf_len(tok_types)))`, so candidate
//! `t` inspected at most 256 tokens starting at its own index. Inside that loop
//! `found_boundary` latched the FIRST semicolon or top level brace, which is the
//! correct "first boundary wins" behaviour and was never the broken part. The
//! problem was what happened when the window contained no boundary at all:
//!
//! 1. `found_boundary` was seeded to `0` and `stmt_bound_end` to `t + 1`.
//! 2. Nothing in the window set either, because there was no boundary to find.
//! 3. Execution fell straight through to the unconditional append. The builder
//!    allocated a slot with `atomic_add(out_counts, 0, 2)` and stored `t` and
//!    `stmt_bound_end` with NO test of `found_boundary`.
//!
//! So a statement longer than 256 tokens was recorded as the span `[t, t + 1)`,
//! a single token statement, and `found_boundary` was dead by the time it could
//! have prevented that. The recorded end was not merely approximate, it was a
//! value that a genuine one token statement also produces, so a consumer could
//! not tell a truncated scan from a real short statement. The defect was
//! reachable from the shipping C frontend, through
//! `vyre-frontend-c/src/pipeline/statement_bounds.rs`, so a real C source file
//! with a long initializer or a wide expression parsed wrong with no error
//! anywhere.
//!
//! This was NOT the shared flag class from santhreal/vyre#2. There was no
//! shared flag, no collective early exit and no barrier. It was a fixed scan
//! window against an unbounded token count.
//!
//! # The contract the suite now pins
//!
//! There is no fixed scan window. Statement length is bounded only by the token
//! count, so `OLD_SCAN_WINDOW_TOKENS` below survives purely as a fixture
//! coordinate: the regression cases sit exactly on the old cliff edge, where a
//! partial fix is most likely to leave a seam.
//!
//! * One candidate span per token position `t` in `[0, n)`, appended in atomic
//!   order. `out_counts[0]` is a WORD count and equals `2 * n`, so a candidate
//!   is emitted for every position rather than one per statement.
//! * For position `t`, let `j` be the smallest index `>= t` that is a statement
//!   boundary. The span is `(t, j + 1)`.
//! * NO TERMINATOR CONVENTION. When no boundary exists at or after `t`, the span
//!   is `(t, t)`, an EMPTY span where `end == start`. This is unambiguous: a
//!   terminated statement always includes its own terminator, so its `end` is at
//!   least `start + 1`. `end == start` therefore never occurs for a genuine
//!   statement, and is the operator visible truncation signal that the old
//!   `t + 1` collapse could never provide.
//! * A boundary at index `j` is `tok[j] == TOK_SEMICOLON`, or `tok[j]` is
//!   `TOK_LBRACE`/`TOK_RBRACE` while the ABSOLUTE paren depth and ABSOLUTE
//!   bracket depth are both zero. Absolute means measured from token 0, not from
//!   `t`, with a clamped decrement: `(` and `[` increment their counter, `)` and
//!   `]` decrement only when the counter is already above zero, so a stream with
//!   unmatched closers cannot drive a depth negative and cannot mask a later
//!   top level brace. Braces themselves change neither counter.
//!
//! Do NOT weaken, ignore, delete, or invert these assertions to make the suite
//! green. They are the specification the fix was written against.
//!
//! Two of the cases below are load bearing controls that discriminate between
//! the two candidate causes, so read them before touching the scan logic:
//!   `statement_shorter_than_scan_window_ends_at_its_semicolon`
//!   `statement_terminated_on_last_scanned_token_is_bounded_correctly`
//! The second places the semicolon on index 255, the very last token the old
//! window inspected, and it passed even while the defect was live. Together they
//! prove the scan was correct INSIDE its window and that there was no off by one
//! at the window edge. The bug was the fallthrough when no boundary was found at
//! all, not the search.

#![cfg(feature = "c-parser")]

use vyre_libs::parsing::c::lex::tokens::{
    TOK_IDENTIFIER, TOK_LBRACE, TOK_LBRACKET, TOK_LPAREN, TOK_PLUS, TOK_RBRACE, TOK_RBRACKET,
    TOK_RPAREN, TOK_SEMICOLON,
};
use vyre_libs::parsing::c::parse::structure_statement::{
    c11_statement_bounds, c11_statement_bounds_scratch_words,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

use vyre_foundation::ir::Expr;

/// HISTORICAL scan window the truncation defect was pinned to, mirrored here so
/// the regression fixtures still sit exactly on the old cliff edge.
///
/// The builder no longer clamps the scan to any window, so this constant does
/// not describe live behaviour. It is retained as a fixture coordinate: 255,
/// 256 and 257 are where a partial fix is most likely to leave a seam, and a
/// test named after the old window is easier to trace back to the defect than a
/// bare literal.
const OLD_SCAN_WINDOW_TOKENS: u32 = 256;

/// Build `n` tokens that are never a statement boundary at any depth.
///
/// Alternating identifier and plus keeps every token non terminating and leaves
/// both the paren and the bracket counter untouched, so filler can be spliced
/// anywhere in a fixture without moving a boundary.
fn filler(n: u32) -> Vec<u32> {
    (0..n)
        .map(|i| if i % 2 == 0 { TOK_IDENTIFIER } else { TOK_PLUS })
        .collect()
}

/// Build `n` non boundary tokens followed by a single semicolon.
///
/// Alternating identifier and plus keeps every token at top level paren and
/// bracket depth, so the ONLY boundary in the stream is the trailing semicolon.
/// That isolates the scan window: nothing else can terminate the statement.
fn long_statement(n: u32) -> Vec<u32> {
    let mut tokens = filler(n);
    tokens.push(TOK_SEMICOLON);
    tokens
}

/// Run the builder under the CPU reference oracle.
///
/// Returns the `(start, end)` span pairs the kernel appended, in slot order.
fn spans(tokens: &[u32]) -> Vec<(u32, u32)> {
    let token_count = u32::try_from(tokens.len()).expect("token count must fit u32");
    let program = c11_statement_bounds(
        "tok_types",
        Expr::u32(token_count),
        "out_statements",
        "out_counts",
    );
    let scratch_words = c11_statement_bounds_scratch_words(token_count) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(tokens)),
            Value::from(vec![0_u8; tokens.len() * 2 * std::mem::size_of::<u32>()]),
            Value::from(vec![0_u8; std::mem::size_of::<u32>()]),
            Value::from(vec![0_u8; scratch_words * std::mem::size_of::<u32>()]),
        ],
    )
    .expect("reference evaluation of c11_statement_bounds must succeed");

    let statements = decode_u32_le_bytes_all(&outputs[0].to_bytes());
    let counts = decode_u32_le_bytes_all(&outputs[1].to_bytes());
    let used = counts[0] as usize;
    statements[..used]
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

/// Return the span recorded for the candidate that starts at token 0.
fn span_from_zero(tokens: &[u32]) -> (u32, u32) {
    *spans(tokens)
        .iter()
        .find(|(start, _)| *start == 0)
        .expect("a candidate span starting at token 0 must be recorded")
}

/// Return the span recorded for the candidate that starts at token `start`.
fn span_from(tokens: &[u32], start: u32) -> (u32, u32) {
    *spans(tokens)
        .iter()
        .find(|(candidate, _)| *candidate == start)
        .unwrap_or_else(|| panic!("a candidate span starting at token {start} must be recorded"))
}

/// Sort span pairs so the kernel's atomic append order cannot affect a set
/// comparison against the oracle.
fn sorted_spans(mut recorded: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    recorded.sort_unstable();
    recorded
}

/// Pure Rust restatement of the statement bound contract, used as a
/// differential oracle.
///
/// Deliberately written straight from the specification rather than from the IR
/// builder: absolute clamped paren and bracket depth swept from token 0, the
/// first boundary at or after each position, and the `end == start` empty span
/// when no boundary exists. If the kernel and this function disagree, one of
/// them is wrong and the disagreement is the finding.
fn oracle_spans(tokens: &[u32]) -> Vec<(u32, u32)> {
    let mut is_boundary = vec![false; tokens.len()];
    let mut paren_depth: u32 = 0;
    let mut bracket_depth: u32 = 0;
    for (index, &token) in tokens.iter().enumerate() {
        match token {
            TOK_LPAREN => paren_depth += 1,
            TOK_RPAREN => paren_depth = paren_depth.saturating_sub(1),
            TOK_LBRACKET => bracket_depth += 1,
            TOK_RBRACKET => bracket_depth = bracket_depth.saturating_sub(1),
            _ => {}
        }
        let at_top_level = paren_depth == 0 && bracket_depth == 0;
        is_boundary[index] = token == TOK_SEMICOLON
            || (at_top_level && (token == TOK_LBRACE || token == TOK_RBRACE));
    }

    // Suffix sweep: `next_boundary[t]` is the smallest boundary index `>= t`.
    let mut next_boundary: Vec<Option<usize>> = vec![None; tokens.len()];
    let mut nearest: Option<usize> = None;
    for index in (0..tokens.len()).rev() {
        if is_boundary[index] {
            nearest = Some(index);
        }
        next_boundary[index] = nearest;
    }

    (0..tokens.len())
        .map(|start| {
            let start_u32 = u32::try_from(start).expect("token index must fit u32");
            match next_boundary[start] {
                Some(boundary) => (
                    start_u32,
                    u32::try_from(boundary + 1).expect("token index must fit u32"),
                ),
                None => (start_u32, start_u32),
            }
        })
        .collect()
}

/// Control: a statement that fits inside the scan window is bounded correctly.
///
/// Bug locked out: a fix that repairs the long statement case by breaking the
/// ordinary one. If this fails, the scan no longer finds a boundary that is
/// plainly inside its window and the whole primitive is broken, not just its
/// truncation behaviour.
#[test]
fn statement_shorter_than_scan_window_ends_at_its_semicolon() {
    let tokens = long_statement(8);
    let (start, end) = span_from_zero(&tokens);
    assert_eq!(start, 0, "candidate must start at token 0");
    assert_eq!(
        end, 9,
        "8 tokens then a semicolon at index 8 must yield end = 9, got {end}"
    );
}

/// Boundary: a statement whose terminator sits on the LAST token the historical
/// window reached is still bounded correctly.
///
/// The old scan covered `[0, 256)` from candidate 0, so index 255 was the final
/// token it inspected. Placing the semicolon exactly there is the off by one
/// case: a scan bound of `t + 255` rather than `t + 256`, or a `<` versus `<=`
/// slip, fails here while leaving both the short and the long cases looking
/// correct.
#[test]
fn statement_terminated_on_last_scanned_token_is_bounded_correctly() {
    let tokens = long_statement(OLD_SCAN_WINDOW_TOKENS - 1);
    assert_eq!(
        tokens[(OLD_SCAN_WINDOW_TOKENS - 1) as usize],
        TOK_SEMICOLON,
        "fixture must place the semicolon on the last token thread 0 scans"
    );
    let (_, end) = span_from_zero(&tokens);
    assert_eq!(
        end,
        OLD_SCAN_WINDOW_TOKENS,
        "a semicolon at index {} must yield end = {OLD_SCAN_WINDOW_TOKENS}, got {end}",
        OLD_SCAN_WINDOW_TOKENS - 1
    );
}

/// The defect: a statement one token too long to fit the historical scan window
/// was recorded as a one token span instead.
///
/// Bug locked out: silent truncation of any C statement longer than the fixed
/// 256 token lookahead. The recorded end collapsed to `t + 1`, which is
/// indistinguishable from a genuine single token statement, so no consumer could
/// detect that the scan ran out. If this regresses, long statements parse as
/// garbage with no error anywhere.
#[test]
fn statement_longer_than_scan_window_is_not_silently_truncated() {
    let tokens = long_statement(OLD_SCAN_WINDOW_TOKENS);
    let semicolon_index = OLD_SCAN_WINDOW_TOKENS;
    assert_eq!(
        tokens[semicolon_index as usize], TOK_SEMICOLON,
        "fixture must place the semicolon one token past the scan window"
    );
    let (_, end) = span_from_zero(&tokens);
    assert_ne!(
        end, 1,
        "end collapsed to 1, the truncation signature: a {OLD_SCAN_WINDOW_TOKENS} token statement \
         was recorded as a single token span"
    );
    assert_eq!(
        end,
        semicolon_index + 1,
        "a semicolon at index {semicolon_index} must yield end = {}, got {end}",
        semicolon_index + 1
    );
}

/// The defect at a realistic size, to show it is not a one token edge case.
///
/// Bug locked out: a fix that special cases the exactly one past the window
/// boundary and still truncates everything beyond it.
#[test]
fn long_statement_well_past_scan_window_is_not_silently_truncated() {
    let tokens = long_statement(OLD_SCAN_WINDOW_TOKENS * 3);
    let semicolon_index = OLD_SCAN_WINDOW_TOKENS * 3;
    let (_, end) = span_from_zero(&tokens);
    assert_eq!(
        end,
        semicolon_index + 1,
        "a semicolon at index {semicolon_index} must yield end = {}, got {end}",
        semicolon_index + 1
    );
}

/// A brace terminates a statement too, and must not be truncated either.
///
/// Bug locked out: repairing only the semicolon path. `is_statement_boundary`
/// is the OR of a semicolon and a top level brace, so both terminators travel
/// the same truncation path and both must be covered.
#[test]
fn long_statement_terminated_by_brace_is_not_silently_truncated() {
    let mut tokens: Vec<u32> = (0..OLD_SCAN_WINDOW_TOKENS + 10)
        .map(|i| if i % 2 == 0 { TOK_IDENTIFIER } else { TOK_PLUS })
        .collect();
    let brace_index = tokens.len() as u32;
    tokens.push(TOK_LBRACE);
    let (_, end) = span_from_zero(&tokens);
    assert_eq!(
        end,
        brace_index + 1,
        "a brace at index {brace_index} must yield end = {}, got {end}",
        brace_index + 1
    );
}

/// Seam check one token past the old cliff: semicolon exactly at index 256.
///
/// Bug locked out: a fix that restores the correct end for statements well past
/// the old window but leaves a one token seam right where the window used to
/// stop. Index 255 is covered by the control above and index 256 is the first
/// index the old scan could not see, so an off by one reintroduced while ripping
/// the window out lands precisely here. If this regresses, a 257 token statement
/// is bounded at 1 or at 256 instead of 257 and every consumer downstream slices
/// the wrong token range with no error.
#[test]
fn semicolon_immediately_past_old_window_edge_is_bounded_at_257() {
    let tokens = long_statement(OLD_SCAN_WINDOW_TOKENS);
    assert_eq!(
        tokens.len(),
        (OLD_SCAN_WINDOW_TOKENS + 1) as usize,
        "fixture must be 256 filler tokens plus the semicolon"
    );
    assert_eq!(
        tokens[OLD_SCAN_WINDOW_TOKENS as usize], TOK_SEMICOLON,
        "fixture must place the semicolon at index {OLD_SCAN_WINDOW_TOKENS}"
    );
    let (start, end) = span_from_zero(&tokens);
    assert_eq!(start, 0, "candidate must start at token 0");
    assert_eq!(
        end, 257,
        "a semicolon at index 256 must yield end = 257, got {end}"
    );
}

/// Seam check two tokens past the old cliff: semicolon exactly at index 257.
///
/// Bug locked out: the same off by one as the index 256 case, plus a fix that
/// gets the candidate at position 0 right while corrupting the candidate that
/// starts ON the terminator. The boundary search is "smallest boundary index
/// `>= t`", so the semicolon is its own boundary and position 257 must yield
/// `(257, 258)`. A fix that searches strictly after `t` would report `(257, 257)`
/// here, silently reclassifying a real one token statement as unterminated.
#[test]
fn semicolon_two_tokens_past_old_window_edge_is_bounded_at_258() {
    let tokens = long_statement(OLD_SCAN_WINDOW_TOKENS + 1);
    assert_eq!(
        tokens.len(),
        (OLD_SCAN_WINDOW_TOKENS + 2) as usize,
        "fixture must be 257 filler tokens plus the semicolon"
    );
    assert_eq!(
        tokens[(OLD_SCAN_WINDOW_TOKENS + 1) as usize],
        TOK_SEMICOLON,
        "fixture must place the semicolon at index {}",
        OLD_SCAN_WINDOW_TOKENS + 1
    );

    let recorded = sorted_spans(spans(&tokens));
    assert_eq!(
        recorded.len(),
        tokens.len(),
        "one candidate span per token position: expected {}, got {}",
        tokens.len(),
        recorded.len()
    );
    assert_eq!(
        recorded[0],
        (0, 258),
        "position 0 must end just past the semicolon at index 257"
    );
    assert_eq!(
        recorded[257],
        (257, 258),
        "position 257 is the semicolon itself and must end at 258"
    );
    assert_eq!(
        recorded,
        oracle_spans(&tokens),
        "the full span set must match the contract oracle exactly"
    );
}

/// The no terminator signal: an unterminated stream yields EMPTY spans.
///
/// Bug locked out: the original truncation signature. When no boundary exists at
/// or after `t` the span must be `(t, t)`, never `(t, t + 1)`. `end == start` is
/// impossible for a genuine statement, because a terminated statement always
/// includes its own terminator, so it is the only value that lets a consumer
/// distinguish "nothing to parse here" from "a one token statement". If this
/// regresses, an unterminated tail is silently reported as a stream of real one
/// token statements, which is exactly the defect that made the old behaviour
/// undetectable.
#[test]
fn unterminated_stream_reports_empty_spans_distinguishable_from_one_token_statement() {
    let tokens = filler(300);
    assert!(
        !tokens
            .iter()
            .any(|&token| token == TOK_SEMICOLON || token == TOK_LBRACE || token == TOK_RBRACE),
        "fixture must contain no boundary token at all"
    );

    let recorded = sorted_spans(spans(&tokens));
    assert_eq!(
        recorded.len(),
        tokens.len(),
        "one candidate span per token position: expected {}, got {}",
        tokens.len(),
        recorded.len()
    );
    for &(start, end) in &recorded {
        assert_eq!(
            end, start,
            "no boundary exists at or after {start}, so the span must be empty, got ({start}, {end})"
        );
    }

    let unterminated_at_zero = span_from_zero(&tokens);
    assert_eq!(
        unterminated_at_zero,
        (0, 0),
        "position 0 of an unterminated stream must be the empty span (0, 0)"
    );

    let single_statement = vec![TOK_SEMICOLON];
    let terminated_at_zero = span_from_zero(&single_statement);
    assert_eq!(
        terminated_at_zero,
        (0, 1),
        "a genuine one token statement must be (0, 1), never (0, 0)"
    );

    assert_ne!(
        unterminated_at_zero, terminated_at_zero,
        "the unterminated signal must be distinguishable from a real one token \
         statement, got {unterminated_at_zero:?} for both"
    );
}

/// Several statements, each longer than the old window, in one stream.
///
/// Bug locked out: a fix that repairs only the first long statement, for example
/// by hoisting the boundary search into a single sweep anchored at token 0 or by
/// reusing state across candidates without resetting it. Each statement here is
/// 300, 400 and 500 tokens, all past the old 256 cliff, and each must be bounded
/// by its OWN semicolon. If this regresses, the first statement looks right
/// while every later long statement is truncated or is stretched to a later
/// terminator, so a multi statement C file parses wrong from the second
/// statement onward.
#[test]
fn consecutive_long_statements_each_end_at_their_own_semicolon() {
    let lengths = [300_u32, 400, 500];
    let mut tokens: Vec<u32> = Vec::new();
    let mut statement_starts: Vec<u32> = Vec::new();
    let mut semicolon_indices: Vec<u32> = Vec::new();
    for &length in &lengths {
        statement_starts.push(u32::try_from(tokens.len()).expect("index must fit u32"));
        tokens.extend(filler(length));
        semicolon_indices.push(u32::try_from(tokens.len()).expect("index must fit u32"));
        tokens.push(TOK_SEMICOLON);
    }

    assert_eq!(
        statement_starts,
        vec![0, 301, 702],
        "fixture layout drifted"
    );
    assert_eq!(
        semicolon_indices,
        vec![300, 701, 1202],
        "fixture semicolon indices drifted"
    );
    assert_eq!(tokens.len(), 1203, "fixture length drifted");

    for (&start, &semicolon) in statement_starts.iter().zip(semicolon_indices.iter()) {
        let (_, end) = span_from(&tokens, start);
        assert_eq!(
            end,
            semicolon + 1,
            "the statement starting at {start} is terminated by the semicolon at \
             {semicolon} and must yield end = {}, got {end}",
            semicolon + 1
        );
    }

    // The interior of a long statement must reach the SAME terminator, not the
    // next one: a per candidate search that runs away past its own boundary is
    // the mirror image of a truncated one.
    let (_, interior_end) = span_from(&tokens, 500);
    assert_eq!(
        interior_end, 702,
        "position 500 sits inside the second statement and must end at its \
         semicolon index 701 plus one, got {interior_end}"
    );
}

/// A statement longer than the AST pipeline's own token scan cap.
///
/// Bug locked out: a fix that swaps the 256 token window for a larger but still
/// fixed window, for example `C11_AST_MAX_TOK_SCAN`. Any residual constant
/// reintroduces the same silent truncation one order of magnitude further out,
/// where it is far harder to notice: the statement here is 65600 tokens, past
/// the 65536 cap, and must still be bounded at its real semicolon rather than
/// collapsed to `t + 1` or clipped to the cap. If this regresses, a generated or
/// machine emitted C file with one very wide initializer parses wrong with no
/// error anywhere.
#[test]
fn statement_longer_than_pipeline_token_scan_cap_is_bounded_at_its_semicolon() {
    let length = 65_600_u32;
    assert!(
        length > vyre_libs::parsing::c::pipeline::stages::C11_AST_MAX_TOK_SCAN,
        "fixture must exceed the pipeline token scan cap of {}",
        vyre_libs::parsing::c::pipeline::stages::C11_AST_MAX_TOK_SCAN
    );
    let tokens = long_statement(length);
    assert_eq!(
        tokens[length as usize], TOK_SEMICOLON,
        "fixture must place the semicolon at index {length}"
    );
    let (_, end) = span_from_zero(&tokens);
    assert_eq!(
        end,
        length + 1,
        "a semicolon at index {length} must yield end = {}, got {end}",
        length + 1
    );
}

/// Differential: the kernel's whole span set must equal the contract oracle on a
/// mixed corpus.
///
/// Bug locked out: any divergence from the specification that the targeted cases
/// above do not name, in particular a boundary rule that measures paren or
/// bracket depth from the candidate `t` instead of from token 0, a decrement
/// that is not clamped so unmatched closers push a counter negative and mask a
/// later top level brace, or a brace treated as a boundary while nested inside
/// parens or brackets. The comparison is over sorted `(start, end)` pairs, so
/// the kernel's atomic append order is free, but every pair must match exactly:
/// counts alone would hide a wrong `end`. If this regresses, statement bounds go
/// subtly wrong on ordinary C, which is the failure mode nobody notices until an
/// AST is already built from the wrong ranges.
#[test]
fn span_set_matches_cpu_oracle_over_mixed_corpus() {
    let mut corpus: Vec<(&str, Vec<u32>)> = Vec::new();

    // Short statements back to back: the ordinary case.
    let mut short_statements = Vec::new();
    for length in [3_u32, 1, 5, 2] {
        short_statements.extend(filler(length));
        short_statements.push(TOK_SEMICOLON);
    }
    corpus.push(("four short statements", short_statements));

    // The old cliff edge and its two neighbours.
    corpus.push((
        "statement of 255 filler tokens",
        long_statement(OLD_SCAN_WINDOW_TOKENS - 1),
    ));
    corpus.push((
        "statement of 256 filler tokens",
        long_statement(OLD_SCAN_WINDOW_TOKENS),
    ));
    corpus.push((
        "statement of 257 filler tokens",
        long_statement(OLD_SCAN_WINDOW_TOKENS + 1),
    ));

    // Brace terminated statement, past the old cliff.
    let mut brace_terminated = filler(OLD_SCAN_WINDOW_TOKENS + 4);
    brace_terminated.push(TOK_LBRACE);
    brace_terminated.extend(filler(3));
    brace_terminated.push(TOK_RBRACE);
    corpus.push(("brace terminated statement", brace_terminated));

    // Balanced parens and brackets: depth returns to zero, so the trailing
    // semicolon is the only boundary.
    let mut balanced = filler(3);
    balanced.push(TOK_LPAREN);
    balanced.extend(filler(4));
    balanced.push(TOK_LBRACKET);
    balanced.extend(filler(2));
    balanced.push(TOK_RBRACKET);
    balanced.extend(filler(1));
    balanced.push(TOK_RPAREN);
    balanced.extend(filler(2));
    balanced.push(TOK_SEMICOLON);
    corpus.push(("balanced parens and brackets", balanced));

    // A brace nested inside parens is NOT a boundary. A candidate whose own
    // paren depth looks like zero, because the opening paren is behind it, still
    // must see absolute depth one here.
    let mut brace_in_parens = filler(2);
    brace_in_parens.push(TOK_LPAREN);
    brace_in_parens.extend(filler(2));
    brace_in_parens.push(TOK_LBRACE);
    brace_in_parens.extend(filler(2));
    brace_in_parens.push(TOK_RBRACE);
    brace_in_parens.extend(filler(1));
    brace_in_parens.push(TOK_RPAREN);
    brace_in_parens.push(TOK_SEMICOLON);
    corpus.push(("brace nested inside parens", brace_in_parens));

    // A brace nested inside brackets is likewise suppressed, then a brace at
    // depth zero right after the closing bracket is a boundary.
    let mut brace_in_brackets = vec![TOK_LBRACKET];
    brace_in_brackets.extend(filler(2));
    brace_in_brackets.push(TOK_LBRACE);
    brace_in_brackets.push(TOK_RBRACKET);
    brace_in_brackets.push(TOK_LBRACE);
    brace_in_brackets.push(TOK_SEMICOLON);
    corpus.push(("brace nested inside brackets", brace_in_brackets));

    // Unmatched closers: the clamped decrement must keep both counters at zero
    // so the later brace is still recognised at top level.
    let mut unmatched_closers = vec![TOK_RPAREN, TOK_RPAREN, TOK_RBRACKET];
    unmatched_closers.extend(filler(3));
    unmatched_closers.push(TOK_LBRACE);
    unmatched_closers.extend(filler(2));
    unmatched_closers.push(TOK_SEMICOLON);
    corpus.push(("unmatched closing parens and brackets", unmatched_closers));

    // An unclosed paren suppresses braces indefinitely, but a semicolon is a
    // boundary at ANY depth.
    let mut unclosed_paren = vec![TOK_LPAREN];
    unclosed_paren.extend(filler(2));
    unclosed_paren.push(TOK_LBRACE);
    unclosed_paren.extend(filler(2));
    unclosed_paren.push(TOK_SEMICOLON);
    unclosed_paren.push(TOK_LBRACE);
    unclosed_paren.extend(filler(2));
    corpus.push(("semicolon inside an unclosed paren", unclosed_paren));

    // A terminated statement followed by an unterminated tail: the head has real
    // spans, the tail is all empty spans, in one stream.
    let mut unterminated_tail = long_statement(10);
    unterminated_tail.extend(filler(20));
    corpus.push(("terminated head then unterminated tail", unterminated_tail));

    assert!(
        corpus.len() >= 8,
        "the corpus must cover at least 8 streams, got {}",
        corpus.len()
    );

    for (name, tokens) in &corpus {
        let expected = oracle_spans(tokens);
        let recorded = sorted_spans(spans(tokens));
        assert_eq!(
            recorded,
            expected,
            "kernel span set diverged from the contract oracle on the {name} stream \
             ({} tokens)",
            tokens.len()
        );
    }
}
