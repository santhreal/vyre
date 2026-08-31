//! HIR to fragment lowering: anchors, concatenation, alternation, repetition.

use regex_syntax::hir::{Hir, HirKind, Look, Repetition};

use super::char_class::build_class;
use super::compile_error::FEATURE_LOOKAROUND;
use super::construct_budget::STATE_CAP;
use super::nfa_builder::{ByteSet, Fragment, NfaBuilder, PatternAnchors};
use super::RegexCompileError;

pub(super) fn empty_fragment(b: &mut NfaBuilder) -> Result<Fragment, RegexCompileError> {
    let s = b.fresh_state()?;
    Ok(Fragment {
        start: s,
        end: s,
        match_len: 0,
    })
}

pub(super) fn build_pattern_hir(
    b: &mut NfaBuilder,
    hir: &Hir,
    pid: usize,
) -> Result<(Fragment, PatternAnchors), RegexCompileError> {
    match hir.kind() {
        HirKind::Look(Look::Start) => Ok((
            empty_fragment(b)?,
            PatternAnchors {
                start: true,
                end: false,
            },
        )),
        HirKind::Look(Look::End) => Ok((
            empty_fragment(b)?,
            PatternAnchors {
                start: false,
                end: true,
            },
        )),
        HirKind::Concat(parts) => {
            let mut first = 0usize;
            let mut last = parts.len();
            let mut anchors = PatternAnchors::default();

            if first < last && is_text_start_look(&parts[first]) {
                anchors.start = true;
                first += 1;
            }
            if first < last && is_text_end_look(&parts[last - 1]) {
                anchors.end = true;
                last -= 1;
            }

            Ok((build_hir_slice(b, &parts[first..last], pid)?, anchors))
        }
        _ => Ok((build_hir(b, hir, pid)?, PatternAnchors::default())),
    }
}

fn is_text_start_look(hir: &Hir) -> bool {
    matches!(hir.kind(), HirKind::Look(Look::Start))
}

fn is_text_end_look(hir: &Hir) -> bool {
    matches!(hir.kind(), HirKind::Look(Look::End))
}

pub(super) fn build_hir_slice(
    b: &mut NfaBuilder,
    parts: &[Hir],
    pid: usize,
) -> Result<Fragment, RegexCompileError> {
    let Some(first_part) = parts.first() else {
        return empty_fragment(b);
    };
    let mut acc = build_hir(b, first_part, pid)?;
    for child in &parts[1..] {
        let next = build_hir(b, child, pid)?;
        b.add_epsilon(acc.end, next.start);
        acc = Fragment {
            start: acc.start,
            end: next.end,
            match_len: acc.match_len + next.match_len,
        };
    }
    Ok(acc)
}

pub(super) fn build_hir(
    b: &mut NfaBuilder,
    hir: &Hir,
    pid: usize,
) -> Result<Fragment, RegexCompileError> {
    match hir.kind() {
        HirKind::Empty => empty_fragment(b),
        HirKind::Literal(lit) => {
            // Each literal byte gets its own state.
            let start = b.fresh_state()?;
            let mut prev = start;
            for &byte in lit.0.iter() {
                let next = b.fresh_state()?;
                b.add_byte_transition(prev, ByteSet::from_byte(byte), next);
                prev = next;
            }
            Ok(Fragment {
                start,
                end: prev,
                match_len: lit.0.len(),
            })
        }
        HirKind::Class(cls) => build_class(b, cls, pid),
        HirKind::Repetition(rep) => build_repetition(b, rep, pid),
        HirKind::Concat(parts) => build_hir_slice(b, parts, pid),
        HirKind::Alternation(alts) => {
            // Diamond: shared fork → each branch → shared join.
            let fork = b.fresh_state()?;
            let join = b.fresh_state()?;
            let mut max_len = 0usize;
            for child in alts {
                let frag = build_hir(b, child, pid)?;
                b.add_epsilon(fork, frag.start);
                b.add_epsilon(frag.end, join);
                if frag.match_len > max_len {
                    max_len = frag.match_len;
                }
            }
            Ok(Fragment {
                start: fork,
                end: join,
                match_len: max_len,
            })
        }
        HirKind::Look(_) => Err(RegexCompileError::Unsupported {
            pattern_index: pid,
            feature: FEATURE_LOOKAROUND,
        }),
        HirKind::Capture(c) => {
            // We don't expose capture groups (NFA scan is multimatch,
            // not capture). Strip and recurse.
            build_hir(b, &c.sub, pid)
        }
    }
}

fn build_repetition(
    b: &mut NfaBuilder,
    rep: &Repetition,
    pid: usize,
) -> Result<Fragment, RegexCompileError> {
    let min = rep.min;
    let max = rep.max;

    // Keep pathological repetitions from materializing a giant transient NFA.
    // The final state cap is the source of truth, so oversized repetitions
    // report TooManyStates instead of pretending the syntax is unsupported.
    if let Some(m) = max {
        if m as usize > STATE_CAP {
            return Err(RegexCompileError::TooManyStates {
                states: m as usize,
                cap: STATE_CAP,
            });
        }
    }
    if min as usize > STATE_CAP {
        return Err(RegexCompileError::TooManyStates {
            states: min as usize,
            cap: STATE_CAP,
        });
    }

    // Build by unrolling: emit `min` copies, then either
    //   - a Kleene loop if max is None (`*` / `+`), OR
    //   - `max - min` optional copies if max is bounded.
    let start = b.fresh_state()?;
    let mut tail = start;
    let mut total_len = 0usize;

    for _ in 0..min {
        let frag = build_hir(b, &rep.sub, pid)?;
        b.add_epsilon(tail, frag.start);
        tail = frag.end;
        total_len += frag.match_len;
    }

    match max {
        None => {
            // Open-ended: insert a Kleene wrapper. tail → frag.start →
            // frag.end → tail (loop back) ; tail → join (skip).
            let join = b.fresh_state()?;
            let frag = build_hir(b, &rep.sub, pid)?;
            b.add_epsilon(tail, frag.start);
            b.add_epsilon(frag.end, frag.start); // loop
            b.add_epsilon(frag.end, join);
            b.add_epsilon(tail, join); // zero matches
            tail = join;
        }
        Some(m) => {
            for _ in min..m {
                let frag = build_hir(b, &rep.sub, pid)?;
                let join = b.fresh_state()?;
                b.add_epsilon(tail, frag.start);
                b.add_epsilon(frag.end, join);
                b.add_epsilon(tail, join); // skip this optional copy
                tail = join;
                // `match_len` is the MAXIMUM admissible match length (see
                // `build_class`: extraction uses it only to size the replay
                // window, so over-sizing is harmless but UNDER-sizing truncates
                // the walk before the longer accepts). A bounded repetition
                // `{n,m}` accepts every length in `n..=m` (the ε skip edges make
                // the fragment end reachable after each optional copy), so the
                // window must cover the MAX `m` copies, otherwise the anchored
                // windowed replay caps at `n` and never visits ends `n+1..=m`
                // (the root of BACKLOG items 18/27: `a{2,4}` surfaced only
                // length-2, and `{10,48}` under-scanned). Accumulate every
                // optional copy so `total_len` reaches `m * sub_len`.
                total_len += frag.match_len;
            }
        }
    }
    Ok(Fragment {
        start,
        end: tail,
        match_len: total_len,
    })
}
