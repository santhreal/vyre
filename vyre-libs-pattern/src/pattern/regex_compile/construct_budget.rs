//! State, alternation, and nested-repeat budgets, plus the pre-lowering HIR scan
//! that rejects over-budget constructs with a distinct diagnostic.

use regex_syntax::hir::{Hir, HirKind};

use super::compile_error::{FEATURE_HUGE_ALTERNATION, FEATURE_NESTED_REPEATS};
use super::{RegexCompileError, LANES};

pub(super) const STATE_CAP: usize = LANES * 32;

/// An alternation with more arms than this can NEVER fit the state budget
/// (each arm needs ≥1 state, plus the fork + join), so it is distinctly
/// diagnosed as a huge alternation instead of collapsing into a generic
/// `TooManyStates`. Equal to `STATE_CAP` so the reclassification is SOUND: any
/// alternation this wide already overflowed and never compiled, no successful
/// compile is turned into an error.
pub(super) const MAX_ALTERNATION_ARMS: usize = STATE_CAP;

/// Nested bounded repeats unroll to (product of the bounds) copies of the body,
/// and each copy is ≥1 state, so when that product exceeds this budget the NFA
/// provably cannot fit. Such patterns are distinctly diagnosed as a nested-repeat
/// blowup rather than a generic `TooManyStates`. Equal to `STATE_CAP` (the unroll
/// product lower-bounds the state count), so no currently-compiling nested
/// repeat regresses.
const NESTED_REPEAT_UNROLL_BUDGET: u64 = STATE_CAP as u64;

/// Non-error signals gathered by [`scan_constructs`] while it validates budgets.
pub(super) struct ConstructScan {
    /// A capture group was seen (whole-match compiles; submatch spans are not
    /// proven (a verifier-routed signal, never a compile error)).
    pub(super) captures_present: bool,
}

/// Walk a parsed HIR once to (a) reject over-budget constructs with a DISTINCT
/// diagnostic, huge alternations and nested bounded repeats. BEFORE lowering
/// collapses them into a generic `TooManyStates`, and (b) record whether any
/// capture group is present. Returns the worst-case bounded-repeat unroll
/// product of `hir`, so a parent repetition can detect multiplicative nesting.
pub(super) fn scan_constructs(
    hir: &Hir,
    pid: usize,
    scan: &mut ConstructScan,
) -> Result<u64, RegexCompileError> {
    match hir.kind() {
        HirKind::Alternation(alts) => {
            if alts.len() > MAX_ALTERNATION_ARMS {
                return Err(RegexCompileError::Unsupported {
                    pattern_index: pid,
                    feature: FEATURE_HUGE_ALTERNATION,
                });
            }
            let mut worst = 1u64;
            for a in alts {
                worst = worst.max(scan_constructs(a, pid, scan)?);
            }
            Ok(worst)
        }
        HirKind::Concat(parts) => {
            let mut worst = 1u64;
            for p in parts {
                worst = worst.max(scan_constructs(p, pid, scan)?);
            }
            Ok(worst)
        }
        HirKind::Repetition(rep) => {
            let inner = scan_constructs(&rep.sub, pid, scan)?;
            match rep.max {
                Some(m) => {
                    let product = u64::from(m).saturating_mul(inner.max(1));
                    // `inner > 1` means a bounded repeat is NESTED inside this
                    // bounded repeat (the case that multiplicatively explodes).
                    // A flat `a{5000}` (inner == 1) is left to the per-repeat
                    // `TooManyStates` guard, unchanged.
                    if inner > 1 && product > NESTED_REPEAT_UNROLL_BUDGET {
                        return Err(RegexCompileError::Unsupported {
                            pattern_index: pid,
                            feature: FEATURE_NESTED_REPEATS,
                        });
                    }
                    Ok(product)
                }
                // Unbounded (`*` / `+`) lowers to an O(1) Kleene wrapper: it does
                // not multiply the nesting product.
                None => Ok(inner.max(1)),
            }
        }
        HirKind::Capture(c) => {
            scan.captures_present = true;
            scan_constructs(&c.sub, pid, scan)
        }
        _ => Ok(1),
    }
}

/// Structured scan for a backreference construct (`\1`..`\9`, `\k<name>` /
/// `\k'name'` / `\k{name}`, or `(?P=name)`). `regex-syntax` does not support
/// backreferences at all, they surface as a raw parse error, so this runs
/// ONLY on the parse-failure path, to CLASSIFY the failure as the distinct
/// unsupported construct rather than a generic syntax error. It respects
/// backslash escaping: an escaped backslash (`\\`) consumes both bytes, so the
/// following digit is read as a literal, not a backreference.
pub(super) fn pattern_uses_backreference(pat: &str) -> bool {
    let bytes = pat.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                if let Some(&c) = bytes.get(i + 1) {
                    // Numeric backreference `\1`..`\9` (`\0` is a NUL escape).
                    if c.is_ascii_digit() && c != b'0' {
                        return true;
                    }
                    // Named backreference `\k<name>` / `\k'name'` / `\k{name}`.
                    if c == b'k' && matches!(bytes.get(i + 2), Some(b'<' | b'\'' | b'{')) {
                        return true;
                    }
                }
                // Skip the escape AND the escaped byte so `\\` is not misread.
                i += 2;
            }
            // Python-style named backreference `(?P=name)`. `bytes[i]` is the
            // ASCII `(`, so `i` is a char boundary and the slice is safe.
            b'(' if pat[i..].starts_with("(?P=") => return true,
            _ => i += 1,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::{compile_regex_set, regex_construct_diagnostic_code, RegexConstruct};
    use super::pattern_uses_backreference;

    /// The backreference detector must respect backslash escaping and match every
    /// backreference syntax `regex-syntax` rejects, WITHOUT string-matching parser
    /// error text (a structured source scan. ONE PLACE, no parse-message hacks).
    #[test]
    fn backreference_detector_is_escaping_aware() {
        // Numeric backreferences \1..\9 (in any position).
        assert!(pattern_uses_backreference(r"\1"));
        assert!(pattern_uses_backreference(r"(a)\1"));
        assert!(pattern_uses_backreference(r"foo\9bar"));
        // Named backreferences.
        assert!(pattern_uses_backreference(r"\k<name>"));
        assert!(pattern_uses_backreference(r"\k'name'"));
        assert!(pattern_uses_backreference("(?P=name)"));

        // NOT backreferences: \0 is a NUL escape, an escaped backslash before a
        // digit is a literal backslash + literal digit, and ordinary escapes /
        // classes carry no backreference.
        assert!(!pattern_uses_backreference(r"\0"));
        assert!(
            !pattern_uses_backreference(r"\\1"),
            "an escaped backslash then a literal 1 is not a backreference"
        );
        assert!(!pattern_uses_backreference(r"\d+\w*"));
        assert!(!pattern_uses_backreference(r"[a-z]{3}"));
        assert!(!pattern_uses_backreference("plain text"));
        // `\\\1` = literal backslash, then a real backreference.
        assert!(pattern_uses_backreference(r"\\\1"));
    }

    /// SOUNDNESS / no-regression: the budget reclassification must only relabel
    /// patterns that ALREADY failed (state overflow). Patterns UNDER the budgets
    /// must still compile exactly as before.
    #[test]
    fn budget_reclassification_does_not_regress_compiling_patterns() {
        // A normal multi-arm alternation (well under both the arm budget AND the
        // state cap, each arm is a single byte) still compiles: the arm-count
        // check must not false-fire on ordinary alternations.
        let ok_alt: String = ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("|");
        let compiled = compile_regex_set(&[ok_alt.as_str()])
            .expect("a 62-arm single-byte alternation must still compile");
        // And it must NOT be misclassified as a huge alternation.
        assert!(compiled.plan.num_states > 0);

        // A nested bounded repeat whose unroll product is under the budget still
        // compiles (20*20 = 400 < 1024).
        assert!(
            compile_regex_set(&[r"(?:a{20}){20}"]).is_ok(),
            "a nested repeat under the unroll budget must still compile"
        );

        // The ONE-PLACE construct→code map round-trips every construct.
        assert_eq!(
            regex_construct_diagnostic_code(RegexConstruct::Backreference),
            "VYRE_SCAN_UNSUPPORTED_BACKREFERENCE"
        );
        assert_eq!(
            regex_construct_diagnostic_code(RegexConstruct::NestedRepeats),
            "VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET"
        );
    }
}
