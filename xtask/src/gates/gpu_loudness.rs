//! A test that needs a device says so by failing.
//!
//! A GPU test that returns early when a probe fails is a smoke alarm wired to
//! nothing. This gate finds the silent-skip shapes and allows one only when a
//! loud abort sits near it, either an acquisition that panics or an assertion
//! that names the corrective action.

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Lines above and below a skip site that may carry its loud abort.
const WINDOW_BEFORE: usize = 10;
const WINDOW_AFTER: usize = 20;

/// Evidence that a nearby path aborts loudly instead of skipping.
const LOUD: &[&str] = &[
    "acquire_or_panic",
    "panic!(\"no adapter",
    "panic!(\"adapter probe",
    "panic!(\"GPU required",
    "panic!(\"gpu required",
    "panic!(\"headless backend",
];

/// Silent-skip sites in GPU tests.
pub struct GpuLoudness;

impl Gate for GpuLoudness {
    fn name(&self) -> &'static str {
        "gpu-loudness"
    }

    fn help(&self) -> &'static str {
        "tests that return early when a device probe fails"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        for path in tree.all_rust() {
            let text = tree.read(&path)?;
            let masked = scan::mask_literals(&text);
            let lines: Vec<&str> = text.lines().collect();
            let masked_lines: Vec<&str> = masked.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                let blanked = masked_lines.get(index).copied().unwrap_or(line);
                for skip in silent_skips(line, blanked) {
                    if loud_within_window(&lines, index) {
                        continue;
                    }
                    report.find(Finding::at(
                        path.clone(),
                        (index + 1) as u32,
                        format!("{skip} skips the test when no device is present"),
                        "acquire the backend through the panicking constructor, or pair the \
                         skip with a test that exercises the same path and aborts loudly; a \
                         probe failure is a configuration failure and must be reported",
                    ));
                }
            }
        }
        Ok(report)
    }
}

/// Which silent-skip shapes a line carries.
///
/// Three of these were unreachable in the shell original because their pattern
/// was malformed and the error was discarded, so no tree ever matched them. They
/// are live here, which is why the pin covers occurrences the shell never saw.
///
/// Two inputs, because the discriminator sits in a different place per shape. A
/// guard is code, and a detector that builds the same guard out of string pieces
/// must not read as one, so those shapes are judged on the masked line. An
/// attribute carries its feature name inside a literal, which masking blanks, so
/// those shapes are judged on the raw line and anchored on the attribute opener:
/// a pattern table row does not begin with `#[`.
fn silent_skips(raw: &str, masked: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    if masked.contains("if ") && masked.contains("is_err()") && masked.contains('{') {
        if masked.contains("return Ok(());") {
            found.push("an is_err guard returning Ok");
        }
        if masked.contains("return;") {
            found.push("an is_err guard returning early");
        }
    }
    if masked.contains("if let Err(")
        && masked.contains('=')
        && masked.contains('{')
        && masked.contains("return")
    {
        found.push("an if-let-Err guard returning early");
    }
    for macro_name in ["println!(\"", "eprintln!(\""] {
        for excuse in ["skipped", "no GPU", "GPU unavailable"] {
            if raw.contains(&format!("{macro_name}{excuse}")) {
                found.push("a printed excuse for not running");
            }
        }
    }
    if raw.trim_start().starts_with("#[") {
        if raw.contains("#[cfg(not(") && raw.contains("gpu") {
            found.push("a cfg that compiles the test out without a device");
        }
        if raw.contains("#[cfg_attr(not(feature = \"gpu\")") && raw.contains("ignore") {
            found.push("a cfg_attr that ignores the test without the gpu feature");
        }
        if raw.contains("#[cfg_attr(not(any(") && raw.contains("gpu") && raw.contains("ignore") {
            found.push("a cfg_attr that ignores the test without any gpu feature");
        }
    }
    if let Some((code, comment)) = raw.split_once("//") {
        if comment.contains("no GPU") {
            if code.contains("return Ok(());") {
                found.push("a device-conditional early Ok");
            } else if code.contains("return;") {
                found.push("a device-conditional early return");
            }
        }
    }
    found
}

/// Whether a loud abort sits in the window around a skip site.
fn loud_within_window(lines: &[&str], index: usize) -> bool {
    let start = index.saturating_sub(WINDOW_BEFORE);
    let end = (index + WINDOW_AFTER + 1).min(lines.len());
    lines[start..end].iter().any(|line| {
        LOUD.iter().any(|needle| line.contains(needle))
            || (line.contains("assert!(\"") && line.contains("Fix:"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Judge one line the way the run loop does: raw beside its masked form.
    fn skips(line: &str) -> Vec<&'static str> {
        let masked = scan::mask_literals(line);
        silent_skips(line, &masked)
    }

    /// WHY: the three cfg shapes are the ones the shell original could never
    /// match. If they stop matching here the gate silently returns to asserting
    /// nothing, which is the defect this port exists to fix. They also prove the
    /// attribute shapes are read raw: masking blanks the feature name they key on.
    #[test]
    fn the_cfg_shapes_are_reachable() {
        assert_eq!(
            skips("#[cfg(not(feature = \"gpu\"))]").len(),
            1,
            "a cfg that compiles a test out without a device is a skip"
        );
        assert_eq!(skips("#[cfg_attr(not(feature = \"gpu\"), ignore)]").len(), 1);
        assert_eq!(
            skips("#[cfg_attr(not(any(feature = \"gpu\", feature = \"cuda\")), ignore)]").len(),
            1
        );
    }

    /// WHY: a detector's own pattern table is source that contains every shape it
    /// looks for. A guard written as code that builds a pattern must not read as
    /// a guard, which is what the mask buys, and an attribute row in a table does
    /// not start with the attribute opener, which is what the anchor buys. The
    /// mask must keep comments, because two of the shapes live in a comment.
    #[test]
    fn a_pattern_table_is_not_a_skip_site() {
        assert!(
            skips("if line.contains(\"if let Err(\") && line.contains(\"return\") { hit(); }")
                .is_empty()
        );
        assert!(skips("if line.contains(\"#[cfg(not(\") && line.contains(\"gpu\") {").is_empty());
        let source = "let x = 1; // no GPU here\n";
        let masked = scan::mask_literals(source);
        assert_eq!(masked.trim_end(), "let x = 1; // no GPU here");
        assert_eq!(masked.len(), source.len());
    }

    /// WHY: the allowance is the whole reason a legitimate probe helper does not
    /// read as a violation, and it is bounded. A loud abort thirty lines below a
    /// skip does not cover it.
    #[test]
    fn the_allowance_window_is_bounded() {
        let mut lines = vec!["if probe().is_err() { return; }"];
        for _ in 0..25 {
            lines.push("    // filler");
        }
        lines.push("    let backend = Backend::acquire_or_panic();");
        assert!(!loud_within_window(&lines, 0));
        let near = vec![
            "let backend = Backend::acquire_or_panic();",
            "if probe().is_err() { return; }",
        ];
        assert!(loud_within_window(&near, 1));
    }

    /// WHY: the comment forms only count when the excuse is in the comment. A
    /// return next to an unrelated comment is ordinary control flow.
    #[test]
    fn a_commented_return_counts_only_on_the_device_excuse() {
        assert_eq!(skips("        return; // no GPU here").len(), 1);
        assert!(skips("        return; // caller owns the retry").is_empty());
    }
}
