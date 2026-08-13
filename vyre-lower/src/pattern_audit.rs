//! Backend-neutral scaffolding for per-target pattern audit reports.
//!
//! Every emitter ships a bundle of read-only pattern analyses and one report
//! that combines them. The interesting part of such a report is its fields.
//! Everything around them is the same everywhere: total the findings, answer
//! whether any fired, render one log line, and print that line through
//! `Display`. Written per emitter, that scaffolding is the single largest
//! block of text the emitter crates share.
//!
//! [`PatternAudit`] owns it. An implementor supplies what only it knows: the
//! kernel id, the finding total, a tag naming the target lane, and the
//! per-pattern breakdown. The line shape, the empty-id substitution, and the
//! derived predicates come from here.
//!
//! This module names no target, dialect, driver, or artifact format. The tag
//! is written by the implementor, in the crate that owns that vocabulary.

use std::fmt;

/// Stand-in for a descriptor that carries no kernel id.
pub const UNNAMED_KERNEL: &str = "<unnamed>";

/// Kernel identifier for display, substituting [`UNNAMED_KERNEL`] when empty.
#[must_use]
pub fn display_kernel_id(kernel_id: &str) -> &str {
    if kernel_id.is_empty() {
        UNNAMED_KERNEL
    } else {
        kernel_id
    }
}

/// Shared shape of a combined pattern-audit report.
///
/// The rendered line is `"<id> (<tag>): <count> <noun> (<breakdown>)"`.
pub trait PatternAudit {
    /// Word for what the report counts, such as `"candidates"`.
    const FINDING_NOUN: &'static str;

    /// Descriptor id the report was produced from; may be empty.
    fn kernel_id(&self) -> &str;

    /// Total actionable findings across every pattern in the report.
    fn finding_count(&self) -> usize;

    /// Write the tag naming the target lane this report belongs to,
    /// including any target revision worth showing in a log line.
    ///
    /// # Errors
    ///
    /// Propagates the sink's write failure.
    fn write_target_tag(&self, out: &mut dyn fmt::Write) -> fmt::Result;

    /// Write the per-pattern breakdown rendered inside the trailing
    /// parentheses.
    ///
    /// # Errors
    ///
    /// Propagates the sink's write failure.
    fn write_breakdown(&self, out: &mut dyn fmt::Write) -> fmt::Result;

    /// Whether any pattern fired.
    fn has_any(&self) -> bool {
        self.finding_count() > 0
    }

    /// Whether the kernel is free of findings from this report's patterns.
    fn is_clean(&self) -> bool {
        !self.has_any()
    }

    /// Write the one-line summary without an intermediate allocation.
    ///
    /// # Errors
    ///
    /// Propagates the sink's write failure.
    fn write_short(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        write!(out, "{} (", display_kernel_id(self.kernel_id()))?;
        self.write_target_tag(out)?;
        write!(out, "): {} {} (", self.finding_count(), Self::FINDING_NOUN)?;
        self.write_breakdown(out)?;
        out.write_char(')')
    }

    /// One-line human-readable summary suitable for log lines.
    fn format_short(&self) -> String {
        let mut out = String::new();
        // Writing into a String is infallible.
        let _ = self.write_short(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Report {
        kernel_id: String,
        revision: u32,
        left: usize,
        right: usize,
    }

    impl PatternAudit for Report {
        const FINDING_NOUN: &'static str = "candidates";

        fn kernel_id(&self) -> &str {
            &self.kernel_id
        }

        fn finding_count(&self) -> usize {
            self.left + self.right
        }

        fn write_target_tag(&self, out: &mut dyn fmt::Write) -> fmt::Result {
            write!(out, "lane r{}", self.revision)
        }

        fn write_breakdown(&self, out: &mut dyn fmt::Write) -> fmt::Result {
            write!(out, "{}l, {}r", self.left, self.right)
        }
    }

    fn report(kernel_id: &str, left: usize, right: usize) -> Report {
        Report {
            kernel_id: kernel_id.to_owned(),
            revision: 3,
            left,
            right,
        }
    }

    #[test]
    fn short_line_follows_the_shared_template() {
        assert_eq!(
            report("k", 2, 1).format_short(),
            "k (lane r3): 3 candidates (2l, 1r)"
        );
    }

    #[test]
    fn an_empty_kernel_id_renders_as_the_placeholder() {
        assert_eq!(
            report("", 0, 0).format_short(),
            "<unnamed> (lane r3): 0 candidates (0l, 0r)"
        );
    }

    #[test]
    fn clean_is_the_negation_of_any_finding() {
        assert!(report("k", 0, 0).is_clean());
        assert!(!report("k", 0, 0).has_any());
        assert!(!report("k", 0, 1).is_clean());
        assert!(report("k", 1, 0).has_any());
    }

    #[test]
    fn writing_short_matches_formatting_short() {
        let r = report("k", 4, 5);
        let mut out = String::new();
        r.write_short(&mut out).expect("String writes never fail");
        assert_eq!(out, r.format_short());
    }
}
