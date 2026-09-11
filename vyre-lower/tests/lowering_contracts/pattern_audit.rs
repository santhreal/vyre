//! Shared pattern-audit reporting contracts.

use std::fmt;

use vyre_lower::pattern_audit::PatternAudit;

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
