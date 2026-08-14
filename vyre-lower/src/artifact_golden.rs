//! Byte-stability harness for emitted backend artifacts.
//!
//! A refactor of an emitter is only safe if the bytes it produces do not move.
//! Proving that needs three things per backend: a fixed set of input programs,
//! a deterministic rendering of the artifact, and a pinned copy to compare
//! against. Only the middle one is backend-specific.
//!
//! This module owns the other two. The input set is
//! [`crate::emit_adversarial_corpus::success_cases`], already the shared
//! program corpus every emitter consumes. The pinned copy is a text file the
//! calling crate keeps beside its test. The caller supplies just a closure that
//! turns one verified descriptor into deterministic text.
//!
//! The rendering is text so a failure is readable as a diff instead of a
//! digest mismatch. Binary artifacts render through [`hex_words`].
//!
//! This module names no target, dialect, driver, or artifact format.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::emit_adversarial_corpus;
use crate::KernelDescriptor;

/// Line that opens each case's section in a rendered corpus.
const CASE_MARKER: &str = "===== ";

/// Header written above a rendered corpus, naming how to regenerate it.
const HEADER: &str = "\
# Emitted-artifact byte-stability golden.
#
# One section per shared success-corpus case, in corpus order. Regenerate with
# the `bless_*` test in the file that reads this golden, then review the diff:
# a change here is a change in what the backend emits.
";

/// Render one `(case id, section text)` sequence into a golden corpus.
///
/// A corpus is a header plus one marked section per case, and a caller that
/// restates that framing is one edit away from a golden the shared comparison
/// cannot locate a differing case in. Callers whose cases are not descriptors
/// go through here; [`render_success_corpus`] is the descriptor-shaped wrapper.
#[must_use]
pub fn render_sections<I, S>(sections: I) -> String
where
    I: IntoIterator<Item = (S, String)>,
    S: AsRef<str>,
{
    let mut out = String::from(HEADER);
    for (id, rendered) in sections {
        let _ = writeln!(out, "{CASE_MARKER}{}", id.as_ref());
        out.push_str(&rendered);
        if !rendered.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// Whether a rendered corpus contains a section for `case_id`.
///
/// A pinned corpus that stopped naming a case silently stopped covering it, and
/// the marker framing that decides the answer is owned here.
#[must_use]
pub fn contains_case(corpus: &str, case_id: &str) -> bool {
    corpus.contains(&format!("{CASE_MARKER}{case_id}\n"))
}

/// Render every shared success-corpus case through `render`, in corpus order.
///
/// Each descriptor is verified first, so the rendered artifact is the one a
/// caller going through [`crate::verify_descriptor`] would get.
///
/// # Panics
///
/// Panics when a corpus success case fails descriptor verification, which
/// means the corpus and the verifier disagree rather than that the backend
/// changed.
#[must_use]
pub fn render_success_corpus(render: impl Fn(&KernelDescriptor) -> String) -> String {
    render_sections(emit_adversarial_corpus::success_cases().iter().map(|case| {
        let descriptor = crate::verify_descriptor(&case.descriptor).unwrap_or_else(|failure| {
            panic!(
                "Fix: success-corpus case `{}` must pass descriptor verification: {failure:?}",
                case.id
            )
        });
        (case.id, render(&descriptor))
    }))
}

/// Render bytes as fixed-width hex words, eight per line.
///
/// Binary artifacts need a text form to diff. Grouping by four bytes keeps a
/// word-oriented artifact aligned to one column per word.
#[must_use]
pub fn hex_words(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 3);
    for (index, chunk) in bytes.chunks(4).enumerate() {
        if index > 0 {
            out.push(if index % 8 == 0 { '\n' } else { ' ' });
        }
        for byte in chunk {
            let _ = write!(out, "{byte:02x}");
        }
    }
    out.push('\n');
    out
}

/// Compare `actual` against the golden text at `path`.
///
/// # Panics
///
/// Panics when the golden is unreadable or when the rendered corpus differs,
/// naming the first case whose section changed.
pub fn assert_matches_golden(path: &Path, actual: &str) {
    let expected = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "Fix: cannot read emitted-artifact golden {}: {error}. Run the bless test in this file to create it.",
            path.display()
        )
    });
    if expected == actual {
        return;
    }
    panic!(
        "Fix: emitted artifact changed for {}. This refactor must not alter emitted bytes.\n\
         First differing case: {}\n\
         Golden: {} lines. Actual: {} lines.\n\
         If the change is intended, run the bless test in this file and review the diff.",
        path.display(),
        first_differing_case(&expected, actual),
        expected.lines().count(),
        actual.lines().count(),
    );
}

/// Write `actual` to `path`, creating parent directories.
///
/// # Panics
///
/// Panics when the golden cannot be written.
pub fn write_golden(path: &Path, actual: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!("Fix: cannot create {}: {error}", parent.display());
        });
    }
    fs::write(path, actual).unwrap_or_else(|error| {
        panic!("Fix: cannot write {}: {error}", path.display());
    });
}

/// Name the first case whose rendered section differs between two corpora.
///
/// Falls back to a line number when the difference is outside any section,
/// which means the header or the case set itself moved.
fn first_differing_case(expected: &str, actual: &str) -> String {
    let mut case = None;
    for (index, (left, right)) in expected.lines().zip(actual.lines()).enumerate() {
        if let Some(id) = left.strip_prefix(CASE_MARKER) {
            case = Some(id);
        }
        if left != right {
            return match case {
                Some(id) => format!("`{id}` (line {})", index + 1),
                None => format!("<before the first case> (line {})", index + 1),
            };
        }
    }
    match expected.lines().count().cmp(&actual.lines().count()) {
        std::cmp::Ordering::Equal => "<none: contents match>".to_string(),
        std::cmp::Ordering::Less => "<extra trailing content>".to_string(),
        std::cmp::Ordering::Greater => "<missing trailing content>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_words_groups_four_bytes_per_word_and_eight_words_per_line() {
        let bytes: Vec<u8> = (0..36u8).collect();
        let rendered = hex_words(&bytes);
        let lines: Vec<&str> = rendered.trim_end().lines().collect();
        assert_eq!(lines.len(), 2, "36 bytes is 9 words, so 8 then 1");
        assert_eq!(lines[0].split(' ').count(), 8);
        assert_eq!(lines[1], "20212223");
        assert!(rendered.ends_with('\n'));
    }

    #[test]
    fn hex_words_renders_a_trailing_partial_word_without_padding() {
        assert_eq!(hex_words(&[0xde, 0xad, 0xbe]), "deadbe\n");
    }

    #[test]
    fn hex_words_on_empty_input_is_just_the_terminator() {
        assert_eq!(hex_words(&[]), "\n");
    }

    #[test]
    fn rendered_corpus_opens_one_section_per_success_case() {
        let rendered = render_success_corpus(|desc| format!("{}\n", desc.body.ops.len()));
        let sections = rendered
            .lines()
            .filter(|line| line.starts_with(CASE_MARKER))
            .count();
        assert_eq!(sections, emit_adversarial_corpus::success_cases().len());
        assert!(sections > 0, "the shared corpus must carry success cases");
    }

    #[test]
    fn rendered_corpus_terminates_a_section_whose_render_omits_a_newline() {
        let rendered = render_success_corpus(|_| "no-trailing-newline".to_string());
        assert!(
            !rendered.contains("no-trailing-newline====="),
            "sections must stay line-separated even when a renderer omits the newline"
        );
    }

    #[test]
    fn first_differing_case_names_the_section_containing_the_change() {
        let expected = "h\n===== a\n1\n===== b\n2\n";
        let actual = "h\n===== a\n1\n===== b\n3\n";
        assert!(first_differing_case(expected, actual).starts_with("`b`"));
    }

    #[test]
    fn first_differing_case_reports_a_change_above_the_first_section() {
        let expected = "h\n===== a\n1\n";
        let actual = "H\n===== a\n1\n";
        assert!(first_differing_case(expected, actual).starts_with("<before the first case>"));
    }

    #[test]
    fn first_differing_case_reports_length_drift_when_every_shared_line_matches() {
        assert_eq!(
            first_differing_case("h\n===== a\n1\n", "h\n===== a\n1\n===== b\n2\n"),
            "<extra trailing content>"
        );
        assert_eq!(
            first_differing_case("h\n===== a\n1\n===== b\n2\n", "h\n===== a\n1\n"),
            "<missing trailing content>"
        );
    }

    #[test]
    fn assert_matches_golden_accepts_an_identical_corpus() {
        let dir = std::env::temp_dir().join("vyre_artifact_golden_match");
        let path = dir.join("golden.txt");
        write_golden(&path, "h\n===== a\n1\n");
        assert_matches_golden(&path, "h\n===== a\n1\n");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "emitted artifact changed")]
    fn assert_matches_golden_rejects_a_changed_corpus() {
        let dir = std::env::temp_dir().join("vyre_artifact_golden_drift");
        let path = dir.join("golden.txt");
        write_golden(&path, "h\n===== a\n1\n");
        let result = std::panic::catch_unwind(|| assert_matches_golden(&path, "h\n===== a\n2\n"));
        let _ = fs::remove_dir_all(&dir);
        std::panic::resume_unwind(result.expect_err("changed corpus must panic"));
    }

    #[test]
    #[should_panic(expected = "cannot read emitted-artifact golden")]
    fn assert_matches_golden_rejects_a_missing_golden() {
        assert_matches_golden(
            Path::new("/nonexistent/vyre-artifact-golden/missing.txt"),
            "anything",
        );
    }
}
