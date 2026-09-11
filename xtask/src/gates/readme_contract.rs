//! The `readme-contract` gate: the landing page states the release it ships.
//!
//! `release/evidence/docs/vyre-readme-contracts.json` is cited by the
//! `docs-evidence-linked` release requirement, which reads `exists`,
//! `source_bytes`, `missing_tokens`, `example_count` and `blockers` and fails
//! the release when any of them says the published page is wrong. Nothing wrote
//! that artifact. The committed copy recorded 38237 bytes of a README that now
//! holds 2285, and required the token `0.7.2` two releases after the train
//! moved to `0.8.0`, so the requirement was reading a file that described a
//! different page.
//!
//! The version token comes from the release train rather than the token list,
//! so a version bump turns this gate red until the landing page follows. The
//! remaining tokens are declared in `release/release-train.toml`.
//!
//! A blocker here is a relative link the landing page offers and the checkout
//! does not carry. That is the failure a reader meets first and the one no
//! other gate reads, because `docs-references` judges the book pages rather
//! than the root page that sends a reader into them.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::artifact_gate::{settle_inspection, Inspection};
use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::release::release_train;

/// The recorded contract the release requirement reads.
pub const ARTIFACT: &str = "release/evidence/docs/vyre-readme-contracts.json";

/// The published landing page this gate judges.
const README: &str = "README.md";

/// Largest landing page this gate reads.
const MAX_README_BYTES: u64 = 1024 * 1024;

/// What the published landing page carries.
#[derive(Serialize)]
struct ReadmeContract {
    schema_version: u32,
    path: &'static str,
    exists: bool,
    read_error: Option<String>,
    source_bytes: u64,
    required_tokens: Vec<String>,
    missing_tokens: Vec<String>,
    example_count: usize,
    blockers: Vec<String>,
}

/// The landing page carries every release claim the train declares.
pub struct ReadmeContractGate;

impl GateBehavior for ReadmeContractGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut inspection = Inspection::new();
        let path = ctx.root.join(README);
        let required = required_tokens();
        let contract = match read_landing_page(&path) {
            Ok(text) => measure(&ctx.root, &text, required),
            Err(read_error) => ReadmeContract {
                schema_version: 2,
                path: README,
                exists: path.exists(),
                read_error: Some(read_error),
                source_bytes: 0,
                missing_tokens: required.clone(),
                required_tokens: required,
                example_count: 0,
                blockers: Vec::new(),
            },
        };

        if let Some(read_error) = &contract.read_error {
            inspection.find(Finding::in_file(
                PathBuf::from(README),
                format!("`{README}` could not be read: {read_error}"),
                "restore the landing page the release requirement cites",
            ));
        }
        for token in &contract.missing_tokens {
            inspection.find(Finding::in_file(
                PathBuf::from(README),
                format!("`{README}` does not state `{token}`"),
                "state the fact the token names on the landing page, or drop the token from \
                 `required_readme_tokens`",
            ));
        }
        if contract.read_error.is_none() && contract.example_count == 0 {
            inspection.find(Finding::in_file(
                PathBuf::from(README),
                format!("`{README}` shows no example block"),
                "show what the product looks like in a fenced block",
            ));
        }
        for blocker in &contract.blockers {
            inspection.find(Finding::in_file(
                PathBuf::from(README),
                format!("`{README}` links to `{blocker}`, which the checkout does not carry"),
                "point the link at a path this repository ships, or drop it",
            ));
        }

        let judged = contract.required_tokens.len();
        inspection.generates_host_evidence(ARTIFACT, &contract);
        let mut report = settle_inspection(ctx, "readme-contract", inspection);
        report.cover_complete("landing page release claims", judged);
        Ok(report)
    }
}

/// The tokens the landing page must carry.
///
/// The shipped version is prepended rather than declared, so the list never
/// carries a version that a bump leaves behind.
fn required_tokens() -> Vec<String> {
    let mut tokens = vec![release_train::vyre_version().to_string()];
    tokens.extend(
        release_train::required_readme_tokens()
            .into_iter()
            .map(str::to_string),
    );
    tokens
}

/// The landing page text, or why it could not be read.
fn read_landing_page(path: &Path) -> Result<String, String> {
    let length = std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .len();
    if length > MAX_README_BYTES {
        return Err(format!(
            "{length} bytes exceeds the {MAX_README_BYTES} this gate reads"
        ));
    }
    std::fs::read_to_string(path).map_err(|error| error.to_string())
}

/// What `text` carries, against `required`.
fn measure(root: &Path, text: &str, required: Vec<String>) -> ReadmeContract {
    let missing_tokens = required
        .iter()
        .filter(|token| !text.contains(token.as_str()))
        .cloned()
        .collect();
    ReadmeContract {
        schema_version: 2,
        path: README,
        exists: true,
        read_error: None,
        source_bytes: text.len() as u64,
        required_tokens: required,
        missing_tokens,
        example_count: example_count(text),
        blockers: dead_links(root, text),
    }
}

/// Fenced blocks in `text`.
///
/// A fence opens and closes with the same marker, so the block count is half
/// the marker count. An unterminated fence rounds down, which understates the
/// page rather than crediting it with a block a reader never sees.
fn example_count(text: &str) -> usize {
    text.lines()
        .filter(|line| line.trim_start().starts_with("```"))
        .count()
        / 2
}

/// Every relative link target the page offers that the checkout does not hold.
///
/// An absolute URL is somebody else's to serve and a fragment names a heading
/// on the page itself, so neither is a path this repository can be missing.
fn dead_links(root: &Path, text: &str) -> Vec<String> {
    let mut dead = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("](") {
        let after = &rest[open + 2..];
        let Some(close) = after.find(')') else {
            break;
        };
        let target = after[..close].trim();
        rest = &after[close + 1..];
        let target = target.split('#').next().unwrap_or_default();
        if target.is_empty()
            || target.contains("://")
            || target.starts_with('<')
            || target.starts_with('/')
        {
            continue;
        }
        if !root.join(target).exists() && !dead.iter().any(|seen| seen == target) {
            dead.push(target.to_string());
        }
    }
    dead
}

#[cfg(test)]
mod tests {
    //! The measurement is separated from the checkout so each verdict can be
    //! proved against a page this test writes. Asking the live tree whether its
    //! own README is complete answers what the tree happens to hold, and the
    //! tree is green precisely when the failing paths never run.

    use super::{dead_links, example_count, measure};

    /// A page carrying one token, one fenced block and one link.
    fn page() -> &'static str {
        "# vyre\n\nvyre 0.8.0 runs on cuda.\n\n```rust\nlet x = 1;\n```\n\nSee [the book](docs/SUMMARY.md).\n"
    }

    /// WHY: the release requirement fails the release when `missing_tokens` is
    /// non-empty, so a token the page does not state must reach that list. A
    /// contract that reports every token as present is the state the committed
    /// artifact was in while the page it described had been rewritten.
    #[test]
    fn a_token_the_page_does_not_state_is_reported_missing() {
        let root = std::env::temp_dir();
        let contract = measure(&root, page(), vec!["0.8.0".to_string(), "wgpu".to_string()]);
        assert_eq!(contract.missing_tokens, vec!["wgpu".to_string()]);
        assert_eq!(contract.required_tokens.len(), 2);
        assert_eq!(contract.source_bytes, page().len() as u64);
        assert!(contract.exists);
        assert!(contract.read_error.is_none());
    }

    /// WHY: `example_count` is read as a pass/fail by the release requirement,
    /// so counting fence markers instead of blocks would report one unterminated
    /// fence as an example a reader can run.
    #[test]
    fn a_fenced_block_counts_once_and_an_unterminated_fence_counts_zero() {
        assert_eq!(example_count(page()), 1);
        assert_eq!(example_count("```rust\nlet x = 1;\n"), 0);
        assert_eq!(example_count("```\na\n```\n```\nb\n```\n"), 2);
    }

    /// WHY: a landing page is mostly links, and the one failure a reader meets
    /// first is a link that resolves to nothing. An absolute URL and a bare
    /// fragment are not this repository's to carry, so reading either as a path
    /// would report every external link as a defect.
    #[test]
    fn a_missing_relative_link_is_a_blocker_and_a_url_or_fragment_is_not() {
        let root = std::env::temp_dir();
        let text = "[a](docs/does-not-exist.md) [b](https://example.com/x) [c](#heading)\n";
        assert_eq!(dead_links(&root, text), vec!["docs/does-not-exist.md"]);
    }

    /// WHY: the same dead link stated twice is one defect, and a list that
    /// repeats it reports a page as worse than it is and makes the recorded
    /// blocker count depend on how often a page repeats a link.
    #[test]
    fn one_dead_link_stated_twice_is_reported_once() {
        let root = std::env::temp_dir();
        let text = "[a](docs/gone.md) and again [a](docs/gone.md)\n";
        assert_eq!(dead_links(&root, text), vec!["docs/gone.md"]);
    }
}
