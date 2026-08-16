//! `docs-evidence-linked`: the documentation authority its map claims is published.
//!
//! The map is the only artifact that states which documentation pages the
//! release is certified against, and until this check existed nothing read the
//! paths inside it. The generic markdown check asks whether the document carries
//! the words `Evidence sources:`, which an empty heading satisfies, so the map
//! went on requiring `docs/optimization/README.md` after that page left the
//! tree and the release gate reported a closed requirement.

use std::path::Path;

use super::super::gate_inputs::Requirement;
use super::super::paths::{read_text_bounded, resolve_manifest_path};

/// The evidence document that lists the documentation authority.
const MAP: &str = "evidence/docs/docs-evidence-map.md";

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    if !requirement.evidence.iter().any(|evidence| evidence == MAP) {
        failures.push(format!(
            "requirement `{}` does not cite `{MAP}`, so the documentation authority it claims cannot be read",
            requirement.id
        ));
        return;
    }
    let path = resolve_manifest_path(base_dir, MAP);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read `{MAP}`: {error}",
                requirement.id
            ));
            return;
        }
    };
    let claimed = authority_paths(&text);
    if claimed.is_empty() {
        failures.push(format!(
            "requirement `{}` map `{MAP}` names no documentation authority, so it certifies nothing",
            requirement.id
        ));
        return;
    }
    let Some(root) = base_dir.parent() else {
        failures.push(format!(
            "requirement `{}` cannot resolve `{MAP}` claims: the manifest directory has no parent to read them against",
            requirement.id
        ));
        return;
    };
    for authority in claimed {
        if !root.join(&authority).exists() {
            failures.push(format!(
                "requirement `{}` map `{MAP}` requires documentation authority `{authority}`, which the checkout does not carry",
                requirement.id
            ));
        }
    }
}

/// Every repository path the map lists as documentation authority.
///
/// A list item whose whole body is one code span carrying a slash, which is how
/// the map states a path. The release-contract items below the list are
/// sentences, and reading one as a citation would report a rule as a missing
/// file.
fn authority_paths(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let item = line.trim().strip_prefix("- ")?;
            let span = item.trim().strip_prefix('`')?.strip_suffix('`')?;
            span.contains('/').then(|| span.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! `authority_paths` is crate-private and reads one document's list shape.
    //! An integration test reaches only the whole release gate, which cannot say
    //! whether a prose item was read as a path or a path was skipped.

    use super::authority_paths;

    /// WHY: the map mixes a path list with a rule list under the same bullet
    /// syntax. Reading a rule as a citation reports a sentence as a missing
    /// file; skipping a path leaves the claim unchecked, which is the defect
    /// this check closes.
    #[test]
    fn a_backticked_path_item_is_read_and_a_prose_item_is_not() {
        let text = "# Map\n\n- `docs/DOCS.toml`\n- `docs/optimization/BENCH_TARGETS.toml`\n\n- Every page must have one lifecycle classification.\n- `notapath`\n";
        assert_eq!(
            authority_paths(text),
            vec![
                "docs/DOCS.toml".to_string(),
                "docs/optimization/BENCH_TARGETS.toml".to_string()
            ]
        );
    }
}
