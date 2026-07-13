//! Deprecated compatibility leaf for substring search.

use vyre::ir::Program;

use crate::scan::substring::{substring_search_with_op_id, LEGACY_MATCHING_SUBSTRING_OP_ID};

// The canonical/legacy substring module paths have a single owner:
// `crate::compat_aliases::MATCHING_SUBSTRING_ALIAS.{canonical_path,deprecated_path}`.

/// Build a substring-search Program with the legacy matching op id.
#[must_use]
pub fn substring_search(
    haystack: &str,
    needle: &str,
    matches: &str,
    haystack_len: u32,
    needle_len: u32,
) -> Program {
    substring_search_with_op_id(
        LEGACY_MATCHING_SUBSTRING_OP_ID,
        haystack,
        needle,
        matches,
        haystack_len,
        needle_len,
    )
}
