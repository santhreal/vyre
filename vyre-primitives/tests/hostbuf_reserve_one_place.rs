//! ONE-PLACE lock for the cleared-output-buffer reservation idiom.
//!
//! The `out.clear(); out.try_reserve(target - out.capacity())` form under-reserves
//! on a warm buffer (after `clear()` the length is 0, so it only guarantees
//! `target - capacity` slots and the fill reallocates when `0 < capacity < target`).
//! It was hand-rolled ~20× across `bitset`/`reduce` CPU-reference helpers and is now
//! owned in one place by `crate::hostbuf::reserve_exact_cleared`.
//!
//! This test fails if a second copy of the anti-pattern reappears anywhere under
//! `src/bitset` or `src/reduce`, and asserts the migrated files reference the owner.

use std::fs;
use std::path::{Path, PathBuf};

/// Files still permitted to carry the raw idiom, with the reason. All were foreign-dirty
/// (uncommitted work by another agent) when the crate-wide unification landed, so touching
/// them would clobber that work: `multi_block_prefix_scan.rs` (2 sites), `wire.rs` (1 site,
/// whose reserve is also far from its clear, needs a closer read on a settled tree), and
/// `state_index_frontier.rs` (1 site). They are tracked as open backlog rows and migrate once
/// those files settle; this allowlist then shrinks to empty. The test stays green either way
/// (subset check).
const DEFERRED: &[&str] = &[
    "multi_block_prefix_scan.rs",
    "wire.rs",
    "state_index_frontier.rs",
];

/// Files that were migrated and MUST now reference the shared owner (positive lock:
/// catches a future edit that reintroduces a hand-rolled reserve while removing the call).
const MIGRATED: &[&str] = &[
    "bitset/binary_word.rs",
    "bitset/four_russians.rs",
    "bitset/frontier.rs",
    "bitset/stochastic_compute.rs",
    "reduce/indexed_move.rs",
    "reduce/histogram.rs",
    "reduce/segment_reduce.rs",
    "reduce/radix_sort.rs",
    "decode/base64.rs",
    "hash/hypervector.rs",
    "hash/sketch.rs",
    "hash/sparse_fft.rs",
    "parsing/bytecode_dispatch_table_pack.rs",
    "parsing/line_splice_classify.rs",
    "parsing/whitespace_classify_word.rs",
];

#[test]
fn cleared_buffer_reservation_has_exactly_one_owner() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    // 1. The owner exists.
    let owner = fs::read_to_string(src.join("hostbuf.rs"))
        .expect("src/hostbuf.rs (the ONE-PLACE owner) must exist");
    assert!(
        owner.contains("pub(crate) fn reserve_exact_cleared"),
        "hostbuf.rs must define the single owner `reserve_exact_cleared`"
    );

    // 2. No stray copy of the anti-pattern ANYWHERE under src/, except the deferred files.
    //    (Whole-crate scope is deliberate: the first sweep only covered the hunted
    //    bitset/reduce surface and missed live copies in decode/hash/parsing, a narrow
    //    scan is how ONE-PLACE erodes.)
    let mut offenders = Vec::new();
    let mut files = Vec::new();
    collect_rust_files(&src, &mut files);
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // hostbuf.rs is the OWNER: its doc comment shows the anti-pattern as an
        // illustrative `ignore` example, so it legitimately contains the text.
        if name == "hostbuf.rs" {
            continue;
        }
        let text = fs::read_to_string(path).expect("primitive source must be readable");
        if contains_underreserve_idiom(&text) && !DEFERRED.contains(&name.as_str()) {
            offenders.push(
                path.strip_prefix(&src)
                    .unwrap_or(path)
                    .display()
                    .to_string(),
            );
        }
    }
    assert!(
        offenders.is_empty(),
        "the `try_reserve(target - x.capacity())` under-reservation idiom reappeared, route it \
         through crate::hostbuf::reserve_exact_cleared instead. Offending files:\n{}",
        offenders.join("\n")
    );

    // 3. Every migrated file references the owner (guards against a regression that removes the
    //    call and re-inlines a reserve without tripping the anti-pattern scan).
    for rel in MIGRATED {
        let text = fs::read_to_string(src.join(rel))
            .unwrap_or_else(|_| panic!("migrated file {rel} must exist"));
        assert!(
            text.contains("reserve_exact_cleared"),
            "{rel} was migrated to the shared owner but no longer references reserve_exact_cleared"
        );
    }
}

/// Detect `try_reserve[_exact](<expr> - <expr>.capacity())`: after stripping all
/// whitespace, any `try_reserve` call whose argument list contains `.capacity()`
/// is the under-reservation form (the correct owner call passes a bare `target`).
fn contains_underreserve_idiom(text: &str) -> bool {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let needle = "try_reserve";
    let mut from = 0;
    while let Some(rel) = compact[from..].find(needle) {
        let start = from + rel;
        // The call + its arguments fit comfortably in this window for every site here.
        let end = (start + 80).min(compact.len());
        if compact[start..end].contains(".capacity()") {
            return true;
        }
        from = start + needle.len();
    }
    false
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
