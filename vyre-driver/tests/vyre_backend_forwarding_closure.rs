//! # Why this suite exists
//!
//! `VyreBackend` is 57 methods and all but three carry a default body. A
//! decorator forwards the contract to the backend it wraps, and a method it
//! leaves out of that list does not fail to compile: it silently answers from
//! the trait default. `GridSyncSplitBackend` left out seven, so through the
//! grid-sync registry wrapper a device-buffer-capable backend reported
//! `UnsupportedFeature`, a backend with distributed collectives reported none,
//! and `cooperative_grid_sync_fits` answered `false` for every device.
//!
//! Forwarding now has one owner, `backend::forward`, and this suite is the
//! closure over it. Both halves of the question are derived from this crate's
//! own source at run time rather than from a list somebody remembered to
//! update:
//!
//!   - every method declared on the trait belongs to exactly one of the two
//!     forwarding macros, so a method added to the contract is red until it is
//!     placed;
//!   - every `Program`-carrying method is either specialized by the grid-sync
//!     wrapper or recorded here as deliberately left on a trait default that
//!     routes back through `self`, so a new dispatch entry point is red until
//!     somebody says which it is.
//!
//! What this does NOT catch: a forwarding body that calls the wrong inner
//! method. The signature-generated bodies make that hard rather than impossible,
//! and the behavioral proofs live beside the wrapper in
//! `backend::registry::grid_sync_split`.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// Dispatch-surface methods the grid-sync wrapper deliberately does not
/// specialize, with the reason each is safe to leave alone.
///
/// A trait default is only safe here when it routes back through `self`, so the
/// split decision is taken by one of the wrapper's own overrides rather than
/// bypassed. Both entries below call `self.dispatch_resident_timed`.
const DISPATCH_VIA_SELF_ROUTING_DEFAULT: &[(&str, &str)] = &[
    (
        "dispatch_resident_async",
        "default wraps self.dispatch_resident_timed in ReadyPending",
    ),
    (
        "dispatch_resident_sequence_read_ranges_timed_into",
        "default calls self.dispatch_resident_timed once per step",
    ),
];

fn source(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// Every `fn` name declared at one indentation level inside the first block that
/// starts at `opening`.
fn method_names(text: &str, opening: &str, indent: &str) -> BTreeSet<String> {
    let start = text
        .find(opening)
        .unwrap_or_else(|| panic!("cannot find `{opening}` in the source"));
    let body = &text[start + opening.len()..];
    let mut depth = 1_i32;
    let mut end = body.len();
    for (offset, ch) in body.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = offset;
                    break;
                }
            }
            _ => {}
        }
    }
    let prefix = format!("{indent}fn ");
    body[..end]
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|rest| {
            rest.chars()
                .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                .collect()
        })
        .collect()
}

fn trait_methods() -> BTreeSet<String> {
    method_names(
        &source("src/backend/vyre_backend.rs"),
        "pub trait VyreBackend: private::Sealed + Send + Sync {",
        "    ",
    )
}

fn macro_methods(name: &str) -> BTreeSet<String> {
    method_names(
        &source("src/backend/forward.rs"),
        &format!("macro_rules! {name} {{"),
        "        ",
    )
}

#[test]
fn every_trait_method_belongs_to_exactly_one_forwarding_macro() {
    let declared = trait_methods();
    assert!(
        declared.len() > 40,
        "only {} methods parsed off the trait, so this gate is measuring the parser rather than \
         the contract. Fix: check the trait header this test searches for.",
        declared.len()
    );
    let support = macro_methods("forward_vyre_backend_support");
    let dispatch = macro_methods("forward_vyre_backend_dispatch");

    let both: Vec<&String> = support.intersection(&dispatch).collect();
    assert!(
        both.is_empty(),
        "{both:?} appear in both forwarding macros. A decorator that invokes both would then \
         declare the method twice and fail to compile. Fix: keep the two method sets disjoint."
    );

    let forwarded: BTreeSet<String> = support.union(&dispatch).cloned().collect();
    let unplaced: Vec<&String> = declared.difference(&forwarded).collect();
    assert!(
        unplaced.is_empty(),
        "VyreBackend method(s) {unplaced:?} belong to neither forwarding macro, so every \
         decorator silently answers them from the trait default instead of from the backend it \
         wraps. Fix: add each to backend::forward::forward_vyre_backend_support when it does not \
         inspect a Program, or to forward_vyre_backend_dispatch when it does."
    );

    let stale: Vec<&String> = forwarded.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "forwarding macro(s) still emit {stale:?}, which the trait no longer declares. Fix: \
         delete the stale forwards."
    );
}

#[test]
fn the_grid_sync_wrapper_decides_the_split_on_every_program_carrying_method() {
    let dispatch = macro_methods("forward_vyre_backend_dispatch");
    let specialized = method_names(
        &source("src/backend/registry/grid_sync_split.rs"),
        "impl VyreBackend for GridSyncSplitBackend {",
        "    ",
    );
    let recorded: BTreeSet<String> = DISPATCH_VIA_SELF_ROUTING_DEFAULT
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();

    let unhandled: Vec<&String> = dispatch
        .difference(&specialized)
        .filter(|name| !recorded.contains(*name))
        .collect();
    assert!(
        unhandled.is_empty(),
        "dispatch entry point(s) {unhandled:?} reach the inner backend without consulting the \
         grid-sync split decision, so a program carrying a GridSync barrier is handed to a \
         backend with no native cooperative launch. Fix: specialize each in \
         GridSyncSplitBackend, or add it to DISPATCH_VIA_SELF_ROUTING_DEFAULT once you have \
         checked its trait default routes back through `self`."
    );

    let obsolete: Vec<&(&str, &str)> = DISPATCH_VIA_SELF_ROUTING_DEFAULT
        .iter()
        .filter(|(name, _)| !dispatch.contains(*name) || specialized.contains(*name))
        .collect();
    assert!(
        obsolete.is_empty(),
        "{obsolete:?} are recorded as left on a trait default but are no longer dispatch-surface \
         methods, or are now specialized. Fix: delete the stale entries so the list keeps meaning \
         what it says."
    );
}

#[test]
fn the_grid_sync_wrapper_hand_writes_no_support_surface_method() {
    let support = macro_methods("forward_vyre_backend_support");
    let specialized = method_names(
        &source("src/backend/registry/grid_sync_split.rs"),
        "impl VyreBackend for GridSyncSplitBackend {",
        "    ",
    );

    let drifted: Vec<&String> = support.intersection(&specialized).collect();
    assert!(
        drifted.is_empty(),
        "{drifted:?} are hand-written in the grid-sync wrapper AND emitted by the forwarding \
         macro, so the wrapper carries a second copy that can drift from the owner. Fix: delete \
         the hand-written forward, or move the method to the dispatch macro if the wrapper \
         genuinely has to specialize it."
    );
}
