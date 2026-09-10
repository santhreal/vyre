//! Shared test-only harness helpers for the vyre workspace.
//!
//! # The registry/coverage closure gate, one definitional home
//!
//! Every vyre crate that ships `pub fn ... -> Program` builders owes the same
//! contract: each builder is reachable from that crate's `inventory::submit!`
//! registry, or it is pinned by a parity/behavioral test. A builder that is
//! neither still compiles, still appears in the catalogs generated from source,
//! and still diverges from its reference arm with nothing red.
//!
//! [`assert_registry_closure`] is the one enumerator, and
//! [`registry_closure_gate!`] is how a crate declares its gate. The crate's
//! `tests/registry_closure.rs` carries only what is crate-specific, the floor
//! and the waived builders, because the test name, the manifest-directory
//! argument and the call itself are the same in every crate and were being
//! copied verbatim:
//!
//! ```ignore
//! vyre_test_support::registry_closure_gate! {
//!     floor: 4,
//!     waiver: ["uncovered_builder_with_its_reason_above"],
//! }
//! ```
//!
//! The candidate set is derived from the crate's tree on each run rather than
//! listed in the caller, so a builder added tomorrow is judged tomorrow. That
//! derivation's failure mode is finding nothing: zero builders are trivially
//! all covered, so `BUILDER_FLOOR` is what makes a broken scan fail instead of
//! reporting a clean sweep of a nearly empty set.
//!
//! The enumeration is feature-independent: it reads source files as TEXT and
//! never compiles them, so it reports the same builder set whichever features
//! the runner selects.

/// Declare this crate's registry/coverage closure gate.
///
/// `floor` is the minimum builder count the source enumeration must find, and
/// `waiver` lists builders that are knowingly uncovered. Both are the only
/// crate-specific parts of the gate, so they are the only arguments; the test
/// name and the crate directory are derived here. The directory is the run-time
/// checkout root joined to `CARGO_PKG_NAME`, which expands at the call site and
/// so names the crate that declares the gate. A compiled-in manifest directory
/// would name whichever checkout built the binary, and every checkout here
/// shares one target directory.
#[macro_export]
macro_rules! registry_closure_gate {
    (floor: $floor:expr, waiver: [$($waived:expr),* $(,)?] $(,)?) => {
        #[test]
        fn every_program_builder_is_tested_registered_or_explicitly_waived() {
            $crate::assert_registry_closure(
                $crate::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")),
                &[$($waived),*],
                $floor,
            );
        }
    };
}

/// Declare a test-only `Expr::Opaque` payload type.
///
/// An extension payload is six trait methods of which five are the same in
/// every test that needs one: report `Ok(())` from validation, hand back
/// `self` for downcasting, and answer the two identity questions from a
/// literal. Only the kind string, the debug identity, the result type, the
/// CSE answer and the fingerprint byte differ, so those are the arguments.
///
/// A test that needs a payload with reachable structure, a wire body, or a
/// validation failure writes the impl out: this macro is for the inert leaf.
///
/// `ExprNode` and `DataType` are named unqualified, so the caller must have
/// both in scope. `vyre-foundation` implements these traits on its own types
/// from inside itself, where a path through this crate's dependency on it
/// names a different crate instance and does not compile.
#[macro_export]
macro_rules! test_expr_extension {
    (
        $name:ident,
        kind: $kind:expr,
        identity: $identity:expr,
        result_type: $result_type:expr,
        cse_safe: $cse_safe:expr,
        fingerprint: $fingerprint:expr $(,)?
    ) => {
        #[derive(Debug)]
        struct $name;

        impl ExprNode for $name {
            fn extension_kind(&self) -> &'static str {
                $kind
            }
            fn debug_identity(&self) -> &str {
                $identity
            }
            fn result_type(&self) -> Option<DataType> {
                $result_type
            }
            fn cse_safe(&self) -> bool {
                $cse_safe
            }
            fn stable_fingerprint(&self) -> [u8; 32] {
                [$fingerprint; 32]
            }
            fn validate_extension(&self) -> ::core::result::Result<(), ::std::string::String> {
                Ok(())
            }
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
        }
    };
}

/// Declare a test-only `Node::Opaque` payload type.
///
/// The statement form of [`test_expr_extension!`]: a statement extension has
/// no result type and no CSE answer, so only the kind string, the debug
/// identity and the fingerprint byte differ between tests. `NodeExtension` is
/// named unqualified, so the caller must have it in scope.
#[macro_export]
macro_rules! test_node_extension {
    (
        $name:ident,
        kind: $kind:expr,
        identity: $identity:expr,
        fingerprint: $fingerprint:expr $(,)?
    ) => {
        #[derive(Debug)]
        struct $name;

        impl NodeExtension for $name {
            fn extension_kind(&self) -> &'static str {
                $kind
            }
            fn debug_identity(&self) -> &str {
                $identity
            }
            fn stable_fingerprint(&self) -> [u8; 32] {
                [$fingerprint; 32]
            }
            fn validate_extension(&self) -> ::core::result::Result<(), ::std::string::String> {
                Ok(())
            }
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
        }
    };
}

/// Declare a test operation signature over `u32` values.
///
/// A dialect or operation fixture that only needs "some registered operation"
/// states one signature: named `u32` inputs and one named `u32` output, no
/// attributes, no bytes extraction. Only the parameter names differ, so those
/// are the arguments, and the shape stays one value across the crates that
/// register such an operation.
///
/// Expands to a `Signature` expression usable in a `const`, so `Signature` and
/// `TypedParam` must both be in scope at the call site.
#[macro_export]
macro_rules! u32_signature {
    (inputs: [$($input:expr),+ $(,)?], output: $output:expr $(,)?) => {
        Signature {
            inputs: &[$(TypedParam {
                name: $input,
                ty: "u32",
            }),+],
            outputs: &[TypedParam {
                name: $output,
                ty: "u32",
            }],
            attrs: &[],
            bytes_extraction: false,
        }
    };
}

#[cfg(feature = "ir-fixtures")]
pub mod adversarial_generators;
mod registry_closure;
pub use registry_closure::{
    assert_registry_closure, assert_registry_closure_crates, collect_rust_files,
};
#[cfg(feature = "semantic-requests")]
pub mod artifact_fixtures;
#[cfg(feature = "parity-oracles")]
pub mod async_span_parity;
#[cfg(feature = "ir-fixtures")]
pub mod backend_capabilities;
pub mod backend_execution_domain;
pub mod bin_op_variants;
#[cfg(feature = "ir-fixtures")]
pub mod binop_parity;
pub mod case_table;
#[cfg(feature = "ir-fixtures")]
pub mod cast_parity;
#[cfg(feature = "ir-fixtures")]
pub mod collective_programs;
pub mod consumer_boundary;
pub mod data_type_elements;
#[cfg(feature = "ir-fixtures")]
pub mod data_type_variants;
#[cfg(feature = "parity-oracles")]
pub mod differential_matrix;
#[cfg(feature = "ir-fixtures")]
pub mod elementwise_programs;
pub mod exploded_ifds_cases;
#[cfg(feature = "ir-fixtures")]
pub mod expr_variants;
#[cfg(feature = "ir-fixtures")]
pub mod extension_variants;
pub mod fixed_point;
#[cfg(feature = "driver-artifact-contracts")]
pub mod fixture_instance;
#[cfg(feature = "ir-fixtures")]
pub mod graph_shapes;
/// Two-node and multi-arm program graph shapes planning suites compile.
#[cfg(feature = "ir-fixtures")]
pub mod graph_fixtures;
#[cfg(feature = "ir-fixtures")]
pub mod graph_values;
#[cfg(feature = "parity-oracles")]
pub mod hardware_oracle;
#[cfg(feature = "ir-fixtures")]
pub mod ir_regions;
#[cfg(feature = "ir-fixtures")]
pub mod ir_variants;
pub mod le_words;
#[cfg(feature = "ir-fixtures")]
pub mod logical_markers;
#[cfg(feature = "ir-fixtures")]
pub mod memory_order_variants;
pub mod monorepo;
#[cfg(feature = "ir-fixtures")]
pub mod mutation_testing;
#[cfg(feature = "ir-fixtures")]
pub mod pass_programs;
#[cfg(feature = "driver-contracts")]
pub mod preferred_dispatch_backend_contract;
#[cfg(feature = "parity-oracles")]
pub mod registry_nets;
pub mod replay_capsule;
#[cfg(feature = "driver-artifact-contracts")]
pub mod resident_async_overlap_contract;
#[cfg(feature = "ir-fixtures")]
pub mod selected_schedules;
#[cfg(feature = "semantic-requests")]
pub mod semantic_requests;
pub mod scalar_corpora;
#[cfg(feature = "spec-strategies")]
pub mod spec_op_strategies;
pub mod spec_variant_tables;
#[cfg(feature = "ir-fixtures")]
pub mod strict_float_programs;
#[cfg(feature = "ir-fixtures")]
pub mod structural_ir;
pub mod sweep_rng;
#[cfg(feature = "driver-artifact-contracts")]
pub mod target_compiler_contract;
#[cfg(feature = "semantic-parity")]
pub mod test_parity_oracles;
#[cfg(feature = "ir-fixtures")]
pub mod tile_programs;
pub mod word_corpora;
#[cfg(feature = "ir-fixtures")]
pub mod wire_hostile_inputs;
#[cfg(feature = "ir-fixtures")]
pub mod wire_round_trip;
#[cfg(feature = "ir-fixtures")]
pub use pass_programs::overfire_grid;
pub mod public_api;

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Per-file read cap for the source enumeration.
///
/// The enumerator reads every `.rs` file under `src/` and `tests/` as text. An
/// unbounded `read_to_string` would let one pathological generated file exhaust
/// memory during a test run, so each file is capped and an over-cap file is a
/// loud failure rather than a silent truncation (a truncated file would drop
/// builders from the enumeration and quietly weaken the closure gate).
pub const MAX_SOURCE_FILE_BYTES: u64 = 4_194_304;

/// Read one source file as text, bounded by [`MAX_SOURCE_FILE_BYTES`].
///
/// The reader every source-derived closure test shares with
/// [`top_level_variant_names`] and [`braced_body`]. A silent truncation drops
/// members from a derived variant set, and a short set agrees with a
/// containment assertion, so the cap is enforced as an error here rather than
/// left to each caller.
pub fn read_source_file_bounded(path: &Path) -> std::io::Result<String> {
    read_source_file_with_cap(path, MAX_SOURCE_FILE_BYTES)
}

fn read_source_file_with_cap(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    let mut text = String::new();
    fs::File::open(path)?
        .take(max_bytes.saturating_add(1))
        .read_to_string(&mut text)?;
    if text.len() as u64 > max_bytes {
        return Err(std::io::Error::other(format!(
            "{} exceeds the {max_bytes} byte source read cap; truncating it \
             would silently drop builders from the closure enumeration. Fix: split the \
             file or raise MAX_SOURCE_FILE_BYTES deliberately.",
            path.display()
        )));
    }
    Ok(text)
}

/// The brace-delimited body that follows `declaration` in `source`.
///
/// `declaration` is the text up to and including the opening brace, so
/// `"pub enum DataType {"` or `"pub trait NodeVisitor {"`. The returned slice
/// excludes both braces and is nesting-aware, which is the whole reason this
/// is one function: a scan that stopped at the first `}` would end inside the
/// first struct-shaped variant or the first defaulted method body, and would
/// then report a short member list as fact.
///
/// Returns `None` when `declaration` does not appear, or when its braces never
/// close.
#[must_use]
pub fn braced_body<'a>(source: &'a str, declaration: &str) -> Option<&'a str> {
    let start = source.find(declaration)? + declaration.len();
    let mut depth = 1usize;
    for (offset, ch) in source[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[start..start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Variant names declared directly in an enum body, ignoring payload contents.
///
/// `body` is what [`braced_body`] returns for a `pub enum NAME {` declaration.
/// One owner because every source-derived variant enumeration asks the same
/// question of a different enum, and a second scan that skipped attributes or
/// doc comments differently would report a different member set for the same
/// file.
#[must_use]
pub fn top_level_variant_names(body: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut depth = 0usize;
    let mut at_item_start = true;
    let mut chars = body.char_indices().peekable();
    while let Some((offset, ch)) = chars.next() {
        match ch {
            '{' | '(' | '[' => {
                depth += 1;
                at_item_start = false;
            }
            '}' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => at_item_start = true,
            '/' if depth == 0 && body[offset..].starts_with("//") => {
                for (_, skipped) in chars.by_ref() {
                    if skipped == '\n' {
                        break;
                    }
                }
            }
            // An attribute belongs to the item after it, so skipping one must
            // leave the item-start state alone. Clearing it made every variant
            // of an enum whose variants carry `#[error(..)]` invisible, and an
            // empty derived set certifies nothing it claims to close over.
            '#' if depth == 0 => {
                let mut bracket_depth = 0usize;
                let mut in_string = false;
                let mut escaped = false;
                for (_, skipped) in chars.by_ref() {
                    if escaped {
                        escaped = false;
                        continue;
                    }
                    match skipped {
                        '\\' if in_string => escaped = true,
                        '"' => in_string = !in_string,
                        '[' if !in_string => bracket_depth += 1,
                        ']' if !in_string => {
                            bracket_depth -= 1;
                            if bracket_depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            c if c.is_whitespace() => {}
            c if depth == 0 && at_item_start && c.is_ascii_uppercase() => {
                let end = body[offset..]
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .map_or(body.len(), |len| offset + len);
                names.insert(body[offset..end].to_string());
                at_item_start = false;
                while chars.peek().is_some_and(|(next, _)| *next < end) {
                    chars.next();
                }
            }
            _ => at_item_start = false,
        }
    }
    names
}

/// Every `IrLevel` variant declared in `vyre-spec`, read from source.
///
/// One owner because a level-closure test in one crate and a pipeline-partition
/// test in another ask the same question of the same enum, and two readers can
/// disagree about the member set the compiler actually has.
///
/// # Panics
/// Panics when `vyre-spec/src/ir_level.rs` is unreadable or no longer declares
/// `pub enum IrLevel`, because either one makes the derived set silently short.
#[must_use]
pub fn declared_level_variants() -> BTreeSet<String> {
    let path = monorepo::vyre_crate_directory("vyre-spec")
        .join("src")
        .join("ir_level.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {path:?} to derive the level set: {err}"));
    let body = braced_body(&source, "pub enum IrLevel {")
        .unwrap_or_else(|| panic!("Fix: {path:?} no longer declares `pub enum IrLevel`"));
    top_level_variant_names(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source readers accept input exactly at the configured cap.
    #[test]
    fn bounded_source_reader_accepts_exact_cap() {
        let dir = tempfile::tempdir().expect("Fix: source-reader fixture directory must exist");
        let path = dir.path().join("source.rs");
        fs::write(&path, "12345678").expect("Fix: source-reader fixture must be writable");

        assert_eq!(
            read_source_file_with_cap(&path, 8).expect("Fix: exact-cap input must be readable"),
            "12345678"
        );
    }

    /// Source readers reject oversized input rather than silently truncating coverage.
    #[test]
    fn bounded_source_reader_rejects_oversized_input() {
        let dir = tempfile::tempdir().expect("Fix: source-reader fixture directory must exist");
        let path = dir.path().join("source.rs");
        fs::write(&path, "123456789").expect("Fix: source-reader fixture must be writable");

        let error = read_source_file_with_cap(&path, 8)
            .expect_err("oversized source evidence must fail closed");
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert!(error
            .to_string()
            .contains("exceeds the 8 byte source read cap"));
    }

    /// Missing source evidence preserves the filesystem error for the calling contract.
    #[test]
    fn bounded_source_reader_reports_missing_files() {
        let dir = tempfile::tempdir().expect("Fix: source-reader fixture directory must exist");
        let error = read_source_file_with_cap(&dir.path().join("missing.rs"), 8)
            .expect_err("missing source evidence must fail");

        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}
