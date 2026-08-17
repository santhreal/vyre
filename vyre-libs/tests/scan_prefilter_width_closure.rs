//! Every bounded-ranges prefilter width has its dispatch ABI recorded, and a new
//! width cannot ship without one.
//!
//! # What this gate owns
//!
//! It owns the class "a prefilter width was added and its buffer layout was never
//! recorded anywhere a reader or a host can check". The three shipped
//! match-emitting bounded-ranges programs differ only in how deep their candidate
//! gate looks, and that width decides how many mask buffers sit between the match
//! counter and the match sink. Get the count wrong on a host and the sink is bound
//! to a mask: the dispatch writes triples into a read-only table, or reads a
//! candidate mask out of the output buffer, and the scan silently loses recall.
//!
//! The member set is the `PrefilterWidth` enum itself, read out of the source at
//! run time. Adding a variant makes this suite RED, naming the variant, until a
//! row below records what that width binds. Deleting a variant while its row
//! stays is equally red, so the table cannot rot into a description of a shape
//! nobody ships. Both directions fail closed: a source file that stops parsing as
//! an enum fails rather than passing with an empty member set.
//!
//! # What it deliberately does not own
//!
//! It does not check that a gate ADMITS the right positions. The per-width
//! reference-backend parity tests against the CPU oracle own recall, and this
//! gate would duplicate them badly. It says nothing about the presence-bitmap or
//! region-attributed shapes, whose result buffer sits at binding 6 instead of a
//! match counter, and nothing about the count-only prefilter family, which binds
//! a different ABI entirely.

#![cfg(feature = "pattern-dfa")]

use std::collections::BTreeSet;
use std::fs;

use vyre_foundation::ir::{BufferAccess, Node, Program};
use vyre_foundation::visit::referenced_buffers;
use vyre_libs::pattern::classic_ac::{
    build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce,
    build_ac_bounded_ranges_program_with_subgroup_coalesce,
    build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce, classic_ac_compile,
    CLASSIC_AC_SUFFIX2_MASK_WORDS, CLASSIC_AC_SUFFIX3_BLOOM_WORDS,
};
use vyre_libs::pattern::CompiledDfa;

/// The source file that owns the width table, relative to the crate root. Read at
/// run time so the member set is the shipped enum rather than a copy of it.
const WIDTH_TABLE_SOURCE: &str = "src/scan/classic_ac/bounded_ranges/prefilter/mod.rs";

/// The enum whose variants are the member set.
const WIDTH_TABLE_ENUM: &str = "enum PrefilterWidth {";

/// Binding of the read-write match counter, ahead of the gate masks.
const MATCH_COUNT_BINDING: u32 = 6;

/// Binding the first gate mask occupies. Mirrors `FIRST_GATE_BINDING`, which is
/// crate-internal; a host binds these numbers, so the gate states them
/// independently rather than importing the value under test.
const FIRST_GATE_BINDING: u32 = 7;

const PATTERNS: [&[u8]; 4] = [b"Authorization: Bearer ", b"token", b"tok", b"a"];
const PATTERN_COUNT: u32 = 4;
const MAX_MATCHES: u32 = 1024;

/// One width's recorded dispatch ABI.
struct WidthRow {
    /// The `PrefilterWidth` variant this row records.
    variant: &'static str,
    /// The mask buffers the gate binds, in binding order, with their word counts.
    /// Empty for a width with no candidate mask.
    masks: &'static [(&'static str, u32)],
    /// The region generator the assembled program carries, which is what a host
    /// profile keys a kernel by.
    generator: &'static str,
    /// The shipped builder for this width.
    build: fn(&CompiledDfa, u32, u32, bool) -> Program,
}

/// The recorded ABI of every width the crate ships.
fn rows() -> Vec<WidthRow> {
    vec![
        WidthRow {
            variant: "Unfiltered",
            masks: &[],
            generator: "vyre-libs::matching::classic_ac_bounded_ranges",
            build: build_ac_bounded_ranges_program_with_subgroup_coalesce,
        },
        WidthRow {
            variant: "EndByte",
            masks: &[("candidate_end_mask", 8)],
            generator: "vyre-libs::matching::classic_ac_bounded_ranges_prefilter",
            build: build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce,
        },
        WidthRow {
            variant: "Suffix3",
            masks: &[
                ("candidate_end_mask", 8),
                (
                    "candidate_suffix2_mask",
                    CLASSIC_AC_SUFFIX2_MASK_WORDS as u32,
                ),
                (
                    "candidate_suffix3_bloom",
                    CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32,
                ),
            ],
            generator: "vyre-libs::matching::classic_ac_bounded_ranges_suffix3_prefilter",
            build: build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
        },
    ]
}

fn crate_root() -> std::path::PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root().join("vyre-libs")
}

/// The `PrefilterWidth` variant names, read out of the width table's own source.
///
/// The scan is deliberately literal: find the enum header, then take the leading
/// identifier of every following line until the closing brace, skipping doc
/// comments, attributes and blank lines. A source file that no longer parses that
/// way panics here instead of yielding an empty member set, because an empty set
/// makes every closure assertion below vacuously true.
fn declared_widths() -> BTreeSet<String> {
    let path = crate_root().join(WIDTH_TABLE_SOURCE);
    let source = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: the prefilter width table must be readable at {}: {error}",
            path.display()
        )
    });
    let body = source.split_once(WIDTH_TABLE_ENUM).map(|(_, rest)| rest);
    let Some(body) = body else {
        panic!(
            "Fix: {} no longer declares `{WIDTH_TABLE_ENUM}`. The width closure gate reads the \
             member set from that declaration; point it at the new owner instead of leaving it \
             with nothing to enumerate.",
            path.display()
        );
    };
    let mut variants = BTreeSet::new();
    for line in body.lines() {
        let line = line.trim();
        if line == "}" {
            break;
        }
        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }
        let name = line.trim_end_matches(',');
        assert!(
            name.chars().next().is_some_and(char::is_uppercase)
                && name.chars().all(char::is_alphanumeric),
            "Fix: unexpected line `{line}` inside `{WIDTH_TABLE_ENUM}` in {}. The width closure \
             gate reads one variant per line; keep the declaration to plain variants or teach \
             this scan the new shape.",
            path.display()
        );
        variants.insert(name.to_string());
    }
    assert!(
        variants.len() >= 2,
        "Fix: the width closure gate parsed {} variant(s) out of `{WIDTH_TABLE_ENUM}` in {}. \
         Fewer than two means the scan lost the declaration, and an empty member set would let \
         every assertion below pass without checking anything.",
        variants.len(),
        path.display()
    );
    variants
}

/// The region generator the program's single wrapping region carries.
fn generator_of(program: &Program) -> String {
    let entry = program.entry();
    assert_eq!(
        entry.len(),
        1,
        "Fix: a bounded-ranges program must be one wrapped region"
    );
    match &entry[0] {
        Node::Region { generator, .. } => generator.to_string(),
        other => panic!("Fix: bounded-ranges program entry must be a Region, found {other:?}"),
    }
}

/// Every width the enum declares has a row, and every row names a live width.
///
/// This is the closure itself. The member set comes from the source, so a fourth
/// prefilter width turns this red the moment it lands, naming the variant that
/// has no recorded ABI.
#[test]
fn every_declared_prefilter_width_has_a_recorded_dispatch_abi() {
    let declared = declared_widths();
    let recorded: BTreeSet<String> = rows()
        .iter()
        .map(|row| (*row.variant).to_string())
        .collect();

    let unrecorded: Vec<&String> = declared.difference(&recorded).collect();
    assert!(
        unrecorded.is_empty(),
        "Fix: prefilter width(s) {unrecorded:?} have no row in this gate. A width decides how \
         many mask buffers sit between the match counter and the match sink, so a host that \
         binds the old count writes triples into a read-only mask. Add a row recording the mask \
         names, their word counts and the region generator for each."
    );

    let stale: Vec<&String> = recorded.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "Fix: this gate records prefilter width(s) {stale:?} that `{WIDTH_TABLE_ENUM}` no longer \
         declares. Delete the row, so the recorded ABI cannot describe a shape nobody ships."
    );
}

/// Each row's shipped program binds exactly the ABI the row records: counter,
/// then that width's masks, then the match sink immediately after them.
#[test]
fn each_prefilter_width_binds_the_abi_its_row_records() {
    let dfa = classic_ac_compile(&PATTERNS).dfa;
    for row in rows() {
        let program = (row.build)(&dfa, PATTERN_COUNT, MAX_MATCHES, false);
        let buffers = program.buffers();
        let variant = row.variant;

        assert_eq!(
            buffers.len(),
            8 + row.masks.len(),
            "{variant}: bindings 0-5 inputs, the match counter, {} mask(s) and the match sink",
            row.masks.len()
        );
        assert_eq!(
            generator_of(&program),
            row.generator,
            "{variant}: generator"
        );

        let counter = &buffers[MATCH_COUNT_BINDING as usize];
        assert_eq!(counter.name(), "match_count", "{variant}: counter name");
        assert_eq!(
            counter.binding, MATCH_COUNT_BINDING,
            "{variant}: counter binding"
        );
        assert_eq!(counter.count, 1, "{variant}: counter is one atomic word");
        assert_eq!(
            counter.access,
            BufferAccess::ReadWrite,
            "{variant}: the counter is incremented in place"
        );

        for (offset, (name, words)) in row.masks.iter().enumerate() {
            let binding = FIRST_GATE_BINDING + offset as u32;
            let decl = &buffers[binding as usize];
            assert_eq!(decl.name(), *name, "{variant}: mask at binding {binding}");
            assert_eq!(decl.binding, binding, "{variant}: mask {name} binding");
            assert_eq!(decl.count, *words, "{variant}: mask {name} word count");
            assert_eq!(
                decl.access,
                BufferAccess::ReadOnly,
                "{variant}: mask {name} is a read-only table"
            );
        }

        let sink_binding = FIRST_GATE_BINDING + row.masks.len() as u32;
        let sink = &buffers[sink_binding as usize];
        assert_eq!(sink.name(), "matches", "{variant}: sink at {sink_binding}");
        assert_eq!(sink.binding, sink_binding, "{variant}: sink binding");
        assert_eq!(
            sink.count,
            MAX_MATCHES * 3,
            "{variant}: sink holds max_matches (pattern_id, start, end) triples"
        );
        assert!(sink.is_output, "{variant}: the sink is read back");
    }
}

/// A row cannot claim a gate depth the emitted IR does not have: every mask the
/// program declares is actually read by it.
///
/// A declared-but-unread mask is the exact shape of a half-finished width. It
/// costs a host an upload and a binding while the kernel gates on fewer bytes
/// than the ABI advertises, and no ABI assertion above can see it.
#[test]
fn each_prefilter_width_reads_every_mask_it_declares() {
    let dfa = classic_ac_compile(&PATTERNS).dfa;
    for row in rows() {
        let program = (row.build)(&dfa, PATTERN_COUNT, MAX_MATCHES, false);
        // `referenced_buffers` is the crate-wide exhaustive traversal owner, so a
        // new Node or Expr variant that can hold a load cannot hide one from this.
        let read: BTreeSet<String> = referenced_buffers(&program)
            .into_iter()
            .map(|name| name.to_string())
            .collect();
        for (name, _) in row.masks {
            assert!(
                read.contains(*name),
                "{}: mask `{name}` is declared but never loaded. Either the gate lost a stage or \
                 the row records a width the program does not implement; buffers read: {read:?}",
                row.variant
            );
        }
    }
}

/// A narrower width binds a PREFIX of a wider one: same names, same bindings,
/// same word counts.
///
/// This is what lets one gate serve every width and one host share uploaded
/// masks across shapes. A new width that inserts a stage in the middle, or
/// renumbers an existing one, breaks both and shows up here rather than as lost
/// recall on a dispatch.
#[test]
fn prefilter_widths_form_a_mask_prefix_chain() {
    let mut ordered = rows();
    ordered.sort_by_key(|row| row.masks.len());
    for pair in ordered.windows(2) {
        let [narrow, wide] = pair else { unreachable!() };
        assert!(
            narrow.masks.len() < wide.masks.len(),
            "Fix: widths {} and {} record the same mask count. Two widths that bind the same \
             masks are one width; give the new stage its own mask or drop the variant.",
            narrow.variant,
            wide.variant
        );
        assert_eq!(
            narrow.masks,
            &wide.masks[..narrow.masks.len()],
            "Fix: width {} does not bind the leading masks of {}. Each stage only narrows the \
             candidate set, so a wider gate must keep the narrower gate's masks at the same \
             bindings with the same sizes.",
            narrow.variant,
            wide.variant
        );
    }
}
