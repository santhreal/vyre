//! Row-contract gate for the C typedef-annotation builder family.
//!
//! WHY. `parsing::c::parse::vast::typedef_ann` publishes a family of builders
//! that all declare the same VAST row table and all write the same row back.
//! Every one of them used to size that table and emit that store loop itself,
//! so each was free to
//!
//!   * declare a zero-length buffer for an empty input, which turns into a
//!     dispatch extent with a zero axis rather than into no work,
//!   * size a row table by a row stride of its own, and
//!   * drop a field it only carries forward, which the next pass reads as a
//!     zeroed parent link or a lost symbol hash.
//!
//! All three were reachable at once, and one of them shipped: the
//! global-typedef fast pass read its forward neighbour with no out-of-range
//! fallback while every sibling and the CPU oracle used one.
//!
//! The member set is read from `typedef_ann.rs` at run time, so adding a
//! `c11_*` export without recording what it does turns this suite red.
//!
//! Does NOT catch: whether a pass computes the right flag for a given C
//! construct. That is what the oracle-parity suites assert. This gate only
//! fixes the shape of the table every one of them agrees to work on.

#![cfg(feature = "c-parser")]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use vyre::ir::{BufferAccess, Expr, Program};
use vyre_libs::parsing::c::lex::tokens::{
    TOK_IDENTIFIER, TOK_INT, TOK_LBRACE, TOK_RBRACE, TOK_SEMICOLON, TOK_TYPEDEF,
};
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_global_typedef_names_fast, c11_annotate_typedef_names,
    c11_annotate_typedef_names_packed_haystack, c11_annotate_typedef_names_precomputed_context,
    c11_annotate_typedef_names_precomputed_context_packed_haystack,
    c11_annotate_typedef_names_precomputed_scope,
    c11_annotate_typedef_names_precomputed_scope_packed_haystack, c11_link_vast_typedef_symbols,
    c11_precompute_vast_decl_contexts, c11_precompute_vast_decl_prefix_starts,
    c11_precompute_vast_scopes, c11_precompute_vast_visible_type,
    c11_precompute_vast_visible_type_packed_haystack, c11_prehash_vast_identifiers,
    c11_prehash_vast_identifiers_packed_haystack,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

/// Module whose public re-exports define the family. Read at run time.
const MEMBER_SOURCE: &str = "src/parsing/c/parse/vast/typedef_ann.rs";

const NODES: &str = "vast_nodes";
const HAYSTACK: &str = "haystack";
const IN_CONTEXTS: &str = "in_contexts";
const IN_VISIBLE_TYPE: &str = "in_visible_type";
const GLOBAL_HASHES: &str = "global_hashes";
const OUT: &str = "out_table";

const SENTINEL: u32 = u32::MAX;

/// What one exported member is.
///
/// A program builder records which VAST row fields it may write. Every other
/// field of every row must survive the pass byte for byte, which is the
/// contract the shared store loop exists to keep.
enum Member {
    /// Output is a VAST row table of the same shape as the input.
    RowTable {
        build: fn(u32) -> Program,
        writes: &'static [usize],
    },
    /// Output is the declaration-context side table, one context row per node.
    ContextTable { build: fn(u32) -> Program },
    /// Output is one word per node.
    WordPerRow { build: fn(u32) -> Program },
    /// Not a program builder. The string says what it is instead.
    NotAProgram(&'static str),
}

impl Member {
    fn build(&self) -> Option<fn(u32) -> Program> {
        match self {
            Self::RowTable { build, .. }
            | Self::ContextTable { build }
            | Self::WordPerRow { build } => Some(*build),
            Self::NotAProgram(_) => None,
        }
    }
}

fn members() -> BTreeMap<&'static str, Member> {
    let mut table: BTreeMap<&'static str, Member> = BTreeMap::new();

    // The annotators resolve visibility and declaration kind, so they write the
    // flags field, the scope carrier and the symbol hash.
    const ANNOTATE_WRITES: &[usize] = &[7, 8, 9];
    table.insert(
        "c11_annotate_typedef_names",
        Member::RowTable {
            build: |n| {
                c11_annotate_typedef_names(
                    NODES,
                    HAYSTACK,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: ANNOTATE_WRITES,
        },
    );
    table.insert(
        "c11_annotate_typedef_names_packed_haystack",
        Member::RowTable {
            build: |n| {
                c11_annotate_typedef_names_packed_haystack(
                    NODES,
                    HAYSTACK,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: ANNOTATE_WRITES,
        },
    );
    table.insert(
        "c11_annotate_typedef_names_precomputed_scope",
        Member::RowTable {
            build: |n| {
                c11_annotate_typedef_names_precomputed_scope(
                    NODES,
                    HAYSTACK,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: ANNOTATE_WRITES,
        },
    );
    table.insert(
        "c11_annotate_typedef_names_precomputed_scope_packed_haystack",
        Member::RowTable {
            build: |n| {
                c11_annotate_typedef_names_precomputed_scope_packed_haystack(
                    NODES,
                    HAYSTACK,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: ANNOTATE_WRITES,
        },
    );
    table.insert(
        "c11_annotate_typedef_names_precomputed_context",
        Member::RowTable {
            build: |n| {
                c11_annotate_typedef_names_precomputed_context(
                    NODES,
                    HAYSTACK,
                    IN_CONTEXTS,
                    IN_VISIBLE_TYPE,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: ANNOTATE_WRITES,
        },
    );
    table.insert(
        "c11_annotate_typedef_names_precomputed_context_packed_haystack",
        Member::RowTable {
            build: |n| {
                c11_annotate_typedef_names_precomputed_context_packed_haystack(
                    NODES,
                    HAYSTACK,
                    IN_CONTEXTS,
                    IN_VISIBLE_TYPE,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: ANNOTATE_WRITES,
        },
    );
    table.insert(
        "c11_annotate_global_typedef_names_fast",
        Member::RowTable {
            // Resolves visibility and declaration kind from a hash table rather
            // than the source, so the scope and symbol fields are carried.
            build: |n| {
                c11_annotate_global_typedef_names_fast(
                    NODES,
                    GLOBAL_HASHES,
                    Expr::u32(n),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: &[7],
        },
    );
    table.insert(
        "c11_prehash_vast_identifiers",
        Member::RowTable {
            build: |n| {
                c11_prehash_vast_identifiers(
                    NODES,
                    HAYSTACK,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: &[9],
        },
    );
    table.insert(
        "c11_prehash_vast_identifiers_packed_haystack",
        Member::RowTable {
            build: |n| {
                c11_prehash_vast_identifiers_packed_haystack(
                    NODES,
                    HAYSTACK,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
            writes: &[9],
        },
    );
    table.insert(
        "c11_precompute_vast_scopes",
        Member::RowTable {
            build: |n| c11_precompute_vast_scopes(NODES, Expr::u32(n), OUT),
            writes: &[8],
        },
    );
    table.insert(
        "c11_link_vast_typedef_symbols",
        Member::RowTable {
            build: |n| c11_link_vast_typedef_symbols(NODES, Expr::u32(n), OUT),
            writes: &[7],
        },
    );
    table.insert(
        "c11_precompute_vast_decl_contexts",
        Member::ContextTable {
            build: |n| c11_precompute_vast_decl_contexts(NODES, Expr::u32(n), OUT),
        },
    );
    table.insert(
        "c11_precompute_vast_decl_prefix_starts",
        Member::ContextTable {
            build: |n| c11_precompute_vast_decl_prefix_starts(NODES, Expr::u32(n), OUT),
        },
    );
    table.insert(
        "c11_precompute_vast_visible_type",
        Member::WordPerRow {
            build: |n| {
                c11_precompute_vast_visible_type(
                    NODES,
                    HAYSTACK,
                    IN_CONTEXTS,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
        },
    );
    table.insert(
        "c11_precompute_vast_visible_type_packed_haystack",
        Member::WordPerRow {
            build: |n| {
                c11_precompute_vast_visible_type_packed_haystack(
                    NODES,
                    HAYSTACK,
                    IN_CONTEXTS,
                    Expr::u32(source_len(n)),
                    Expr::u32(n),
                    OUT,
                )
            },
        },
    );
    table.insert(
        "c11_precompute_vast_scopes_uses_global_stack",
        Member::NotAProgram("predicate on the node count that picks the scope-stack memory tier"),
    );

    table
}

/// Every `c11_*` name the family re-exports, read from the module source.
fn exported_members() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory("vyre-libs").join(MEMBER_SOURCE);
    let source = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: cannot read {} to derive the typedef-annotation member set: {error}",
            path.display()
        )
    });

    let mut members = BTreeSet::new();
    let mut rest = source.as_str();
    while let Some(start) = rest.find("pub use ") {
        let statement = &rest[start..];
        let end = statement
            .find(';')
            .unwrap_or_else(|| panic!("Fix: unterminated `pub use` in {}", path.display()));
        members.extend(c11_identifiers(&statement[..end]));
        rest = &statement[end..];
    }
    assert!(
        !members.is_empty(),
        "Fix: found no `c11_*` re-export in {}; this gate derives its member set from that \
         file and cannot run against an empty set",
        path.display()
    );
    members
}

/// Identifiers beginning `c11_` inside one statement.
fn c11_identifiers(statement: &str) -> Vec<String> {
    let bytes = statement.as_bytes();
    let mut found = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        let is_start = bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_';
        if !is_start
            || (index > 0 && (bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_'))
        {
            index += 1;
            continue;
        }
        let mut end = index;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        let word = &statement[index..end];
        if word.starts_with("c11_") {
            found.push(word.to_string());
        }
        index = end;
    }
    found
}

/// Source length the builders are handed for an `n`-row table.
///
/// Each fixture token is at most four bytes plus a separator, so five bytes per
/// row covers any row count the gate builds.
fn source_len(rows: u32) -> u32 {
    rows.saturating_mul(5)
}

/// An `n`-row VAST whose copied fields carry distinct values.
///
/// The kind, the field-4 back link, the lexeme start and the lexeme length all
/// differ per row, so a pass that stores a constant into any of them, or copies
/// the wrong row's word, is visible. Fields 1 to 3 are the structural links and
/// are `SENTINEL` because that is the only value a link table not yet built can
/// legally hold.
fn marker_rows(stride: usize, rows: usize) -> Vec<u32> {
    const KINDS: [u32; 6] = [
        TOK_TYPEDEF,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_LBRACE,
        TOK_RBRACE,
    ];
    let mut table = vec![0u32; rows * stride];
    for row in 0..rows {
        let base = row * stride;
        table[base] = KINDS[row % KINDS.len()];
        table[base + 1] = SENTINEL;
        table[base + 2] = SENTINEL;
        table[base + 3] = SENTINEL;
        table[base + 4] = row.saturating_sub(1) as u32;
        table[base + 5] = (row * 5) as u32;
        table[base + 6] = 3;
        table[base + 7] = 0;
        table[base + 8] = SENTINEL;
        table[base + 9] = 0;
    }
    table
}

/// Run `program`, feeding `node_words` to the node table and zeros elsewhere.
///
/// Buffer values are supplied in declaration order, skipping workgroup memory,
/// which the evaluator allocates itself.
fn run(name: &str, program: &Program, node_words: &[u32]) -> Vec<u32> {
    let mut inputs: Vec<Value> = Vec::new();
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        let words = buffer.count() as usize;
        let data = if buffer.name() == NODES {
            assert_eq!(
                node_words.len(),
                words,
                "Fix: `{name}` declares {NODES} as {words} words but the gate built \
                 {} for it; the row stride the gate derived disagrees with the declaration",
                node_words.len()
            );
            node_words.to_vec()
        } else {
            vec![0u32; words]
        };
        inputs.push(Value::from(pack_u32_slice(&data)));
    }
    let outputs = vyre_reference::reference_eval(program, &inputs).unwrap_or_else(|error| {
        panic!(
            "Fix: `{name}` failed to execute on a {}-row table: {error}",
            node_words.len() / 10
        )
    });
    decode_u32_le_bytes_all(&outputs[0].to_bytes())
}

/// Declared word count of the buffer named `name`.
fn declared(program: &Program, buffer: &str) -> u32 {
    program
        .buffers()
        .iter()
        .find(|decl| decl.name() == buffer)
        .map(|decl| decl.count())
        .unwrap_or_else(|| panic!("Fix: no buffer named `{buffer}` in this program"))
}

#[test]
fn every_exported_member_has_a_recorded_decision() {
    let discovered = exported_members();
    let table = members();
    let recorded: BTreeSet<String> = table.keys().map(|name| (*name).to_string()).collect();

    let undecided: Vec<&String> = discovered.difference(&recorded).collect();
    assert!(
        undecided.is_empty(),
        "Fix: `{MEMBER_SOURCE}` exports these members with no row in this gate's decision \
         table; add each one as RowTable (naming the VAST fields it writes), ContextTable, \
         WordPerRow, or NotAProgram with the reason: {undecided:?}"
    );

    let stale: Vec<&String> = recorded.difference(&discovered).collect();
    assert!(
        stale.is_empty(),
        "Fix: this gate records members that `{MEMBER_SOURCE}` no longer exports; delete \
         their rows: {stale:?}"
    );
}

#[test]
fn no_builder_declares_an_empty_buffer_for_an_empty_input() {
    for (name, member) in members() {
        let Some(build) = member.build() else {
            continue;
        };
        let program = build(0);
        for buffer in program.buffers() {
            assert!(
                buffer.count() >= 1,
                "Fix: `{name}` declares buffer `{}` with 0 words for an empty node table. \
                 Size it through row_io::declared_rows, which never returns zero: a \
                 zero-length declaration becomes a dispatch extent with a zero axis, which \
                 the launcher rejects instead of treating as no work",
                buffer.name()
            );
        }
    }
}

#[test]
fn every_builder_sizes_its_tables_by_the_one_shared_stride() {
    let table = members();
    let mut node_stride: Option<u32> = None;
    let mut context_stride: Option<u32> = None;

    for (name, member) in &table {
        let Some(build) = member.build() else {
            continue;
        };

        // The stride is whatever the builder declares for a single row.
        let stride = declared(&build(1), NODES);
        assert!(
            stride >= 1,
            "Fix: `{name}` declares a zero-word node table for one row"
        );
        match node_stride {
            None => node_stride = Some(stride),
            Some(shared) => assert_eq!(
                stride, shared,
                "Fix: `{name}` sizes its node table by a row stride of {stride} while the \
                 rest of the family uses {shared}. Every builder must size row tables \
                 through row_io::row_table_words"
            ),
        }

        for rows in [0u32, 1, 2, 3, 7] {
            let program = build(rows);
            let expected_nodes = rows.max(1).saturating_mul(stride);
            assert_eq!(
                declared(&program, NODES),
                expected_nodes,
                "Fix: `{name}` declares the wrong node-table length for {rows} rows"
            );
            match member {
                Member::RowTable { .. } => assert_eq!(
                    declared(&program, OUT),
                    expected_nodes,
                    "Fix: `{name}` declares an output row table of a different length than \
                     its input row table for {rows} rows; both are the same shape"
                ),
                Member::ContextTable { .. } => {
                    let per_row = declared(&build(1), OUT);
                    match context_stride {
                        None => context_stride = Some(per_row),
                        Some(shared) => assert_eq!(
                            per_row, shared,
                            "Fix: `{name}` sizes the declaration-context table by a stride of \
                             {per_row} while its sibling writer uses {shared}; the two write \
                             the same table"
                        ),
                    }
                    assert_eq!(
                        declared(&program, OUT),
                        rows.max(1).saturating_mul(per_row),
                        "Fix: `{name}` declares the wrong context-table length for {rows} rows"
                    );
                }
                Member::WordPerRow { .. } => assert_eq!(
                    declared(&program, OUT),
                    rows.max(1),
                    "Fix: `{name}` writes one word per node, so its output must be exactly \
                     one word per declared row, for {rows} rows"
                ),
                Member::NotAProgram(_) => unreachable!(),
            }
        }
    }

    assert!(
        node_stride.is_some(),
        "Fix: no member of the family builds a program, so this gate proved nothing"
    );
}

#[test]
fn every_row_table_pass_carries_the_fields_it_does_not_write() {
    let table = members();
    let stride = table
        .values()
        .find_map(|member| member.build())
        .map(|build| declared(&build(1), NODES) as usize)
        .expect("Fix: no member of the family builds a program");

    for (name, member) in &table {
        let Member::RowTable { build, writes } = member else {
            continue;
        };
        for rows in [1usize, 4] {
            let input = marker_rows(stride, rows);
            let output = run(name, &build(rows as u32), &input);
            assert_eq!(
                output.len(),
                input.len(),
                "Fix: `{name}` returned {} words for a {rows}-row table, not {}",
                output.len(),
                input.len()
            );
            for row in 0..rows {
                for field in 0..stride {
                    if writes.contains(&field) {
                        continue;
                    }
                    let at = row * stride + field;
                    assert_eq!(
                        output[at], input[at],
                        "Fix: `{name}` changed row {row} field {field} from {} to {} on a \
                         {rows}-row table. It declares that it writes {writes:?}; every other \
                         field must be carried through by row_io::store_row_with_overrides",
                        input[at], output[at]
                    );
                }
            }
        }
    }
}
