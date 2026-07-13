//! Behavioral parity for `c11_annotate_global_typedef_names_fast` (registry-closure orphan).
//!
//! The pass annotates each VAST identifier node's TYPEDEF_FLAGS field from a table of global
//! typedef-name hashes: a node is a VISIBLE_TYPEDEF_NAME when its symbol hash is a known global
//! typedef AND a prior same-hash identifier was itself declared with a `typedef` prefix; a
//! declaration-candidate identifier becomes a TYPEDEF_DECLARATOR (its own prefix has `typedef`)
//! or an ORDINARY_DECLARATOR (its prefix has a type, including a *prior typedef-name used as a
//! type*, resolved via the global-hash table). We drive it through `reference_eval` on a
//! hand-built VAST for `typedef int foo; foo bar;` and assert the exact flag on every node.
//!
//! This is precisely the typedef-name-as-type resolution the haystack-free precomputed-context
//! variant CANNOT do (BACKLOG.md WIRING-vyrelibs-precomputed-context-divergence), here the fast
//! pass gets it right because it consults the global-typedef-hash table during its prefix scan.
//! Drains the vyre-libs slice of BACKLOG.md WIRING-tautology-closure-25crates.
#![cfg(feature = "c-parser")]
#![forbid(unsafe_code)]

use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::{TOK_IDENTIFIER, TOK_INT, TOK_SEMICOLON, TOK_TYPEDEF};
use vyre_libs::parsing::c::parse::vast::c11_annotate_global_typedef_names_fast;
use vyre_reference::value::Value;

// VAST wire layout (mirrors the private consts in parse/vast/mod.rs).
const STRIDE: usize = 10;
const FLAGS_FIELD: usize = 7;
const SCOPE_FIELD: usize = 8;
const SYMBOL_FIELD: usize = 9;
const SENTINEL: u32 = u32::MAX;
// Flag values.
const VISIBLE_TYPEDEF_NAME: u32 = 1;
const TYPEDEF_DECLARATOR: u32 = 1 << 1;
const ORDINARY_DECLARATOR: u32 = 1 << 2;

// Two distinct non-zero identifier symbol hashes.
const H_FOO: u32 = 0x0000_0F00;
const H_BAR: u32 = 0x0000_0BA1;

/// Build one VAST row: `kind` in field 0, `SENTINEL` links, `symbol_hash` in field 9, scope 0.
fn row(kind: u32, symbol_hash: u32) -> [u32; STRIDE] {
    let mut r = [0u32; STRIDE];
    r[0] = kind;
    r[1] = SENTINEL; // links / parent are unused by this pass and copied through
    r[2] = SENTINEL;
    r[3] = SENTINEL;
    r[4] = SENTINEL;
    r[FLAGS_FIELD] = 0;
    r[SCOPE_FIELD] = 0; // no enclosing brace scope -> no aggregate-body suppression
    r[SYMBOL_FIELD] = symbol_hash;
    r
}

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn global_typedef_annotate_marks_declarator_visible_and_ordinary_flags() {
    // `typedef int foo ; foo bar ;`
    let nodes: Vec<[u32; STRIDE]> = vec![
        row(TOK_TYPEDEF, 0),        // 0: typedef
        row(TOK_INT, 0),            // 1: int
        row(TOK_IDENTIFIER, H_FOO), // 2: foo  (typedef declarator)
        row(TOK_SEMICOLON, 0),      // 3: ;
        row(TOK_IDENTIFIER, H_FOO), // 4: foo  (type name used as a type)
        row(TOK_IDENTIFIER, H_BAR), // 5: bar  (ordinary declarator of type foo)
        row(TOK_SEMICOLON, 0),      // 6: ;
    ];
    let num_nodes = nodes.len() as u32;
    let flat: Vec<u32> = nodes.iter().flat_map(|r| r.iter().copied()).collect();
    let global_hashes = [H_FOO]; // bar's hash is deliberately NOT a global typedef
    let out_init = vec![0u32; flat.len()];

    let program = c11_annotate_global_typedef_names_fast(
        "vast_nodes",
        "global_typedef_hashes",
        Expr::u32(num_nodes),
        Expr::u32(global_hashes.len() as u32),
        "out_annotated_vast_nodes",
    );

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&flat)),
            Value::from(pack(&global_hashes)),
            Value::from(pack(&out_init)),
        ],
    )
    .expect("global typedef annotate program must execute under reference_eval");
    let out = unpack(&outputs[0].to_bytes());

    let flag = |node: usize| out[node * STRIDE + FLAGS_FIELD];

    assert_eq!(flag(0), 0, "`typedef` keyword carries no typedef flag");
    assert_eq!(flag(1), 0, "`int` keyword carries no typedef flag");
    assert_eq!(
        flag(2),
        TYPEDEF_DECLARATOR,
        "`foo` in `typedef int foo;` is the TYPEDEF_DECLARATOR (its prefix has `typedef`)"
    );
    assert_eq!(flag(3), 0, "`;` carries no typedef flag");
    assert_eq!(
        flag(4),
        VISIBLE_TYPEDEF_NAME,
        "`foo` in `foo bar;` is a VISIBLE_TYPEDEF_NAME (global hash + prior typedef-declared same hash)"
    );
    assert_eq!(
        flag(5),
        ORDINARY_DECLARATOR,
        "`bar` is an ORDINARY_DECLARATOR, its prefix `foo` is a prior typedef-name used as a type"
    );
    assert_eq!(flag(6), 0, "trailing `;` carries no typedef flag");

    // Every non-FLAGS field must be copied through unchanged (kind + symbol hash preserved).
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(
            out[i * STRIDE],
            node[0],
            "node {i} kind must be copied through unchanged"
        );
        assert_eq!(
            out[i * STRIDE + SYMBOL_FIELD],
            node[SYMBOL_FIELD],
            "node {i} symbol hash must be copied through unchanged"
        );
    }
}
