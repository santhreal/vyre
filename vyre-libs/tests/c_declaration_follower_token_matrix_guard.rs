//! Every token kind, in both positions that decide whether an identifier is a
//! declarator name.
//!
//! The decision "is this identifier a declarator" is spelled three times: once
//! in the CPU reference annotator, once in the self-contained GPU annotator, and
//! once in the precomputed-context GPU annotator. Each carried its own token
//! table, and the tables drifted, so the three answered differently for tokens
//! nobody had a fixture for. `token_grammar::declarations` owns the tables now,
//! and this gate is what keeps them one table.
//!
//! The member set is read out of `vyre-spec/src/c11_token.rs` at test time, not
//! listed here. Adding a token kind therefore adds two matrix rows on its own,
//! and a kind the tables classify inconsistently fails by name.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod common;

use common::decode_u32_words as words_from_bytes;
use common::u32_bytes as bytes;
use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::{TOK_IDENTIFIER, TOK_INT, TOK_SEMICOLON};
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_typedef_names, c11_annotate_typedef_names_precomputed_context,
    c11_precompute_vast_decl_contexts, c11_precompute_vast_scopes,
    c11_precompute_vast_visible_type, c11_prehash_vast_identifiers,
};
use vyre_reference::value::Value;

const VAST_NODE_STRIDE_U32: usize = 10;
const VAST_DECL_CONTEXT_STRIDE_U32: usize = 4;
const SENTINEL: u32 = u32::MAX;

/// Token vocabulary, read from the spec crate that owns the numbering.
///
/// A hardcoded copy of this list would go stale in silence, which is the same
/// failure as having no matrix at all.
fn token_vocabulary() -> Vec<(String, u32)> {
    let spec = vyre_test_support::monorepo::vyre_crate_directory("vyre-spec")
        .join("src")
        .join("c11_token.rs");
    let text = std::fs::read_to_string(&spec)
        .unwrap_or_else(|err| panic!("token vocabulary must be readable at {spec:?}: {err}"));
    let mut tokens: Vec<(String, u32)> = text
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("pub const TOK_")?;
            let (name, rest) = rest.split_once(": u32 = ")?;
            let value = rest.split(';').next()?.trim().parse::<u32>().ok()?;
            Some((format!("TOK_{name}"), value))
        })
        .collect();
    tokens.sort_unstable_by_key(|(_, value)| *value);
    tokens.dedup_by_key(|(_, value)| *value);
    assert!(
        tokens.len() > 100,
        "token vocabulary parse found only {} entries; the spec layout changed",
        tokens.len()
    );
    tokens
}

fn build_vast(tokens: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut vast = vec![0u32; tokens.len() * VAST_NODE_STRIDE_U32];
    for (idx, (kind, start, len)) in tokens.iter().copied().enumerate() {
        let base = idx * VAST_NODE_STRIDE_U32;
        vast[base] = kind;
        vast[base + 1] = SENTINEL;
        vast[base + 2] = SENTINEL;
        vast[base + 3] = SENTINEL;
        vast[base + 4] = idx.saturating_sub(1) as u32;
        vast[base + 5] = start;
        vast[base + 6] = len;
    }
    vast
}

fn expanded_haystack(source: &[u8]) -> Vec<u8> {
    bytes(
        &source
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>(),
    )
}

fn eval_words(program: &vyre::ir::Program, inputs: Vec<Vec<u8>>) -> Vec<u32> {
    let values = inputs.into_iter().map(Value::from).collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(program, &values)
        .expect("declaration-follower matrix program must execute under reference_eval");
    words_from_bytes(&outputs[0].to_bytes())
}

fn reference_annotation(vast: &[u32], source: &[u8], n: u32) -> Vec<u32> {
    eval_words(
        &c11_annotate_typedef_names(
            "vast_nodes",
            "haystack",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "annotated",
        ),
        vec![
            bytes(vast),
            expanded_haystack(source),
            vec![0u8; vast.len() * 4],
        ],
    )
}

/// Annotation through the precomputed-context pipeline: prehash, scopes,
/// decl-contexts, visible-type, then the variant annotator.
fn precomputed_annotation(vast: &[u32], source: &[u8], n: u32) -> Vec<u32> {
    let hashed = eval_words(
        &c11_prehash_vast_identifiers(
            "vast_nodes",
            "haystack",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "out_hashed",
        ),
        vec![
            bytes(vast),
            expanded_haystack(source),
            vec![0u8; vast.len() * 4],
        ],
    );
    let prepared = eval_words(
        &c11_precompute_vast_scopes("vast_nodes", Expr::u32(n), "out_scoped"),
        vec![
            bytes(&hashed),
            vec![0u8; vast.len() * 4],
            vec![0u8; n.max(1) as usize * 4],
        ],
    );
    let decl_contexts = eval_words(
        &c11_precompute_vast_decl_contexts("vast_nodes", Expr::u32(n), "out_decl_contexts"),
        vec![
            bytes(&prepared),
            vec![0u8; n.max(1) as usize * VAST_DECL_CONTEXT_STRIDE_U32 * 4],
        ],
    );
    let visible_type = eval_words(
        &c11_precompute_vast_visible_type(
            "vast_nodes",
            "haystack",
            "decl_contexts",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "out_visible_type",
        ),
        vec![
            bytes(&prepared),
            expanded_haystack(source),
            bytes(&decl_contexts),
            vec![0u8; n.max(1) as usize * 4],
        ],
    );
    eval_words(
        &c11_annotate_typedef_names_precomputed_context(
            "vast_nodes",
            "haystack",
            "decl_contexts",
            "visible_type",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "annotated",
        ),
        vec![
            bytes(&prepared),
            expanded_haystack(source),
            bytes(&decl_contexts),
            bytes(&visible_type),
            vec![0u8; vast.len() * 4],
        ],
    )
}

/// Typedef flags the two annotators assign to node `subject`.
fn flags_pair(tokens: &[(u32, u32, u32)], source: &[u8], subject: usize) -> (u32, u32) {
    let n = tokens.len() as u32;
    let vast = build_vast(tokens);
    let reference = reference_annotation(&vast, source, n);
    let precomputed = precomputed_annotation(&vast, source, n);
    let flags_field = subject * VAST_NODE_STRIDE_U32 + 7;
    (reference[flags_field], precomputed[flags_field])
}

/// `int a <tok>` for every `<tok>`: whichever tokens may follow a declarator
/// name, both annotators must agree on which ones do.
#[test]
fn every_token_kind_agrees_in_the_declarator_follower_position() {
    // "int a X" - node 2 carries the token under test, one byte wide, so no
    // token's lexeme can reach into a neighbour's span.
    let source: &[u8] = b"int a X";
    let mut drift: Vec<String> = Vec::new();
    for (name, id) in token_vocabulary() {
        let tokens = vec![(TOK_INT, 0u32, 3u32), (TOK_IDENTIFIER, 4, 1), (id, 6, 1)];
        let (reference, precomputed) = flags_pair(&tokens, source, 1);
        if reference != precomputed {
            drift.push(format!(
                "  follower {name}: reference flags {reference:#x}, precomputed flags {precomputed:#x}"
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "{} token kinds are classified differently by the two declarator-follower \
         tables; both must read token_grammar::declarations:\n{}",
        drift.len(),
        drift.join("\n"),
    );
}

/// `int <tok> a ;` for every `<tok>`: whichever tokens disqualify the identifier
/// after them from being a declarator name, both annotators must agree.
#[test]
fn every_token_kind_agrees_in_the_declarator_previous_position() {
    // "int X a ;" - the type prefix is present, so the disqualifier is the only
    // thing that can hold the identifier back.
    let source: &[u8] = b"int X a ;";
    let mut drift: Vec<String> = Vec::new();
    for (name, id) in token_vocabulary() {
        let tokens = vec![
            (TOK_INT, 0u32, 3u32),
            (id, 4, 1),
            (TOK_IDENTIFIER, 6, 1),
            (TOK_SEMICOLON, 8, 1),
        ];
        let (reference, precomputed) = flags_pair(&tokens, source, 2);
        if reference != precomputed {
            drift.push(format!(
                "  previous {name}: reference flags {reference:#x}, precomputed flags {precomputed:#x}"
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "{} token kinds are classified differently by the two declarator-previous \
         tables; both must read token_grammar::declarations:\n{}",
        drift.len(),
        drift.join("\n"),
    );
}

/// `<tok> a ;` for every `<tok>`: whichever tokens make up a declaration prefix,
/// both annotators must agree on which ones do.
#[test]
fn every_token_kind_agrees_in_the_declaration_prefix_position() {
    // "X a ;" - the token under test is the whole prefix, so the identifier is a
    // declarator only if that token alone establishes a declaration.
    let source: &[u8] = b"X a ;";
    let mut drift: Vec<String> = Vec::new();
    for (name, id) in token_vocabulary() {
        let tokens = vec![
            (id, 0u32, 1u32),
            (TOK_IDENTIFIER, 2, 1),
            (TOK_SEMICOLON, 4, 1),
        ];
        let (reference, precomputed) = flags_pair(&tokens, source, 1);
        if reference != precomputed {
            drift.push(format!(
                "  prefix {name}: reference flags {reference:#x}, precomputed flags {precomputed:#x}"
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "{} token kinds are classified differently by the two declaration-prefix \
         tables; both must read token_grammar::declarations:\n{}",
        drift.len(),
        drift.join("\n"),
    );
}
