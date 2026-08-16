//! The per-row phases of typedef annotation, as ops of their own.
//!
//! `c11_annotate_typedef_names` used to carry every phase inline, which put it
//! at 613 statement nodes against a 200 budget, control-flow depth 20 against
//! 6, and 37 loops against 8. The composition-discipline gate has no exemption
//! list by design, so the op was red.
//!
//! Each phase answers one question about one VAST row and returns a single
//! `u32`. The annotator calls them and the calls inline away before lowering,
//! so the emitted kernel is unchanged while the op the gate measures is a
//! fraction of its former size.
//!
//! The buffer contract lives in `vast::phase_program`; the program builders live
//! beside the emitters they are built from. This module owns the witness
//! fixtures and the registrations.

use super::super::phase_program::{
    HAYSTACK_SIGNATURE, ROW_AND_NUM_NODES_SIGNATURE, ROW_SIGNATURE,
};
use super::super::phase_witness::PHASE_WITNESS_ROWS;
use vyre_foundation::operation::OperationFixtures;

#[cfg(any(test, feature = "cpu-parity"))]
use super::super::phase_witness::PhaseWitness;
#[cfg(any(test, feature = "cpu-parity"))]
use super::super::ref_typedef::{
    declaration_kind_at, enclosing_function_lparen, identifier_row_hash, scope_open_before,
    visible_declaration_kind,
};

pub(super) use super::super::build::{
    BUILTIN_DECL_KIND_FOR_ROW_OP_ID, DECL_KIND_FOR_ROW_OP_ID, DECL_KIND_FOR_ROW_PACKED_OP_ID,
    ENCLOSING_FUNCTION_LPAREN_FOR_ROW_OP_ID, IDENTIFIER_ROW_HASH_OP_ID,
    IDENTIFIER_ROW_HASH_PACKED_OP_ID, SCOPE_OPEN_FOR_ROW_OP_ID, VISIBLE_NAME_FOR_ROW_OP_ID,
    VISIBLE_NAME_FOR_ROW_PACKED_OP_ID,
};

/// Row the scope-walk and declaration-scan fixtures ask about: the declarator
/// `v` inside the block. Both scans walk backwards from it and stop on real
/// structure rather than falling out at their initial sentinel.
#[cfg(any(test, feature = "cpu-parity"))]
const WITNESS_DECLARATOR_ROW: u32 = 11;

/// Row the visibility fixtures ask about: the use of the typedef name `T`
/// inside the block, which resolves back to the file-scope `typedef int T`.
#[cfg(any(test, feature = "cpu-parity"))]
const WITNESS_TYPEDEF_USE_ROW: u32 = 10;

/// Row the builtin declaration-kind fixture asks about: the typedef name being
/// declared at file scope. Its prefix carries `typedef` and `int`, so the scan
/// classifies it from keywords alone, which is what the phase sees without a
/// source haystack to resolve type names against.
#[cfg(any(test, feature = "cpu-parity"))]
const WITNESS_TYPEDEF_DECLARATOR_ROW: u32 = 2;

/// Buffers for [`ROW_SIGNATURE`], in declaration order.
#[cfg(any(test, feature = "cpu-parity"))]
fn row_phase_witness_inputs(row: u32) -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    vec![vec![witness.node_bytes, row.to_le_bytes().to_vec()]]
}

/// Buffers for [`ROW_AND_NUM_NODES_SIGNATURE`], in declaration order.
#[cfg(any(test, feature = "cpu-parity"))]
fn row_and_count_witness_inputs(row: u32) -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    vec![vec![
        witness.node_bytes,
        row.to_le_bytes().to_vec(),
        PHASE_WITNESS_ROWS.to_le_bytes().to_vec(),
    ]]
}

/// Buffers for [`HAYSTACK_SIGNATURE`], in declaration order.
#[cfg(any(test, feature = "cpu-parity"))]
fn haystack_phase_witness_inputs(row: u32, packed_haystack: bool) -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    let haystack = witness.haystack_bytes(packed_haystack);
    vec![vec![
        witness.node_bytes,
        haystack,
        row.to_le_bytes().to_vec(),
        (witness.source.len() as u32).to_le_bytes().to_vec(),
        PHASE_WITNESS_ROWS.to_le_bytes().to_vec(),
    ]]
}

/// One `u32` output, little-endian.
#[cfg(any(test, feature = "cpu-parity"))]
fn phase_witness_expected(value: u32) -> Vec<Vec<Vec<u8>>> {
    vec![vec![value.to_le_bytes().to_vec()]]
}

#[cfg(any(test, feature = "cpu-parity"))]
fn scope_open_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    row_phase_witness_inputs(WITNESS_DECLARATOR_ROW)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn scope_open_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    phase_witness_expected(scope_open_before(
        &witness.node_words,
        WITNESS_DECLARATOR_ROW as usize,
    ))
}

#[cfg(any(test, feature = "cpu-parity"))]
fn function_lparen_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    row_and_count_witness_inputs(WITNESS_DECLARATOR_ROW)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn function_lparen_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    phase_witness_expected(enclosing_function_lparen(
        &witness.node_words,
        WITNESS_DECLARATOR_ROW as usize,
    ))
}

#[cfg(any(test, feature = "cpu-parity"))]
fn builtin_decl_kind_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    row_and_count_witness_inputs(WITNESS_TYPEDEF_DECLARATOR_ROW)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn builtin_decl_kind_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    phase_witness_expected(declaration_kind_at(
        &witness.node_words,
        WITNESS_TYPEDEF_DECLARATOR_ROW as usize,
        &witness.source,
    ))
}

#[cfg(any(test, feature = "cpu-parity"))]
fn identifier_hash_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_TYPEDEF_USE_ROW, false)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn identifier_hash_witness_packed_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_TYPEDEF_USE_ROW, true)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn identifier_hash_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    phase_witness_expected(identifier_row_hash(
        &witness.node_words,
        WITNESS_TYPEDEF_USE_ROW as usize,
        &witness.source,
    ))
}

#[cfg(any(test, feature = "cpu-parity"))]
fn visible_name_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_TYPEDEF_USE_ROW, false)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn visible_name_witness_packed_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_TYPEDEF_USE_ROW, true)
}

/// The phase answers the visibility question as a flag, so the oracle's
/// three-valued declaration kind collapses to "is a visible typedef name".
#[cfg(any(test, feature = "cpu-parity"))]
fn visible_name_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    let kind = visible_declaration_kind(
        &witness.node_words,
        WITNESS_TYPEDEF_USE_ROW as usize,
        &witness.source,
        witness.lexeme(WITNESS_TYPEDEF_USE_ROW),
    );
    phase_witness_expected(u32::from(kind == 1))
}

#[cfg(any(test, feature = "cpu-parity"))]
fn decl_kind_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_DECLARATOR_ROW, false)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn decl_kind_witness_packed_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_DECLARATOR_ROW, true)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn decl_kind_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    phase_witness_expected(declaration_kind_at(
        &witness.node_words,
        WITNESS_DECLARATOR_ROW as usize,
        &witness.source,
    ))
}

/// Bind one fixture to a cfg-selected const so `inventory::submit!` stays a
/// single const expression. The oracles behind these fixtures are themselves
/// gated on `cpu-parity`, so every other configuration registers `None`.
/// `default` implies `matching-dfa` implies `cpu-parity`, so every conformance
/// harness still sees them.
macro_rules! witness_fixture {
    ($name:ident = $function:ident) => {
        #[cfg(any(test, feature = "cpu-parity"))]
        const $name: Option<OperationFixtures> = Some($function);
        #[cfg(not(any(test, feature = "cpu-parity")))]
        const $name: Option<OperationFixtures> = None;
    };
}

witness_fixture!(SCOPE_OPEN_INPUTS = scope_open_witness_inputs);
witness_fixture!(SCOPE_OPEN_EXPECTED = scope_open_witness_expected);
witness_fixture!(FUNCTION_LPAREN_INPUTS = function_lparen_witness_inputs);
witness_fixture!(FUNCTION_LPAREN_EXPECTED = function_lparen_witness_expected);
witness_fixture!(BUILTIN_DECL_KIND_INPUTS = builtin_decl_kind_witness_inputs);
witness_fixture!(BUILTIN_DECL_KIND_EXPECTED = builtin_decl_kind_witness_expected);
witness_fixture!(IDENTIFIER_HASH_INPUTS = identifier_hash_witness_inputs);
witness_fixture!(IDENTIFIER_HASH_PACKED_INPUTS = identifier_hash_witness_packed_inputs);
witness_fixture!(IDENTIFIER_HASH_EXPECTED = identifier_hash_witness_expected);
witness_fixture!(VISIBLE_NAME_INPUTS = visible_name_witness_inputs);
witness_fixture!(VISIBLE_NAME_PACKED_INPUTS = visible_name_witness_packed_inputs);
witness_fixture!(VISIBLE_NAME_EXPECTED = visible_name_witness_expected);
witness_fixture!(DECL_KIND_INPUTS = decl_kind_witness_inputs);
witness_fixture!(DECL_KIND_PACKED_INPUTS = decl_kind_witness_packed_inputs);
witness_fixture!(DECL_KIND_EXPECTED = decl_kind_witness_expected);

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SCOPE_OPEN_FOR_ROW_OP_ID,
        super::super::build::c11_typedef_scope_open_for_row,
        SCOPE_OPEN_INPUTS,
        SCOPE_OPEN_EXPECTED,
    )
    .with_signature(ROW_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        ENCLOSING_FUNCTION_LPAREN_FOR_ROW_OP_ID,
        super::super::build::c11_enclosing_function_lparen_for_row,
        FUNCTION_LPAREN_INPUTS,
        FUNCTION_LPAREN_EXPECTED,
    )
    .with_signature(ROW_AND_NUM_NODES_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        BUILTIN_DECL_KIND_FOR_ROW_OP_ID,
        super::super::build::c11_builtin_declaration_kind_for_row,
        BUILTIN_DECL_KIND_INPUTS,
        BUILTIN_DECL_KIND_EXPECTED,
    )
    .with_signature(ROW_AND_NUM_NODES_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        IDENTIFIER_ROW_HASH_OP_ID,
        super::super::build::c11_identifier_row_hash,
        IDENTIFIER_HASH_INPUTS,
        IDENTIFIER_HASH_EXPECTED,
    )
    .with_signature(HAYSTACK_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        IDENTIFIER_ROW_HASH_PACKED_OP_ID,
        super::super::build::c11_identifier_row_hash_packed_haystack,
        IDENTIFIER_HASH_PACKED_INPUTS,
        IDENTIFIER_HASH_EXPECTED,
    )
    .with_signature(HAYSTACK_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        VISIBLE_NAME_FOR_ROW_OP_ID,
        super::super::build::c11_typedef_visible_name_for_row,
        VISIBLE_NAME_INPUTS,
        VISIBLE_NAME_EXPECTED,
    )
    .with_signature(HAYSTACK_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        VISIBLE_NAME_FOR_ROW_PACKED_OP_ID,
        super::super::build::c11_typedef_visible_name_for_row_packed_haystack,
        VISIBLE_NAME_PACKED_INPUTS,
        VISIBLE_NAME_EXPECTED,
    )
    .with_signature(HAYSTACK_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        DECL_KIND_FOR_ROW_OP_ID,
        super::super::build::c11_typedef_decl_kind_for_row,
        DECL_KIND_INPUTS,
        DECL_KIND_EXPECTED,
    )
    .with_signature(HAYSTACK_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        DECL_KIND_FOR_ROW_PACKED_OP_ID,
        super::super::build::c11_typedef_decl_kind_for_row_packed_haystack,
        DECL_KIND_PACKED_INPUTS,
        DECL_KIND_EXPECTED,
    )
    .with_signature(HAYSTACK_SIGNATURE)
}
