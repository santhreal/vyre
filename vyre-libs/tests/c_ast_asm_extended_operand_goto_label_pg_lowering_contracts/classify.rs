// C parser contract tests for GNU inline asm with output operands, input
// operands, memory/cc clobbers, goto labels, and symbolic operand names  -
// constructs likely to break VAST/PG lowering.
//
// Constructs under test:
//   - asm with multiple output operands
//   - asm with multiple input operands
//   - asm with memory and cc clobbers
//   - asm goto with multiple destination labels
//   - asm with earlyclobber and symbolic names
//   - PG lowering preservation and GPU/CPU parity
//
// A missing GPU adapter is a configuration failure; tests do not skip.

use super::asm_extended_operands::*;
use crate::c_frontend::rows::{
    assert_pg_preserves_fixture_row as assert_pg_preserves_row, row_indices,
};
use crate::c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ASM_CLOBBERS_LIST,
    C_AST_KIND_ASM_GOTO_LABELS, C_AST_KIND_ASM_INPUT_OPERAND, C_AST_KIND_ASM_OUTPUT_OPERAND,
    C_AST_KIND_ASM_TEMPLATE, C_AST_KIND_INLINE_ASM,
};

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_asm_multiple_output_input_operands_classifies() {
    let fix = fixture_asm_multiple_output_input_operands();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![0],
        "asm must classify as INLINE_ASM"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_TEMPLATE),
        vec![3],
        "template must classify"
    );
    let outputs = row_indices(&typed, C_AST_KIND_ASM_OUTPUT_OPERAND);
    assert_eq!(
        outputs.len(),
        2,
        "two output operands must classify, got {outputs:?}"
    );
    let inputs = row_indices(&typed, C_AST_KIND_ASM_INPUT_OPERAND);
    assert_eq!(
        inputs.len(),
        2,
        "two input operands must classify, got {inputs:?}"
    );
}

#[test]
pub(crate) fn cpu_asm_memory_and_cc_clobbers_classifies() {
    let fix = fixture_asm_memory_and_cc_clobbers();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![0],
        "asm must classify as INLINE_ASM"
    );
    let clobbers = row_indices(&typed, C_AST_KIND_ASM_CLOBBERS_LIST);
    assert_eq!(
        clobbers.len(),
        2,
        "memory and cc clobbers must classify, got {clobbers:?}"
    );
}

#[test]
pub(crate) fn cpu_asm_goto_multiple_labels_classifies() {
    let fix = fixture_asm_goto_multiple_labels();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![0],
        "asm goto must classify as INLINE_ASM"
    );
    let labels = row_indices(&typed, C_AST_KIND_ASM_GOTO_LABELS);
    assert_eq!(
        labels.len(),
        2,
        "fail and ok must classify as ASM_GOTO_LABELS, got {labels:?}"
    );
}

#[test]
pub(crate) fn cpu_asm_symbolic_names_and_earlyclobber_classifies() {
    let fix = fixture_asm_symbolic_names_and_earlyclobber();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![0],
        "asm must classify as INLINE_ASM"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_OUTPUT_OPERAND),
        vec![6],
        "output operand with earlyclobber must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_INPUT_OPERAND),
        vec![12],
        "input operand with symbolic name must classify"
    );
}

#[test]
pub(crate) fn cpu_asm_extended_output_only_classifies() {
    let fix = fixture_asm_extended_output_only();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![0],
        "__asm__ must classify as INLINE_ASM"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_OUTPUT_OPERAND),
        vec![6],
        "output-only operand must classify"
    );
}

#[test]
pub(crate) fn cpu_asm_goto_three_labels_classifies() {
    let fix = fixture_asm_goto_three_labels();
    let typed = classify(&fix);
    let labels = row_indices(&typed, C_AST_KIND_ASM_GOTO_LABELS);
    assert_eq!(
        labels.len(),
        3,
        "three goto labels must classify, got {labels:?}"
    );
}

// ---------------------------------------------------------------------------
// PG lowering preservation contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn pg_lower_preserves_asm_multiple_output_input_operands() {
    let fix = fixture_asm_multiple_output_input_operands();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 0, C_AST_KIND_INLINE_ASM);
    for idx in row_indices(&typed, C_AST_KIND_ASM_OUTPUT_OPERAND) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_ASM_OUTPUT_OPERAND);
    }
    for idx in row_indices(&typed, C_AST_KIND_ASM_INPUT_OPERAND) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_ASM_INPUT_OPERAND);
    }
}

#[test]
pub(crate) fn pg_lower_preserves_asm_memory_and_cc_clobbers() {
    let fix = fixture_asm_memory_and_cc_clobbers();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 0, C_AST_KIND_INLINE_ASM);
    for idx in row_indices(&typed, C_AST_KIND_ASM_CLOBBERS_LIST) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_ASM_CLOBBERS_LIST);
    }
}

#[test]
pub(crate) fn pg_lower_preserves_asm_goto_multiple_labels() {
    let fix = fixture_asm_goto_multiple_labels();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 0, C_AST_KIND_INLINE_ASM);
    for idx in row_indices(&typed, C_AST_KIND_ASM_GOTO_LABELS) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_ASM_GOTO_LABELS);
    }
}
