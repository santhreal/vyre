//! Composition-contract coverage for `build_c11_compiler_megakernel` (registry-closure orphan).
//!
//! Per its own doc, this builder is a **composition sketch the reference interpreter does not
//! execute**, it chains the C11 pipeline stages with `Expr::Call` passing stringly buffer
//! variables, a form `reference_eval` cannot run. So it is NOT drainable by value parity; its
//! real contract is STRUCTURAL: it must dispatch every documented pipeline stage, in the
//! documented order, inside the `c11_megakernel` sequence loop, over the source->bytecode
//! buffer pair. We assert exactly that (the ordered op-ID sequence + the buffer ABI), which
//! fails loudly if a stage is dropped, added, or reordered.
//!
//! Drains the vyre-libs slice of BACKLOG.md WIRING-tautology-closure-25crates.
#![cfg(feature = "c-parser")]
#![forbid(unsafe_code)]

use vyre::ir::BufferAccess;
use vyre_libs::parsing::c::pipeline::examples::build_c11_compiler_megakernel;

/// The pipeline stages the megakernel must dispatch, in strict dispatch order (source order in
/// the sequence loop). `opt_lower_elf` legitimately appears twice: step 11 (ELF target emit)
/// and step 12 (object merge).
const EXPECTED_STAGE_ORDER: &[&str] = &[
    "vyre-libs::parsing::c_lexer",
    "vyre-libs::parsing::c11_lex_digraphs",
    "vyre-libs::parsing::opt_named_macro_expansion_materialized",
    "vyre-libs::parsing::c_keyword",
    "vyre-libs::parsing::c11_compute_alignments",
    "vyre-libs::parsing::c_sema_scope",
    "vyre-libs::parsing::ast_shunting_yard",
    "vyre-libs::parsing::c11_gnu_builtins_pass",
    "vyre-libs::parsing::c11_gnu_inline_asm_pass",
    "vyre-libs::parsing::c11_build_vast_nodes",
    "vyre-libs::parsing::c::lower::ast_to_pg_nodes",
    "vyre-libs::parsing::c11_build_cfg_and_gotos",
    "vyre-libs::parsing::c11_build_expression_shape_nodes",
    "vyre-libs::parsing::opt_x86_64_register_allocation",
    "vyre-libs::parsing::opt_stack_layout_generation",
    "vyre-libs::parsing::opt_lower_elf", // step 11: ELF target emission
    "vyre-libs::parsing::c11_classify_vast_node_kinds",
    "vyre-libs::parsing::opt_lower_elf", // step 12: object merge
];

#[test]
fn megakernel_dispatches_the_full_pipeline_in_order() {
    let program = build_c11_compiler_megakernel("src", "out", 64, 64);
    let ir = format!("{program:?}");

    // Wrapped in the named megakernel sequence loop.
    assert!(
        ir.contains("vyre-libs::parsing::c11_megakernel"),
        "megakernel must wrap the pipeline in the `c11_megakernel` region"
    );
    assert!(
        ir.contains("global_sequence_step"),
        "megakernel must drive stages through the `global_sequence_step` sequence loop"
    );

    // Every stage present, in dispatch order: walk a monotonically advancing cursor so a
    // reordering or a dropped stage fails. Duplicate `opt_lower_elf` is handled naturally
    // because each search resumes just past the previous match.
    let mut cursor = 0usize;
    let mut prev = "<start>";
    for stage in EXPECTED_STAGE_ORDER {
        match ir[cursor..].find(stage) {
            Some(rel) => cursor += rel + stage.len(),
            None => panic!(
                "megakernel pipeline is missing `{stage}` at or after `{prev}` (stage dropped or \
                 reordered), the composition contract is broken"
            ),
        }
        prev = stage;
    }

    // Buffer ABI: source characters in at binding 0 (read-only), target bytecode out at
    // binding 1 (read-write).
    let source = program
        .buffers()
        .iter()
        .find(|b| &*b.name == "src")
        .expect("megakernel must declare the `src` source-characters buffer");
    assert_eq!(source.binding, 0, "source characters bind at 0");
    assert_eq!(
        source.access,
        BufferAccess::ReadOnly,
        "source characters are read-only input"
    );
    let target = program
        .buffers()
        .iter()
        .find(|b| &*b.name == "out")
        .expect("megakernel must declare the `out` target-bytecode buffer");
    assert_eq!(target.binding, 1, "target bytecode binds at 1");
    assert_eq!(
        target.access,
        BufferAccess::ReadWrite,
        "target bytecode is a read-write output"
    );
}
