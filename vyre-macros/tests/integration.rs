#![allow(missing_docs)]

extern crate self as vyre;

mod support;

pub use support::{ir, optimizer};

use vyre_macros::{vyre_ast_registry, vyre_pass};

#[vyre_pass(
    name = "macro_compile_backed_pass",
    requires = ["domtree", "alias"],
    invalidates = ["cfg"],
    phase = "dataflow",
    boundary_class = "abi_preserving",
    requires_caps = ["cuda"],
    preserves_abi = false,
    cost_model_family = "megakernel"
)]
pub struct CompileBackedPass;

crate::define_id_gated_pass_body!(CompileBackedPass);

#[vyre_pass(name = "macro_analyze_always", requires = [], invalidates = [], analyze = "always")]
pub struct AnalyzeAlwaysPass;

crate::define_unchanged_pass_body!(AnalyzeAlwaysPass);

#[vyre_pass(name = "macro_defaulted_pass", requires = [], invalidates = [])]
pub struct DefaultedPass;

crate::define_always_run_pass_body!(DefaultedPass);

vyre_ast_registry! {
    TestExpr {
        Unit,
        Unary(u32),
        Pair { left: u32, right: u32 },
    }
}

vyre_ast_registry! {}

#[test]
fn vyre_pass_expands_to_metadata_analysis_transform_and_inventory_entry() {
    let pass = CompileBackedPass;
    let metadata = optimizer::ProgramPass::metadata(&pass);

    assert_eq!(metadata.name, "macro_compile_backed_pass");
    assert_eq!(metadata.requires, &["domtree", "alias"]);
    assert_eq!(metadata.invalidates, &["cfg"]);
    assert_eq!(metadata.phase, optimizer::PassPhase::Dataflow);
    assert_eq!(
        metadata.boundary_class,
        optimizer::PassBoundaryClass::AbiPreserving
    );
    assert_eq!(metadata.requires_caps, &["cuda"]);
    assert!(!metadata.preserves_abi);
    assert_eq!(
        metadata.cost_model_family,
        optimizer::CostModelFamily::Megakernel
    );

    assert!(!optimizer::ProgramPass::analyze(&pass, &ir::Program { id: 0 }).should_run);
    assert!(optimizer::ProgramPass::analyze(&pass, &ir::Program { id: 7 }).should_run);
    assert!(optimizer::ProgramPass::transform(&pass, ir::Program { id: 7 }).changed);
    assert_eq!(
        optimizer::ProgramPass::fingerprint(&pass, &ir::Program { id: 7 }),
        optimizer::fingerprint_program(&ir::Program { id: 7 })
    );

    let registered = inventory::iter::<optimizer::ProgramPassRegistration>
        .into_iter()
        .any(|registration| registration.metadata.name == "macro_compile_backed_pass");
    assert!(registered);

    let factory_metadata = inventory::iter::<optimizer::ProgramPassRegistration>
        .into_iter()
        .find(|registration| registration.metadata.name == "macro_compile_backed_pass")
        .map(|registration| (registration.factory)().metadata())
        .expect("registered pass factory should instantiate macro_compile_backed_pass");
    assert_eq!(factory_metadata.name, "macro_compile_backed_pass");
}

#[test]
fn vyre_pass_analyze_always_skips_missing_analyze_impl_requirement() {
    let pass = AnalyzeAlwaysPass;
    assert!(optimizer::ProgramPass::analyze(&pass, &ir::Program { id: 0 }).should_run);
}

#[test]
fn vyre_pass_defaults_are_abi_preserving_unknown_metadata() {
    let metadata = optimizer::ProgramPass::metadata(&DefaultedPass);
    support::assert_default_metadata(&metadata, "macro_defaulted_pass");
}

#[test]
fn ast_registry_generates_enum_equality_and_operation_ids() {
    assert_eq!(testexpr_op_id(&TestExpr::Unit), "vyre.testexpr.unit");
    assert_eq!(testexpr_op_id(&TestExpr::Unary(5)), "vyre.testexpr.unary");
    assert_eq!(
        testexpr_op_id(&TestExpr::Pair { left: 1, right: 2 }),
        "vyre.testexpr.pair"
    );

    assert_eq!(TestExpr::Unary(5), TestExpr::Unary(5));
    assert_ne!(TestExpr::Unary(5), TestExpr::Unary(6));
    assert_eq!(
        TestExpr::Pair { left: 1, right: 2 },
        TestExpr::Pair { left: 1, right: 2 }
    );
    assert_ne!(
        TestExpr::Pair { left: 1, right: 2 },
        TestExpr::Pair { left: 2, right: 1 }
    );
}

#[test]
fn ast_registry_accepts_empty_manifest_as_noop() {
    let pass = DefaultedPass;
    assert_eq!(
        optimizer::ProgramPass::metadata(&pass).name,
        "macro_defaulted_pass"
    );
}
