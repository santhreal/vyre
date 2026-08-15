//! External semantic operation and target registration fixture.

use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use vyre::compiler::{
    self, compile_selected_modules, CompileRequest, Digest, EmittedTargetModule, ExternalFacts,
    SearchBudget, TargetCompileError, TargetCompiler, TargetPayload, TargetPayloadFormat,
    TargetProfile,
};
use vyre::ir::{BufferDecl, DataType, Expr, Node, OpId, Program, ProgramGraph};
use vyre_driver::{BackendError, BackendRegistration};
use vyre_foundation::operation::{
    OperationRegistration, OperationRegistry, OperationTier, TargetId,
};

const OPERATION_ID: &str = "external_ir_extension::identity";
const TARGET_NAME: &str = "external-ir-fixture";
const TARGET_ID: TargetId = TargetId::expect_valid(TARGET_NAME);

fn build_operation() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    )
}

inventory::submit! {
    OperationRegistration::new(
        OPERATION_ID,
        OperationTier::External,
        Some(build_operation),
        None,
        None,
    )
    .with_category("fixture")
}

struct ExternalTargetCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for ExternalTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(
        &self,
        artifact: &compiler::Artifact,
    ) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            |selected, _profile| {
                let bytes = selected.descriptor.id.as_bytes().to_vec();
                Ok(EmittedTargetModule {
                    entry_point: "external_entry".to_string(),
                    grid_size: [selected.logical_element_count, 1, 1],
                    dynamic_shared_bytes: 0,
                    workgroup_size: selected.descriptor.dispatch.workgroup_size,
                    resource_bindings: selected.canonical_bindings.clone(),
                    bytes,
                })
            },
        )
    }
}

fn unavailable_dispatch() -> Result<Box<dyn vyre_driver::VyreBackend>, BackendError> {
    Err(BackendError::new(
        "external fixture target has no dispatch device. Fix: use its target compiler facet only.",
    ))
}

fn supported_operations() -> &'static HashSet<OpId> {
    static OPERATIONS: LazyLock<HashSet<OpId>> =
        LazyLock::new(|| HashSet::from([OPERATION_ID.into()]));
    &OPERATIONS
}

fn target_compiler() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new(TARGET_NAME, 1)
        .map_err(|error| BackendError::new(format!("target format is invalid: {error}")))?;
    let profile = TargetProfile::new(TARGET_NAME, 1, [1, 1, 1], 1, 0, 0)
        .map_err(|error| BackendError::new(format!("target profile is invalid: {error}")))?;
    Ok(Box::new(ExternalTargetCompiler { format, profile }))
}

inventory::submit! {
    BackendRegistration {
        id: TARGET_NAME,
        target_id: TARGET_ID,
        payload_format: Some(TARGET_NAME),
        reference_oracle: false,
        factory: unavailable_dispatch,
        semantic_operations: supported_operations,
        supported_ops: supported_operations,
        target_compiler: Some(target_compiler),
        materializer: None,
    }
}

fn main() {
    let operation = OperationRegistry::global()
        .get(OPERATION_ID)
        .expect("linked extension operation");
    let program = operation.program().expect("neutral operation body");
    let graph = ProgramGraph::from_program("extension", program).expect("valid extension graph");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .expect("valid extension compile request");
    let artifact = compiler::compile(&request).expect("neutral extension artifact");
    let target = vyre_driver::backend_registration(TARGET_NAME)
        .expect("linked extension target");
    let compiler = target.target_compiler().expect("extension target compiler");
    let envelope =
        compiler::attach_target(artifact, compiler.as_ref()).expect("authenticated target payload");
    let payload = envelope
        .require_target_payload(compiler.format())
        .expect("attached extension payload");
    assert!(
        vyre_driver::registered_target_operation_facets()
            .expect("valid target facet registry")
            .iter()
            .any(|facet| facet.operation_id == OPERATION_ID && facet.target_id == TARGET_ID)
    );

    println!(
        "registered {OPERATION_ID} for target {} with {} payload bytes",
        TARGET_ID.as_str(),
        payload.bytes().len()
    );
}
