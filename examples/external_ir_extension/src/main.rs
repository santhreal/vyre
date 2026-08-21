//! External semantic operation and target registration fixture.

use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use vyre::compiler::{
    self, compile_selected_modules, CompileRequest, DeviceFacts, Digest, EmittedTargetModule,
    ExternalFacts, SearchBudget, TargetCompileError, TargetCompiler, TargetPayload,
    TargetPayloadFormat, TargetProfile,
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

    fn compile(&self, artifact: &compiler::Artifact) -> Result<TargetPayload, TargetCompileError> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let operation = OperationRegistry::global().get(OPERATION_ID).ok_or_else(|| {
        format!("{OPERATION_ID} is not in the global operation registry. Fix: keep the `inventory::submit!` registration in this binary; a linker that drops the section drops the operation with it.")
    })?;
    let program = operation.program().ok_or_else(|| {
        format!("{OPERATION_ID} carries no neutral body. Fix: give the registration a program builder so a backend has IR to compile.")
    })?;
    let graph = ProgramGraph::from_program("extension", program)
        .map_err(|error| format!("{OPERATION_ID} does not form a program graph: {error}"))?;
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .map_err(|error| format!("the extension compile request is invalid: {error}"))?;
    let artifact = compiler::compile(&request)
        .map_err(|error| format!("the extension program did not compile: {error}"))?;
    let target = vyre_driver::backend_registration(TARGET_NAME).map_err(|error| {
        format!("{TARGET_NAME} is not a usable registered backend: {error}. Fix: keep the `BackendRegistration` submission in this binary.")
    })?;
    let compiler = target.target_compiler().map_err(|error| {
        format!("{TARGET_NAME} exposes no target compiler: {error}. Fix: set `target_compiler` in its `BackendRegistration`.")
    })?;
    let envelope = compiler::attach_target(artifact, compiler.as_ref())
        .map_err(|error| format!("the target payload was not authenticated: {error}"))?;
    let payload = envelope
        .require_target_payload(compiler.format())
        .map_err(|error| format!("the envelope carries no payload for {TARGET_NAME}: {error}"))?;
    let facets = vyre_driver::registered_target_operation_facets()
        .map_err(|error| format!("the target facet registry is invalid: {error}"))?;
    if !facets
        .iter()
        .any(|facet| facet.operation_id == OPERATION_ID && facet.target_id == TARGET_ID)
    {
        return Err(format!(
            "no facet pairs {OPERATION_ID} with target {}. Fix: register the operation for that target so an external IR extension is reachable from it.",
            TARGET_ID.as_str()
        )
        .into());
    }

    println!(
        "registered {OPERATION_ID} for target {} with {} payload bytes",
        TARGET_ID.as_str(),
        payload.bytes().len()
    );
    Ok(())
}
