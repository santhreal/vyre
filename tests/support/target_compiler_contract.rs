//! The registered-target-compiler contract, owned by the workspace.
//!
//! Every backend that registers a target compiler must satisfy the same four
//! statements: its registration is reachable, its compiler reports the payload
//! format the registration declares, the payload it emits is sealed against the
//! artifact it was asked to compile, and its materializer executes that payload
//! without recompiling anything. Four backends each carried their own copy of
//! all four, and the copies had already drifted: two asserted the payload format
//! version and two did not, three asserted the completion digest and one did
//! not, and each looked its registration up through only one of the two registry
//! routes, so a backend reachable by enumeration but not by id lookup passed.
//!
//! The backend is the parameter. Nothing here names a backend, a shader dialect
//! or a backend-specific rejection string: the id, the format identity and
//! version, the entry point and the output access class arrive as arguments, and
//! whatever is genuinely target-native about a module is inspected by the
//! caller's own closure inside its own crate.
//!
//! Shared the same way as `tests/support/preferred_dispatch_backend_contract.rs`:
//! each consumer includes this file with `#[path]`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use vyre_driver::BackendRegistration;
use vyre_driver::{BindingSet, BoundResource};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphOutput, Node, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    Artifact, ArtifactValueId, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget,
    TargetModuleBundle, TargetPayload,
};

/// Value the fixture program stores into lane zero.
///
/// The program that writes it and the assertion that reads it back were separate
/// literals in every copy, so a fixture edit could silently make the assertion
/// describe the previous program.
const LANE_ZERO: u32 = 1;

/// Canonical output value name, and the buffer the fixture Program declares.
const OUTPUT: &str = "out";

/// The fixture Program: one output buffer, one store of [`LANE_ZERO`] to lane zero.
pub(crate) fn store_one_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output(OUTPUT, 0, DataType::U32).with_count(1)],
        [64, 1, 1],
        vec![Node::store(OUTPUT, Expr::u32(0), Expr::u32(LANE_ZERO))],
    )
}

/// The fixture's single graph output.
///
/// `access` is the one position the backends genuinely differ on: a materializer
/// that reads its result back through a read-write binding declares
/// [`BufferAccess::ReadWrite`], and one that only writes declares
/// [`BufferAccess::WriteOnly`].
fn store_one_output(access: BufferAccess) -> GraphOutput {
    GraphOutput {
        buffer: OUTPUT.into(),
        name: OUTPUT.into(),
        contract: ValueContract {
            dtype: DataType::U32,
            shape: vec![ShapeDim::Known(1)],
            access,
            lifetime: ValueLifetime::Output,
        },
        retained_successor_of: None,
    }
}

/// The canonical single-lane artifact every registered-target-compiler test compiles.
///
/// `facts_seed` participates in the request digest, so two seeds compile to two
/// artifacts that are not authentic for each other. That is how the negative
/// cases obtain a foreign artifact without hand-building a second graph.
pub(crate) fn single_lane_artifact(access: BufferAccess, facts_seed: u8) -> Artifact {
    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "main",
            store_one_program(),
            Vec::new(),
            vec![store_one_output(access)],
        )
        .expect("single-lane fixture graph must accept its one node");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([facts_seed; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .expect("single-lane fixture request must validate");
    vyre_megakernel::compile(&request).expect("single-lane fixture must compile")
}

/// The registration a backend publishes, proven reachable by both registry routes.
///
/// Enumerating the inventory and looking the id up directly are two independent
/// discovery paths, and each caller used only one of them. A backend that
/// answered one and not the other satisfied every test that existed.
pub(crate) fn registration(backend_id: &str) -> &'static BackendRegistration {
    let enumerated = vyre_driver::registered_backends()
        .expect("backend registry must build")
        .iter()
        .find(|candidate| candidate.id == backend_id)
        .unwrap_or_else(|| {
            panic!("Fix: link the driver crate that registers backend `{backend_id}`")
        });
    let by_id = vyre_driver::backend_registration(backend_id).unwrap_or_else(|error| {
        panic!("Fix: backend `{backend_id}` is enumerated but its id lookup failed: {error}")
    });
    assert!(
        std::ptr::eq(enumerated, by_id),
        "Fix: backend `{backend_id}` must resolve to one registration through both registry routes"
    );
    by_id
}

/// What a backend declares about the payload its registered compiler produces.
pub(crate) struct TargetExpectation<'a> {
    /// Registered backend id under test.
    pub backend_id: &'a str,
    /// Payload format identity the registration declares.
    pub format_identity: &'a str,
    /// Payload format version the registration declares.
    pub format_version: u16,
    /// Entry point the emitted module and the payload entry must agree on.
    pub entry_point: &'a str,
    /// Access class this backend's output value contract declares.
    pub output_access: BufferAccess,
}

impl TargetExpectation<'_> {
    /// The fixture artifact this backend compiles.
    pub(crate) fn artifact(&self) -> Artifact {
        single_lane_artifact(self.output_access.clone(), 0)
    }

    /// A fixture artifact this backend's payload must never be authentic for.
    pub(crate) fn foreign_artifact(&self) -> Artifact {
        single_lane_artifact(self.output_access.clone(), 1)
    }

    /// The fixture artifact and the payload this backend's registered compiler emits for it.
    pub(crate) fn compiled(&self) -> (Artifact, TargetPayload) {
        let compiler = registration(self.backend_id)
            .target_compiler()
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: backend `{}` must register a pure target compiler: {error}",
                    self.backend_id
                )
            });
        assert_eq!(
            compiler.format().identity(),
            self.format_identity,
            "Fix: backend `{}` must report the payload format identity it registers",
            self.backend_id
        );
        assert_eq!(
            compiler.format().version(),
            self.format_version,
            "Fix: backend `{}` must report the payload format version it registers",
            self.backend_id
        );
        let artifact = self.artifact();
        let payload = compiler.compile(&artifact).unwrap_or_else(|error| {
            panic!(
                "Fix: backend `{}` must compile the single-lane artifact: {error}",
                self.backend_id
            )
        });
        (artifact, payload)
    }
}

/// WHY: payload production is a pure registered compiler operation. It must not
/// acquire a device, and the payload it returns must be sealed against the exact
/// artifact it was handed, or a later materialization would run a kernel
/// compiled for another program.
///
/// `inspect_module` receives the decoded bundle so the caller can assert what is
/// genuinely target-native about its own module bytes. That stays in the
/// concrete driver crate; this file must never learn a dialect.
pub(crate) fn assert_target_compiler_emits_bundle(
    expectation: &TargetExpectation<'_>,
    inspect_module: impl FnOnce(&TargetModuleBundle),
) {
    let backend_id = expectation.backend_id;
    let (artifact, payload) = expectation.compiled();
    let bundle = TargetModuleBundle::from_bytes(payload.bytes()).unwrap_or_else(|error| {
        panic!("Fix: backend `{backend_id}` must emit a decodable module bundle: {error}")
    });

    assert_eq!(
        bundle.modules.len(),
        artifact.fusion().len(),
        "Fix: backend `{backend_id}` must emit one module per compiler-selected fusion group"
    );
    assert_eq!(
        payload.entries().len(),
        bundle.modules.len(),
        "Fix: backend `{backend_id}` must publish one payload entry per emitted module"
    );
    for (module, entry) in bundle.modules.iter().zip(payload.entries()) {
        assert_eq!(
            module.entry_point, entry.name,
            "Fix: backend `{backend_id}` must name the emitted entry point in its payload entry"
        );
        assert_eq!(
            module.entry_point, expectation.entry_point,
            "Fix: backend `{backend_id}` must emit the neutral entry point admission requires"
        );
    }
    assert_eq!(
        payload.neutral_artifact(),
        artifact.digest(),
        "Fix: backend `{backend_id}` must seal its payload against the compiled artifact"
    );

    inspect_module(&bundle);
}

/// WHY: materialization must load the authenticated payload and execute it, never
/// re-emit the caller's Program. A backend that recompiles here runs code no
/// artifact digest covers.
pub(crate) fn assert_materializer_executes_payload(expectation: &TargetExpectation<'_>) {
    let backend_id = expectation.backend_id;
    let (artifact, payload) = expectation.compiled();
    let materializer = materializer_for(backend_id);
    let instance = materializer
        .materialize(&artifact, &payload)
        .unwrap_or_else(|error| {
            panic!("Fix: backend `{backend_id}` must materialize its own payload: {error}")
        });

    let completion = instance
        .submit(BindingSet::new(artifact.digest()))
        .unwrap_or_else(|error| {
            panic!("Fix: backend `{backend_id}` must accept its own artifact bindings: {error}")
        })
        .wait()
        .unwrap_or_else(|error| {
            panic!("Fix: backend `{backend_id}` submission must retire: {error}")
        });

    assert_eq!(
        completion.artifact,
        artifact.digest(),
        "Fix: backend `{backend_id}` must attribute the completion to the executed artifact"
    );
    assert_lane_zero(backend_id, &completion.outputs);
}

/// WHY: a resident binding must stay inside the authenticated artifact route.
/// The resident hot loop is where a backend is most tempted to bypass
/// materialization and dispatch a raw Program instead.
pub(crate) fn assert_materializer_executes_resident_binding(expectation: &TargetExpectation<'_>) {
    let backend_id = expectation.backend_id;
    let (artifact, payload) = expectation.compiled();
    let materializer = materializer_for(backend_id);
    let instance = materializer
        .materialize(&artifact, &payload)
        .unwrap_or_else(|error| {
            panic!("Fix: backend `{backend_id}` must materialize its own payload: {error}")
        });

    let resource = materializer.allocate_resident(4).unwrap_or_else(|error| {
        panic!("Fix: backend `{backend_id}` must allocate a resident output: {error}")
    });
    let run = (|| {
        materializer.upload_resident(&resource, &0_u32.to_le_bytes())?;
        let mut bindings = BindingSet::new(artifact.digest());
        bindings.insert(
            ArtifactValueId(0),
            BoundResource::Resident(resource.clone()),
        );
        instance.submit(bindings)?.wait()
    })();
    let freed = materializer.free_resident(resource);

    let completion = run.unwrap_or_else(|error| {
        panic!("Fix: backend `{backend_id}` must execute a resident artifact binding: {error}")
    });
    freed.unwrap_or_else(|error| {
        panic!("Fix: backend `{backend_id}` must release its resident output: {error}")
    });
    assert_lane_zero(backend_id, &completion.outputs);
}

/// The device materializer a backend registers.
fn materializer_for(backend_id: &str) -> Box<dyn vyre_driver::ArtifactMaterializer> {
    registration(backend_id)
        .materializer()
        .unwrap_or_else(|error| {
            panic!(
                "Fix: backend `{backend_id}` must acquire its materializer on a host that requires \
                 a live device; a probe failure here is a configuration failure, not a skip: \
                 {error}"
            )
        })
}

/// The fixture's only observable result.
fn assert_lane_zero(backend_id: &str, outputs: &BTreeMap<ArtifactValueId, Vec<u8>>) {
    assert_eq!(
        outputs.get(&ArtifactValueId(0)),
        Some(&LANE_ZERO.to_le_bytes().to_vec()),
        "Fix: backend `{backend_id}` must write the fixture constant into output lane zero"
    );
}
