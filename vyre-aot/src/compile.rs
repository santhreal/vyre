//! Program to canonical neutral artifact plus attached target payload.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_foundation::ir::{
    inline_calls_with_resolver, BufferAccess, MemoryKind, OpResolver, Program, ProgramGraph,
    ShapeDim, TensorContract, ValueLifetime,
};
use vyre_megakernel::{
    ArtifactRoute, CompileOptions, MegakernelArtifactEnvelope, TargetEntryPoint, TargetPayload,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory, ValidatedCompileRequest,
};

use crate::artifact::{target_payload_format, CompiledArtifact, Target};
use crate::VERSION;

const MAX_NEUTRAL_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

/// Errors returned by [`compile`].
#[derive(Debug, Error)]
pub enum CompileError {
    /// The chosen `Target` is not enabled in this build.
    #[error(
        "vyre-aot: target {0:?} has no linked AOT emitter. Fix: link the concrete driver crate that owns this target."
    )]
    TargetNotEnabled(Target),

    /// The backend rejected the Program with a structured message.
    #[error("vyre-aot: backend rejected Program: {0}")]
    BackendError(String),

    /// The Program cannot be represented accurately in the canonical artifact envelope.
    #[error("vyre-aot: artifact layout rejected Program: {0}")]
    ArtifactLayout(String),

    /// Canonical artifact construction or payload association failed at an AOT stage.
    #[error("vyre-aot: canonical artifact stage `{stage}` failed: {source}")]
    CanonicalArtifact {
        /// AOT stage that failed.
        stage: &'static str,
        /// Structured canonical artifact error, including its exact field path.
        #[source]
        source: vyre_megakernel::CompileError,
    },
}

/// Compile a `Program` into a canonical artifact envelope for a chosen target.
pub fn compile(program: &Program, target: Target) -> Result<CompiledArtifact, CompileError> {
    compile_with_resolver(program, target, None)
}

/// Compile with a caller-supplied resolver to inline `Expr::Call` nodes.
pub fn compile_with_resolver(
    program: &Program,
    target: Target,
    resolver: Option<OpResolver>,
) -> Result<CompiledArtifact, CompileError> {
    let inlined = match resolver {
        Some(resolver) => inline_calls_with_resolver(program, resolver)
            .map_err(|error| CompileError::BackendError(format!("{error:?}")))?,
        None => program.clone(),
    };
    let optimized = vyre_foundation::ir::optimize(inlined);
    let vsa_fingerprint = vyre_driver::program_vsa_fingerprint(&optimized);

    let dispatch = derive_dispatch_grid(&optimized)?;
    let driver_dispatch = vyre_driver::DispatchConfig::default();
    let target_bytes =
        vyre_driver::aot::emit_aot_target(target.aot_target_id(), &optimized, &driver_dispatch)
            .map_err(|error| match error {
                vyre_driver::BackendError::UnsupportedFeature { .. } => {
                    CompileError::TargetNotEnabled(target)
                }
                other => CompileError::BackendError(other.to_string()),
            })?;

    let neutral = compile_neutral_artifact(&optimized)?;
    let entry_node = neutral
        .nodes()
        .iter()
        .find(|node| node.name == "main")
        .map(|node| node.id)
        .ok_or_else(|| {
            CompileError::ArtifactLayout(
                "canonical single-entry graph did not produce its `main` node".to_string(),
            )
        })?;
    let bindings = collect_resource_bindings(&optimized, &neutral)?;
    let entry = TargetEntryPoint {
        name: "main".to_string(),
        node: entry_node,
        grid_size: dispatch,
        dynamic_shared_bytes: 0,
        resource_bindings: bindings,
    };
    let format = target_payload_format(target).map_err(|source| CompileError::CanonicalArtifact {
        stage: "target-format",
        source,
    })?;
    let payload = TargetPayload::new(&neutral, format, vec![entry], target_bytes).map_err(
        |source| CompileError::CanonicalArtifact {
            stage: "target-payload",
            source,
        },
    )?;
    let mut envelope = MegakernelArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .map_err(|source| CompileError::CanonicalArtifact {
            stage: "payload-association",
            source,
        })?;
    CompiledArtifact::new(target, envelope, VERSION, vsa_fingerprint).map_err(|source| {
        CompileError::CanonicalArtifact {
            stage: "package-admission",
            source,
        }
    })
}

fn compile_neutral_artifact(
    program: &Program,
) -> Result<vyre_megakernel::MegakernelArtifact, CompileError> {
    let mut graph = ProgramGraph::new();
    for buffer in program.buffers() {
        graph
            .add_external_value(
                buffer.name(),
                TensorContract {
                    dtype: buffer.element(),
                    shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
                    access: buffer.access(),
                    lifetime: resource_lifetime(buffer.access(), buffer.kind()),
                },
            )
            .map_err(|error| {
                CompileError::ArtifactLayout(format!(
                    "resource `{}` cannot enter the canonical graph: {error}",
                    buffer.name()
                ))
            })?;
    }
    graph
        .add_node("main", program.clone(), Vec::new(), Vec::new())
        .map_err(|error| {
            CompileError::ArtifactLayout(format!(
                "optimized Program cannot enter the canonical graph: {error}"
            ))
        })?;
    let request = ValidatedCompileRequest::new(
        graph,
        CompileOptions::new(
            ArtifactRoute::Static,
            BTreeMap::new(),
            MAX_NEUTRAL_ARTIFACT_BYTES,
        ),
    )
    .map_err(|source| CompileError::CanonicalArtifact {
        stage: "neutral-request",
        source,
    })?;
    vyre_megakernel::compile(&request).map_err(|source| CompileError::CanonicalArtifact {
        stage: "neutral-compile",
        source,
    })
}

fn derive_dispatch_grid(program: &Program) -> Result<[u32; 3], CompileError> {
    let plan = vyre_driver::binding::BindingPlan::build(program)
        .map_err(|error| CompileError::BackendError(error.to_string()))?;
    let element_count =
        vyre_driver::program_walks::dispatch_element_count_for_program(program, &plan.bindings);
    vyre_driver::infer_dispatch_grid_for_count(element_count, program.workgroup_size)
        .map_err(|error| CompileError::BackendError(error.to_string()))
}

fn collect_resource_bindings(
    program: &Program,
    neutral: &vyre_megakernel::MegakernelArtifact,
) -> Result<Vec<TargetResourceBinding>, CompileError> {
    program
        .buffers()
        .iter()
        .map(|buffer| {
            let resource = neutral
                .resources()
                .iter()
                .find(|resource| resource.name == buffer.name())
                .ok_or_else(|| {
                    CompileError::ArtifactLayout(format!(
                        "canonical artifact omitted resource `{}`",
                        buffer.name()
                    ))
                })?;
            Ok(TargetResourceBinding {
                resource: resource.value,
                slot: buffer.binding(),
                memory: convert_memory_kind(buffer.kind()),
                access: convert_access(buffer.name(), buffer.access())?,
            })
        })
        .collect()
}

fn resource_lifetime(access: BufferAccess, memory: MemoryKind) -> ValueLifetime {
    match (access, memory) {
        (BufferAccess::ReadOnly | BufferAccess::Uniform, MemoryKind::Uniform | MemoryKind::Push | MemoryKind::Readonly) => {
            ValueLifetime::ImmutableWeight
        }
        (BufferAccess::WriteOnly, _) => ValueLifetime::Output,
        (BufferAccess::ReadWrite, _) => ValueLifetime::SequenceState,
        _ => ValueLifetime::Invocation,
    }
}

fn convert_memory_kind(kind: MemoryKind) -> TargetResourceMemory {
    match kind {
        MemoryKind::Shared | MemoryKind::Local => TargetResourceMemory::Shared,
        MemoryKind::Uniform | MemoryKind::Push | MemoryKind::Readonly => {
            TargetResourceMemory::Constant
        }
        _ => TargetResourceMemory::Global,
    }
}

fn convert_access(name: &str, access: BufferAccess) -> Result<TargetResourceAccess, CompileError> {
    match access {
        BufferAccess::ReadOnly | BufferAccess::Uniform => Ok(TargetResourceAccess::ReadOnly),
        BufferAccess::WriteOnly => Ok(TargetResourceAccess::WriteOnly),
        BufferAccess::ReadWrite | BufferAccess::Workgroup => Ok(TargetResourceAccess::ReadWrite),
        other => Err(CompileError::ArtifactLayout(format!(
            "buffer `{name}` uses unrecognised access kind {other:?}. Fix: lower it to an explicit target resource access before AOT emission."
        ))),
    }
}

#[cfg(test)]
pub(crate) fn artifact_fixture(program: &Program, target_bytes: Vec<u8>) -> CompiledArtifact {
    let neutral = compile_neutral_artifact(program).expect("test Program must compile neutrally");
    let entry = TargetEntryPoint {
        name: "main".to_string(),
        node: neutral.nodes()[0].id,
        grid_size: derive_dispatch_grid(program).expect("test dispatch must be finite"),
        dynamic_shared_bytes: 0,
        resource_bindings: collect_resource_bindings(program, &neutral)
            .expect("test resources must bind"),
    };
    let payload = TargetPayload::new(
        &neutral,
        target_payload_format(Target::Ptx).unwrap(),
        vec![entry],
        target_bytes,
    )
    .unwrap();
    let mut envelope = MegakernelArtifactEnvelope::new(neutral);
    envelope.attach_target_payload(payload).unwrap();
    CompiledArtifact::new(Target::Ptx, envelope, VERSION, Vec::new()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    #[test]
    fn dispatch_geometry_is_explicit_not_runtime_grid_placeholder() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(1024),
                BufferDecl::read_write("out", 1, DataType::U32).with_count(1024),
            ],
            [128, 1, 1],
            vec![
                Node::let_bind("idx", Expr::u32(0)),
                Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::load("input", Expr::var("idx")),
                ),
            ],
        );

        let dispatch = derive_dispatch_grid(&program)
            .expect("Fix: AOT dispatch grid derivation must accept finite buffer shapes.");

        assert_eq!(
            dispatch,
            [8, 1, 1],
            "Fix: vyre-aot must attach explicit finite grid metadata to its canonical payload."
        );
    }

    #[test]
    fn dispatch_geometry_rejects_zero_workgroup_axes_before_artifact_emission() {
        let program = Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(16)],
            [0, 1, 1],
            vec![
                Node::let_bind("idx", Expr::u32(0)),
                Node::store("out", Expr::var("idx"), Expr::u32(1)),
            ],
        );

        let err = derive_dispatch_grid(&program).expect_err(
            "Fix: AOT must reject zero workgroup axes before target payload emission.",
        );

        assert!(
            err.to_string().contains("workgroup dimensions must be non-zero"),
            "Fix: zero-workgroup AOT rejection must point at the dispatch shape contract, got {err}."
        );
    }

    #[test]
    fn convert_access_uniform_maps_to_read_only_not_read_write() {
        // Before the fix, BufferAccess::Uniform fell through the wildcard arm
        // and silently mapped to ReadWrite (convert-access-wildcard-fallback).
        // Uniform is semantically read-only; mapping it to ReadWrite is a
        // miscompile that wastes memory bandwidth on restricted-access GPU APIs.
        use vyre_foundation::ir::BufferAccess;
        let result = convert_access("my_uniform_buf", BufferAccess::Uniform)
            .expect("Fix: BufferAccess::Uniform must map to ReadOnly, not Err");
        assert_eq!(
            result,
            TargetResourceAccess::ReadOnly,
            "Fix: BufferAccess::Uniform must map to BufferAccessKind::ReadOnly, \
             not ReadWrite. Got {result:?}."
        );
    }

    #[test]
    fn convert_access_workgroup_maps_to_read_write_explicitly_not_via_wildcard() {
        // BufferAccess::Workgroup is workgroup-local shared memory, legitimately
        // ReadWrite. The old code reached this correct result only via the
        // silent wildcard fallback, which also maps future unknown variants
        // silently. This test pins the explicit mapping.
        use vyre_foundation::ir::BufferAccess;
        let result = convert_access("scratch", BufferAccess::Workgroup)
            .expect("Fix: BufferAccess::Workgroup must map to ReadWrite, not Err");
        assert_eq!(
            result,
            TargetResourceAccess::ReadWrite,
            "Fix: BufferAccess::Workgroup must map to BufferAccessKind::ReadWrite. \
             Got {result:?}."
        );
    }

    #[test]
    fn convert_access_all_known_variants_map_deterministically() {
        // All known BufferAccess variants must map to the correct access kind.
        // This test will fail to compile if a new non-exhaustive variant is added
        // to vyre_foundation::ir::BufferAccess but not handled in convert_access.
        use vyre_foundation::ir::BufferAccess;
        let cases = [
            (BufferAccess::ReadOnly, TargetResourceAccess::ReadOnly),
            (BufferAccess::WriteOnly, TargetResourceAccess::WriteOnly),
            (BufferAccess::ReadWrite, TargetResourceAccess::ReadWrite),
            (BufferAccess::Uniform, TargetResourceAccess::ReadOnly),
            (BufferAccess::Workgroup, TargetResourceAccess::ReadWrite),
        ];
        for (access, expected) in cases {
            let got = convert_access("buf", access.clone())
                .unwrap_or_else(|e| panic!("Fix: convert_access({access:?}) returned Err: {e}"));
            assert_eq!(
                got, expected,
                "Fix: convert_access({access:?}) must map to {expected:?}, got {got:?}."
            );
        }
    }
}
