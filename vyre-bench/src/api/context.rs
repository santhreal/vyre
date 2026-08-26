use crate::api::metric::elapsed_ns;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use vyre_driver::{BackendError, BackendRegistration, DispatchConfig, VyreBackend};

pub struct ScratchPool {
    pub buffer: Vec<u8>,
}

pub struct OptimizerPipeline {}

pub struct CpuReference {}

impl CpuReference {
    pub fn dispatch(
        &self,
        prog: &vyre::ir::Program,
        inputs: &[Vec<u8>],
        _config: &vyre_driver::DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, String> {
        let ref_inputs: Vec<vyre_reference::value::Value> = inputs
            .iter()
            .map(|b| vyre_reference::value::Value::Bytes(std::sync::Arc::from(b.clone())))
            .collect();
        vyre_reference::reference_eval(prog, &ref_inputs)
            .map(|values| values.iter().map(|v| v.to_bytes()).collect())
            .map_err(|e| format!("{:?}", e))
    }
}

#[derive(Default)]
pub(crate) struct CachedArtifactSessions {
    sessions: BTreeMap<ArtifactSessionKey, Arc<vyre_runtime::artifact_admission::ArtifactSession>>,
    last_fingerprint: Option<[u8; 32]>,
}

/// A compiled benchmark artifact belongs to one program on one device profile.
/// Keying on the program alone would serve a session compiled for other device
/// facts once a caller repoints the context at another backend.
type ArtifactSessionKey = ([u8; 32], String);

pub struct BenchContext {
    pub preferred_backend: Arc<dyn VyreBackend>,
    pub preferred_registration: &'static BackendRegistration,
    pub materializer: Arc<dyn vyre_driver::ArtifactMaterializer>,
    pub(crate) artifact_sessions: Mutex<CachedArtifactSessions>,
    pub reference: CpuReference,
    pub optimizer: OptimizerPipeline,
    pub scratch: ScratchPool,
    pub rng: rand::rngs::StdRng,
    pub dispatch_config: DispatchConfig,
    pub evolve_candidate: Option<vyre::ir::Program>,
    pub include_baseline_outputs: bool,
}

/// Build the validated compile request for one benchmark program against the
/// device the measurement runs on.
///
/// Capabilities come from the probed backend, so a program that uses subgroup
/// intrinsics compiles on a device that has them. A capability-free request
/// rejects such a program during validation and the case never reaches the
/// device.
pub(crate) fn benchmark_compile_request(
    prog: &vyre::ir::Program,
    profile: vyre_driver::DeviceProfile,
) -> Result<vyre::compiler::ValidatedCompileRequest, vyre_driver::BackendError> {
    let graph = vyre::ir::ProgramGraph::from_program("benchmark", prog.clone())
        .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
    vyre::compiler::CompileRequest::new(
        graph,
        vyre::compiler::ExternalFacts::new(vyre::compiler::Digest([0; 32]), BTreeMap::new()),
        profile.compile_facts(),
        vyre::compiler::SearchBudget::new(256, 100_000, 1, 0, 1_000_000_000),
        64 * 1024 * 1024,
    )
    .validate()
    .map_err(|error| vyre_driver::BackendError::new(error.to_string()))
}

impl BenchContext {
    pub(crate) fn artifact_session_for(
        &self,
        prog: &vyre::ir::Program,
    ) -> Result<Arc<vyre_runtime::artifact_admission::ArtifactSession>, vyre_driver::BackendError>
    {
        let fingerprint = prog.fingerprint();
        let key = (
            fingerprint,
            crate::report::json::benchmark_device_signature(
                self.preferred_backend.device_profile(),
            ),
        );
        let mut cached = self.artifact_sessions.lock().map_err(|error| {
            vyre_driver::BackendError::new(format!(
                "benchmark artifact session cache is poisoned: {error}. Fix: restart the benchmark process after the panic that poisoned compilation state."
            ))
        })?;
        cached.last_fingerprint = Some(fingerprint);
        if let Some(session) = cached.sessions.get(&key) {
            return Ok(Arc::clone(session));
        }

        let request = benchmark_compile_request(prog, self.preferred_backend.device_profile())?;
        let session = Arc::new(
            vyre_runtime::artifact_admission::ArtifactSession::compile_with_materializer(
                self.preferred_registration,
                &request,
                Arc::clone(&self.materializer),
            )
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?,
        );
        cached.sessions.insert(key, Arc::clone(&session));
        Ok(session)
    }
    pub(crate) fn take_artifact_session(
        &self,
    ) -> Result<Option<[u8; 32]>, vyre_driver::BackendError> {
        let mut cached = self.artifact_sessions.lock().map_err(|error| {
            vyre_driver::BackendError::new(format!(
                "benchmark artifact session cache is poisoned: {error}. Fix: restart the benchmark process after the panic that poisoned compilation state."
            ))
        })?;
        let fingerprint = cached.last_fingerprint.take();
        cached.sessions.clear();
        Ok(fingerprint)
    }

    /// Compile and materialize the benchmark artifact outside measured submissions.
    pub fn prepare_artifact(
        &self,
        prog: &vyre::ir::Program,
    ) -> Result<(), vyre_driver::BackendError> {
        self.artifact_session_for(prog).map(|_| ())
    }

    pub fn dispatch(
        &self,
        prog: &vyre::ir::Program,
        inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, vyre_driver::BackendError> {
        let session = self.artifact_session_for(prog)?;
        let borrowed_inputs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let bindings = session
            .host_bindings(&borrowed_inputs)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let completion = session
            .submit_and_wait(bindings)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        session
            .program_outputs(prog, &completion)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))
    }

    pub fn dispatch_timed(
        &self,
        prog: &vyre::ir::Program,
        inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
        let session = self.artifact_session_for(prog)?;
        let borrowed_inputs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let bindings = session
            .host_bindings(&borrowed_inputs)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let start = Instant::now();
        let completion = session
            .submit_and_wait(bindings)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let outputs = session
            .program_outputs(prog, &completion)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        Ok(vyre_driver::TimedDispatchResult::device_timed(
            outputs,
            elapsed_ns(start),
            completion.device_ns,
        ))
    }
    /// Dispatch resident resources and include output readback in wall time.
    pub fn dispatch_resident_timed(
        &self,
        prog: &vyre::ir::Program,
        resources: &[vyre_driver::Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
        self.dispatch_resident_timed_inner(prog, resources, config, true)
    }

    /// Dispatch resident resources while timing execution before output readback.
    pub fn dispatch_resident_execution_timed(
        &self,
        prog: &vyre::ir::Program,
        resources: &[vyre_driver::Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
        self.dispatch_resident_timed_inner(prog, resources, config, false)
    }

    fn dispatch_resident_timed_inner(
        &self,
        prog: &vyre::ir::Program,
        resources: &[vyre_driver::Resource],
        _config: &DispatchConfig,
        include_readback: bool,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
        let session = self.artifact_session_for(prog)?;
        let bindings = session
            .program_resident_bindings(prog, resources)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let start = Instant::now();
        let completion = session
            .submit_and_wait(bindings)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let execution_wall_ns = elapsed_ns(start);
        let outputs = session
            .program_outputs(prog, &completion)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        Ok(vyre_driver::TimedDispatchResult::device_timed(
            outputs,
            if include_readback {
                elapsed_ns(start)
            } else {
                execution_wall_ns
            },
            completion.device_ns,
        ))
    }

    pub fn dispatch_resident_sequence_read_ranges_into(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
        read_ranges: &[vyre_driver::ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), vyre_driver::BackendError> {
        let (_, bindings, completion) = self.submit_resident_steps(steps)?;
        copy_typed_read_ranges(&bindings, &completion, read_ranges, outputs)
    }

    pub fn dispatch_resident_sequence_read_ranges_timed_into(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
        read_ranges: &[vyre_driver::ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<vyre_driver::ResidentSequenceTiming, vyre_driver::BackendError> {
        let started = Instant::now();
        let (device_ns, bindings, completion) = self.submit_resident_steps(steps)?;
        copy_typed_read_ranges(&bindings, &completion, read_ranges, outputs)?;
        Ok(vyre_driver::ResidentSequenceTiming {
            wall_ns: elapsed_ns(started),
            device_ns,
            enqueue_ns: None,
            wait_ns: None,
        })
    }

    pub fn dispatch_resident_repeated_sequence_read_ranges_into(
        &self,
        prefix_steps: &[vyre_driver::ResidentDispatchStep<'_>],
        repeated_steps: &[vyre_driver::ResidentDispatchStep<'_>],
        repeat_count: u32,
        read_ranges: &[vyre_driver::ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), vyre_driver::BackendError> {
        let mut last = None;
        for step in prefix_steps {
            last = Some(self.submit_resident_step(step)?);
        }
        for _ in 0..repeat_count {
            for step in repeated_steps {
                last = Some(self.submit_resident_step(step)?);
            }
        }
        let (bindings, completion) = last.ok_or_else(|| {
            vyre_driver::BackendError::new(
                "resident artifact sequence contains no submissions. Fix: provide a prefix step or a positive repeat count with at least one repeated step.",
            )
        })?;
        copy_typed_read_ranges(&bindings, &completion, read_ranges, outputs)
    }

    fn submit_resident_steps(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
    ) -> Result<
        (
            Option<u64>,
            vyre_driver::BindingSet,
            vyre_driver::Completion,
        ),
        vyre_driver::BackendError,
    > {
        let mut device_ns = Some(0_u64);
        let mut last = None;
        for step in steps {
            let (bindings, completion) = self.submit_resident_step(step)?;
            device_ns = sum_optional_device_ns(device_ns, completion.device_ns)?;
            last = Some((bindings, completion));
        }
        let (bindings, completion) = last.ok_or_else(|| {
            vyre_driver::BackendError::new(
                "resident artifact sequence contains no submissions. Fix: provide at least one resident dispatch step.",
            )
        })?;
        Ok((device_ns, bindings, completion))
    }

    fn submit_resident_step(
        &self,
        step: &vyre_driver::ResidentDispatchStep<'_>,
    ) -> Result<(vyre_driver::BindingSet, vyre_driver::Completion), vyre_driver::BackendError> {
        if let Some(workgroup) = step.workgroup_override {
            if workgroup != step.program.workgroup_size {
                return Err(vyre_driver::BackendError::new(format!(
                    "resident artifact step requested workgroup {workgroup:?}, but its immutable program declares {:?}. Fix: compile the requested workgroup into the program before artifact creation.",
                    step.program.workgroup_size
                )));
            }
        }
        let session = self.artifact_session_for(step.program)?;
        let bindings = session
            .program_resident_bindings(step.program, step.resources)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let completion = session
            .submit_and_wait(bindings.clone())
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        Ok((bindings, completion))
    }
}

pub(crate) fn sum_optional_device_ns(
    total: Option<u64>,
    sample: Option<u64>,
) -> Result<Option<u64>, vyre_driver::BackendError> {
    match (total, sample) {
        (Some(total), Some(sample)) => total
            .checked_add(sample)
            .map(Some)
            .ok_or_else(|| {
                vyre_driver::BackendError::new(
                    "resident artifact sequence device timing overflowed u64. Fix: split the benchmark sequence into smaller measured batches.",
                )
            }),
        _ => Ok(None),
    }
}

pub(crate) fn copy_typed_read_ranges(
    bindings: &vyre_driver::BindingSet,
    completion: &vyre_driver::Completion,
    read_ranges: &[vyre_driver::ResidentReadRange<'_>],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), vyre_driver::BackendError> {
    if read_ranges.len() != outputs.len() {
        return Err(vyre_driver::BackendError::new(format!(
            "resident artifact readback requested {} range(s) for {} output slot(s). Fix: provide exactly one output slot per read range.",
            read_ranges.len(),
            outputs.len()
        )));
    }
    for (range, output) in read_ranges.iter().zip(outputs.iter_mut()) {
        let value = bindings
            .resources()
            .iter()
            .find_map(|(value, bound)| match bound {
                vyre_driver::BoundResource::Resident(resource)
                    if resource == range.resource =>
                {
                    Some(value)
                }
                _ => None,
            })
            .ok_or_else(|| {
                vyre_driver::BackendError::new(
                    "resident artifact readback resource is not bound by the final submission. Fix: read a resource present in the final artifact ABI.",
                )
            })?;
        let bytes = completion
            .outputs
            .get(value)
            .or_else(|| completion.retained.get(value))
            .ok_or_else(|| {
                vyre_driver::BackendError::new(format!(
                    "resident artifact completion omitted value {}. Fix: declare the requested value as output or retained state.",
                    value.0
                ))
            })?;
        let end = range
            .byte_offset
            .checked_add(range.byte_len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                vyre_driver::BackendError::new(format!(
                    "resident artifact readback range {}..{} exceeds {} bytes. Fix: constrain the range to the completed value.",
                    range.byte_offset,
                    range.byte_offset.saturating_add(range.byte_len),
                    bytes.len()
                ))
            })?;
        output.clear();
        output.extend_from_slice(&bytes[range.byte_offset..end]);
    }
    Ok(())
}

/// Return a dispatch config with the benchmark's backend-neutral grid inference applied.
pub fn dispatch_config_with_inferred_grid<'a>(
    prog: &vyre::ir::Program,
    inputs: &[Vec<u8>],
    config: &'a DispatchConfig,
) -> Result<Cow<'a, DispatchConfig>, BackendError> {
    if config.grid_override.is_some() {
        return Ok(Cow::Borrowed(config));
    }

    let mut inferred_config = config.clone();
    inferred_config.grid_override = Some(vyre_driver::infer_dispatch_grid(prog, inputs, config)?);
    Ok(Cow::Owned(inferred_config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_config_infers_grid_from_input_bindings_not_sparse_outputs() {
        let program = vyre::ir::Program::wrapped(
            vec![
                vyre::ir::BufferDecl::output("out_count", 0, vyre::ir::DataType::U32).with_count(1),
                vyre::ir::BufferDecl::storage(
                    "records",
                    1,
                    vyre::ir::BufferAccess::ReadOnly,
                    vyre::ir::DataType::U32,
                )
                .with_count(1024),
            ],
            [256, 1, 1],
            vec![vyre::ir::Node::let_bind(
                "_slot",
                vyre::ir::Expr::atomic_add(
                    "out_count",
                    vyre::ir::Expr::u32(0),
                    vyre::ir::Expr::load("records", vyre::ir::Expr::InvocationId { axis: 0 }),
                ),
            )],
        );
        let inputs = vec![vec![0u8; 1024 * 4]];
        let default_config = DispatchConfig::default();

        let inferred = dispatch_config_with_inferred_grid(&program, &inputs, &default_config)
            .expect("Fix: benchmark dispatch grid inference must handle sparse-output cases.");

        assert_eq!(
            inferred.grid_override,
            Some([4, 1, 1]),
            "Fix: resident sparse-output benchmarks must launch over input records, not the one-word output counter."
        );
    }

    /// WHY: sequence readback must follow canonical artifact value identity rather
    /// than the raw resource's position in one benchmark-specific buffer list.
    #[test]
    fn resident_sequence_readback_uses_typed_artifact_binding() {
        let artifact = vyre::compiler::Digest([7; 32]);
        let value = vyre::compiler::ArtifactValueId(3);
        let resource = vyre_driver::Resource::Borrowed(vec![0; 8]);
        let mut bindings = vyre_driver::BindingSet::new(artifact);
        bindings.insert(
            value,
            vyre_driver::BoundResource::Resident(resource.clone()),
        );
        let completion = vyre_driver::Completion {
            artifact,
            outputs: BTreeMap::from([(value, vec![10, 11, 12, 13, 14])]),
            retained: BTreeMap::new(),
            device_ns: Some(9),
        };
        let range = vyre_driver::ResidentReadRange {
            resource: &resource,
            byte_offset: 1,
            byte_len: 3,
        };
        let mut output = vec![99];

        copy_typed_read_ranges(&bindings, &completion, &[range], &mut [&mut output])
            .expect("typed resident range must resolve through the artifact binding");

        assert_eq!(output, [11, 12, 13]);
    }

    fn subgroup_program() -> vyre::ir::Program {
        use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
        Program::wrapped(
            vec![BufferDecl::storage(
                "out",
                0,
                BufferAccess::ReadWrite,
                DataType::U32,
            )],
            [64, 1, 1],
            vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::subgroup_add(Expr::u32(1)),
            )],
        )
    }

    fn device_profile(subgroup_ops: bool) -> vyre_driver::DeviceProfile {
        let mut profile = vyre_driver::DeviceProfile::conservative("test");
        profile.max_invocations_per_workgroup = 256;
        profile.supports_subgroup_ops = subgroup_ops;
        profile.has_subgroup_shuffle = subgroup_ops;
        profile
    }

    /// WHY: the harness compiled every benchmark program against capability-free
    /// device facts, so any case that uses a subgroup intrinsic failed
    /// validation with V041 on a device that has subgroup ops and was recorded
    /// as a failed case instead of a measurement. The facts must come from the
    /// probed backend.
    #[test]
    fn a_subgroup_program_compiles_against_a_subgroup_capable_device() {
        benchmark_compile_request(&subgroup_program(), device_profile(true))
            .expect("Fix: a subgroup program must validate against a subgroup-capable device");
    }

    /// WHY: the same request must still fail closed on a device without the
    /// capability, so the fix does not turn validation off.
    #[test]
    fn a_subgroup_program_is_refused_on_a_device_without_subgroup_ops() {
        let profile = device_profile(false);
        assert!(!profile.supports_subgroup_ops);

        let Err(error) = benchmark_compile_request(&subgroup_program(), profile) else {
            panic!("Fix: a device without subgroup ops must refuse a subgroup program");
        };
        assert!(
            error.to_string().contains("V041"),
            "Fix: the refusal must name the capability rule: {error}"
        );
    }
}
