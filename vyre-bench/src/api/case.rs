use crate::api::metric::elapsed_ns;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use serde::{Deserialize, Serialize};
use vyre_driver::{BackendError, BackendRegistration};
use vyre_driver::{DispatchConfig, VyreBackend};
pub use vyre_spec::DeterminismClass;

use super::metric::BenchMetrics;
use super::suite::SuiteKind;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BenchId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchLayer {
    Foundation,
    Reference,
    Runtime,
    Libs,
    Backend,
    Conform,
    Competition,
    Honest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadClass {
    Micro,
    Macro,
    Adversarial,
    Honest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetadata {
    pub id: BenchId,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub layer: BenchLayer,
    pub workload: WorkloadClass,
    pub determinism: DeterminismClass,
    pub owner_crate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineClass {
    CpuSota,
    GpuSota,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTarget {
    pub name: String,
    pub crate_name: String,
    pub class: BaselineClass,
    pub min_speedup_x: f64,
    pub backend_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContract {
    pub primitive: String,
    pub baselines: Vec<BaselineTarget>,
}

impl PerformanceContract {
    pub fn cpu_sota_min_speedup(
        primitive: impl Into<String>,
        crate_name: impl Into<String>,
        baseline: impl Into<String>,
        min_speedup_x: f64,
    ) -> Self {
        Self::cpu_sota_min_speedup_for_backends(
            primitive,
            crate_name,
            baseline,
            min_speedup_x,
            ["cuda", "wgpu"],
        )
    }

    fn cpu_sota_min_speedup_for_backends(
        primitive: impl Into<String>,
        crate_name: impl Into<String>,
        baseline: impl Into<String>,
        min_speedup_x: f64,
        backend_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            primitive: primitive.into(),
            baselines: vec![BaselineTarget {
                name: baseline.into(),
                crate_name: crate_name.into(),
                class: BaselineClass::CpuSota,
                min_speedup_x,
                backend_ids: backend_ids.into_iter().map(Into::into).collect(),
            }],
        }
    }

    pub fn cpu_sota_100x(
        primitive: impl Into<String>,
        crate_name: impl Into<String>,
        baseline: impl Into<String>,
    ) -> Self {
        Self::cpu_sota_min_speedup(primitive, crate_name, baseline, 100.0)
    }

    pub fn cpu_sota_10x(
        primitive: impl Into<String>,
        crate_name: impl Into<String>,
        baseline: impl Into<String>,
    ) -> Self {
        Self::cpu_sota_min_speedup(primitive, crate_name, baseline, 10.0)
    }

    pub fn cpu_sota_3x(
        primitive: impl Into<String>,
        crate_name: impl Into<String>,
        baseline: impl Into<String>,
    ) -> Self {
        Self::cpu_sota_min_speedup(primitive, crate_name, baseline, 3.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEvaluation {
    pub speedup_x: Option<f64>,
    pub contract_passed: bool,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRequirements {
    pub needs_gpu: bool,
    pub needs_network: bool,
    pub min_vram_bytes: Option<u64>,
    pub min_input_bytes: Option<u64>,
    pub feature_set: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Correctness {
    Exact,
    Toleranced {
        ulp_budget: u32,
        max_observed_ulp: u32,
    },
    Certificate {
        digest: [u8; 32],
    },
    Invalid {
        reason: String,
    },
}

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
    sessions: BTreeMap<[u8; 32], Arc<vyre_runtime::artifact_admission::ArtifactSession>>,
    last_fingerprint: Option<[u8; 32]>,
}

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

impl BenchContext {
    pub(crate) fn artifact_session_for(
        &self,
        prog: &vyre::ir::Program,
    ) -> Result<Arc<vyre_runtime::artifact_admission::ArtifactSession>, vyre_driver::BackendError>
    {
        let fingerprint = prog.fingerprint();
        let mut cached = self.artifact_sessions.lock().map_err(|error| {
            vyre_driver::BackendError::new(format!(
                "benchmark artifact session cache is poisoned: {error}. Fix: restart the benchmark process after the panic that poisoned compilation state."
            ))
        })?;
        cached.last_fingerprint = Some(fingerprint);
        if let Some(session) = cached.sessions.get(&fingerprint) {
            return Ok(Arc::clone(session));
        }

        let graph = vyre::ir::ProgramGraph::from_program("benchmark", prog.clone())
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let request = vyre::compiler::CompileRequest::new(
            graph,
            vyre::compiler::ExternalFacts::new(vyre::compiler::Digest([0; 32]), BTreeMap::new()),
            vyre::compiler::SearchBudget::new(256, 100_000, 1, 0, 1_000_000_000),
            64 * 1024 * 1024,
        )
        .validate()
        .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let session = Arc::new(
            vyre_runtime::artifact_admission::ArtifactSession::compile_with_materializer(
                self.preferred_registration,
                &request,
                Arc::clone(&self.materializer),
            )
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?,
        );
        cached.sessions.insert(fingerprint, Arc::clone(&session));
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
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, vyre_driver::BackendError> {
        let _ = dispatch_config_with_inferred_grid(prog, inputs, config)?;
        let session = self.artifact_session_for(prog)?;
        let borrowed_inputs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let completion = session
            .submit_host_inputs(&borrowed_inputs)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        session
            .ordered_outputs(&completion)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))
    }

    pub fn dispatch_timed(
        &self,
        prog: &vyre::ir::Program,
        inputs: &[Vec<u8>],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
        let _ = dispatch_config_with_inferred_grid(prog, inputs, config)?;
        let session = self.artifact_session_for(prog)?;
        let borrowed_inputs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let start = Instant::now();
        let completion = session
            .submit_host_inputs(&borrowed_inputs)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let outputs = session
            .ordered_outputs(&completion)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        Ok(vyre_driver::TimedDispatchResult {
            outputs,
            wall_ns: elapsed_ns(start),
            device_ns: completion.device_ns,
            enqueue_ns: None,
            wait_ns: None,
        })
    }
    pub fn dispatch_resident_timed(
        &self,
        prog: &vyre::ir::Program,
        resources: &[vyre_driver::Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, vyre_driver::BackendError> {
        let session = self.artifact_session_for(prog)?;
        let mut bindings = session
            .resident_bindings(resources)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        if let Some(grid) = config.grid_override {
            bindings.set_invocation_grid(grid)?;
        }
        let start = Instant::now();
        let completion = session
            .submit_and_wait(bindings)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let outputs = session
            .ordered_outputs(&completion)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        Ok(vyre_driver::TimedDispatchResult {
            outputs,
            wall_ns: elapsed_ns(start),
            device_ns: completion.device_ns,
            enqueue_ns: None,
            wait_ns: None,
        })
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
        let mut bindings = session
            .resident_bindings(step.resources)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        if let Some(grid) = step.grid_override {
            bindings.set_invocation_grid(grid)?;
        }
        let completion = session
            .submit_and_wait(bindings.clone())
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        Ok((bindings, completion))
    }
}

fn sum_optional_device_ns(
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

fn copy_typed_read_ranges(
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
    inferred_config.grid_override = Some(vyre_driver::infer_dispatch_grid(
        prog, inputs, config,
    )?);
    Ok(Cow::Owned(inferred_config))
}

pub type PreparedCase = Box<dyn std::any::Any>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRun {
    pub metrics: BenchMetrics,
    pub baseline_metrics: Option<BenchMetrics>,
    pub outputs: Vec<Vec<u8>>,
    pub baseline_outputs: Option<Vec<Vec<u8>>>,
}

impl BenchRun {
    pub fn verify_exact_outputs(&self) -> Result<Correctness, BenchError> {
        let baseline = self.baseline_outputs.as_ref().ok_or_else(|| {
            BenchError::CorrectnessViolation(
                "benchmark did not capture a baseline output; cannot claim exact correctness"
                    .to_string(),
            )
        })?;
        if self.outputs == *baseline {
            return Ok(Correctness::Exact);
        }
        Err(BenchError::CorrectnessViolation(first_output_difference(
            &self.outputs,
            baseline,
        )))
    }

    pub fn verify_f32_outputs_with_ulp(&self, ulp_budget: u32) -> Result<Correctness, BenchError> {
        let baseline = self.baseline_outputs.as_ref().ok_or_else(|| {
            BenchError::CorrectnessViolation(
                "benchmark did not capture a baseline output; cannot claim f32 ULP correctness"
                    .to_string(),
            )
        })?;
        if self.outputs.len() != baseline.len() {
            return Err(BenchError::CorrectnessViolation(format!(
                "f32 output count mismatch: backend returned {}, baseline returned {}",
                self.outputs.len(),
                baseline.len()
            )));
        }

        let mut max_observed_ulp = 0u32;
        for (buffer_index, (actual, expected)) in self.outputs.iter().zip(baseline).enumerate() {
            if actual.len() != expected.len() {
                return Err(BenchError::CorrectnessViolation(format!(
                    "f32 output buffer {buffer_index} length mismatch: backend returned {} bytes, baseline returned {} bytes",
                    actual.len(),
                    expected.len()
                )));
            }
            if actual.len() % 4 != 0 {
                return Err(BenchError::CorrectnessViolation(format!(
                    "f32 output buffer {buffer_index} has non-f32 byte length {}",
                    actual.len()
                )));
            }
            for (value_index, (actual_chunk, expected_chunk)) in actual
                .chunks_exact(4)
                .zip(expected.chunks_exact(4))
                .enumerate()
            {
                let actual_value = f32::from_le_bytes(actual_chunk.try_into().map_err(|_| {
                    BenchError::CorrectnessViolation(
                        "backend f32 output chunk was not 4 bytes".to_string(),
                    )
                })?);
                let expected_value =
                    f32::from_le_bytes(expected_chunk.try_into().map_err(|_| {
                        BenchError::CorrectnessViolation(
                            "baseline f32 output chunk was not 4 bytes".to_string(),
                        )
                    })?);
                let distance = f32_ulp_distance(actual_value, expected_value).ok_or_else(|| {
                    BenchError::CorrectnessViolation(format!(
                        "f32 output buffer {buffer_index} value {value_index} contains NaN: backend={actual_value:?}, baseline={expected_value:?}"
                    ))
                })?;
                max_observed_ulp = max_observed_ulp.max(distance);
                if distance > ulp_budget {
                    return Err(BenchError::CorrectnessViolation(format!(
                        "f32 output buffer {buffer_index} value {value_index} exceeded ULP budget {ulp_budget}: observed {distance}, backend={actual_value:?}, baseline={expected_value:?}"
                    )));
                }
            }
        }
        Ok(Correctness::Toleranced {
            ulp_budget,
            max_observed_ulp,
        })
    }
}

pub fn prepared_program(prepared: &PreparedCase) -> Result<&vyre::ir::Program, BenchError> {
    prepared.downcast_ref::<vyre::ir::Program>().ok_or_else(|| {
        BenchError::ExecutionFailed(
            "prepared benchmark payload was not a vyre::ir::Program".to_string(),
        )
    })
}

/// Borrow a case's prepared payload as its own type.
///
/// Sixteen cases hand-rolled this downcast with the same error wording, each
/// naming itself in the message. `case` is that name, and it is the only thing
/// that varied.
pub fn prepared_as<'a, T: 'static>(
    prepared: &'a PreparedCase,
    case: &str,
) -> Result<&'a T, BenchError> {
    prepared.downcast_ref::<T>().ok_or_else(|| {
        BenchError::ExecutionFailed(format!("{case} prepared payload type mismatch"))
    })
}

/// Borrow a case's prepared payload mutably as its own type.
///
/// The mutable half of `prepared_as`. Splitting the two is what left five
/// cases still hand-rolling the downcast after the read-only ones were
/// collapsed, so both flavours are named here and the wording is shared.
pub fn prepared_as_mut<'a, T: 'static>(
    prepared: &'a mut PreparedCase,
    case: &str,
) -> Result<&'a mut T, BenchError> {
    prepared.downcast_mut::<T>().ok_or_else(|| {
        BenchError::ExecutionFailed(format!("{case} prepared payload type mismatch"))
    })
}

fn first_output_difference(outputs: &[Vec<u8>], baseline: &[Vec<u8>]) -> String {
    if outputs.len() != baseline.len() {
        return format!(
            "output count mismatch: backend returned {}, baseline returned {}",
            outputs.len(),
            baseline.len()
        );
    }
    for (buffer_index, (actual, expected)) in outputs.iter().zip(baseline).enumerate() {
        if actual.len() != expected.len() {
            return format!(
                "output buffer {buffer_index} length mismatch: backend returned {} bytes, baseline returned {} bytes",
                actual.len(),
                expected.len()
            );
        }
        if let Some(byte_index) = actual
            .iter()
            .zip(expected)
            .position(|(actual_byte, expected_byte)| actual_byte != expected_byte)
        {
            let window_end = actual.len().min(byte_index.saturating_add(16));
            return format!(
                "output buffer {buffer_index} differs at byte {byte_index}: backend=0x{:02x}, baseline=0x{:02x}, backend_window={:02x?}, baseline_window={:02x?}",
                actual[byte_index],
                expected[byte_index],
                &actual[byte_index..window_end],
                &expected[byte_index..window_end]
            );
        }
    }
    "backend output differs from baseline".to_string()
}

fn f32_ulp_distance(actual: f32, expected: f32) -> Option<u32> {
    if actual.to_bits() == expected.to_bits() {
        return Some(0);
    }
    if actual.is_nan() || expected.is_nan() {
        return None;
    }
    let actual_ordered = ordered_f32_bits(actual);
    let expected_ordered = ordered_f32_bits(expected);
    Some(
        actual_ordered
            .abs_diff(expected_ordered)
            .min(u64::from(u32::MAX)) as u32,
    )
}

fn ordered_f32_bits(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        i64::from(bits | 0x8000_0000)
    } else {
        i64::from(!bits)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("Environment invalid: {0}")]
    EnvironmentInvalid(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("GPU probe failed for GPU-required benchmark: {0}. Fix: run `nvidia-smi`, verify CUDA/WGPU backend acquisition, and rerun the benchmark.")]
    GpuProbeFailed(String),
    #[error("Backend failed: {0}")]
    BackendFailed(String),
    #[error("Correctness violation: {0}")]
    CorrectnessViolation(String),
}

pub trait BenchCase: Send + Sync {
    fn id(&self) -> BenchId;
    fn metadata(&self) -> BenchMetadata;
    /// The declaration owner this case was built by.
    ///
    /// A benchmark case is workload data plus the few operations data cannot
    /// carry. The trait itself is implemented by the handful of declaration
    /// owners in `cases`, each serving several cases, never once per case: the
    /// per-case copies of the suite list, the metadata record and the measured
    /// loop had already drifted apart by the time they were collapsed. The
    /// default is empty, so a case that open-codes the trait declares no owner
    /// and the declaration gate names it.
    fn declaration_owner(&self) -> &'static str {
        ""
    }
    fn suites(&self) -> &'static [SuiteKind] {
        &[]
    }
    fn active_in_suite(&self, suite: &SuiteKind) -> bool {
        let suites = self.suites();
        suites.is_empty() || suites.contains(suite)
    }
    /// GPU micro-benchmark defaults: needs a device, no network, no size floor.
    ///
    /// A case overrides this only when it has a VRAM floor, an input floor, a
    /// feature set, or no device need at all.
    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: None,
            min_input_bytes: None,
            feature_set: vec![],
        }
    }
    fn performance_contract(&self) -> Option<PerformanceContract> {
        None
    }
    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError>;
    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a vyre::ir::Program> {
        prepared_program(prepared).ok()
    }
    fn workload_fingerprint_bytes(&self, prepared: &PreparedCase) -> Option<[u8; 32]> {
        self.program(prepared).map(vyre::ir::Program::fingerprint)
    }
    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError>;
    fn verify(&self, ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError>;
    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared_program(prepared)
            .map(static_program_bytes_touched)
            .unwrap_or((0, 0))
    }
}

/// Bytes a case reads and writes per sample, from the program's buffer sizes.
pub(crate) fn static_program_bytes_touched(program: &vyre::ir::Program) -> (u64, u64) {
    let mut read_bytes = 0_u64;
    let mut write_bytes = 0_u64;
    for buffer in program.buffers() {
        let bytes = buffer
            .element()
            .size_bytes()
            .map(|element_bytes| (element_bytes as u64).saturating_mul(u64::from(buffer.count())))
            .unwrap_or(0);
        match buffer.access() {
            vyre::ir::BufferAccess::ReadOnly | vyre::ir::BufferAccess::Uniform => {
                read_bytes = read_bytes.saturating_add(bytes);
            }
            vyre::ir::BufferAccess::ReadWrite => {
                read_bytes = read_bytes.saturating_add(bytes);
                write_bytes = write_bytes.saturating_add(bytes);
            }
            vyre::ir::BufferAccess::WriteOnly => {
                write_bytes = write_bytes.saturating_add(bytes);
            }
            vyre::ir::BufferAccess::Workgroup => {}
            _ => {}
        }
    }
    (read_bytes, write_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        vyre_primitives::wire::pack_f32_slice(values)
    }

    #[test]
    fn cpu_sota_contract_applies_to_cuda_and_wgpu_release_backends() {
        let contract = PerformanceContract::cpu_sota_100x("primitive", "vyre", "cpu baseline");
        let backend_ids = &contract.baselines[0].backend_ids;

        for backend in ["cuda", "wgpu"] {
            assert!(
                backend_ids.iter().any(|candidate| candidate == backend),
                "Fix: CPU-SOTA release contracts must apply to `{backend}` evidence."
            );
        }
    }

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

    #[test]
    fn f32_ulp_verifier_accepts_budgeted_difference() {
        let one = 1.0f32;
        let next = f32::from_bits(one.to_bits() + 1);
        let run = BenchRun {
            metrics: BenchMetrics::default(),
            baseline_metrics: None,
            outputs: vec![f32_bytes(&[next])],
            baseline_outputs: Some(vec![f32_bytes(&[one])]),
        };

        assert!(matches!(
            run.verify_f32_outputs_with_ulp(1).unwrap(),
            Correctness::Toleranced {
                ulp_budget: 1,
                max_observed_ulp: 1
            }
        ));
    }

    #[test]
    fn f32_ulp_verifier_rejects_over_budget_difference() {
        let one = 1.0f32;
        let far = f32::from_bits(one.to_bits() + 8);
        let run = BenchRun {
            metrics: BenchMetrics::default(),
            baseline_metrics: None,
            outputs: vec![f32_bytes(&[far])],
            baseline_outputs: Some(vec![f32_bytes(&[one])]),
        };

        let error = run.verify_f32_outputs_with_ulp(2).unwrap_err();
        assert!(
            error.to_string().contains("exceeded ULP budget"),
            "Fix: over-budget f32 mismatch should be actionable: {error}"
        );
    }
}
