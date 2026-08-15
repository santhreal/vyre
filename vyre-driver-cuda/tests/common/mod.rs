//! Shared CUDA integration-test harness.

#![allow(dead_code, unused_imports)]

pub(crate) mod self_optimizer;

use std::sync::Arc;

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use vyre_foundation::MemoryOrdering;
use vyre_reference::value::Value;

/// Default generated-matrix lane count for live CUDA/reference differential tests.
pub(crate) const GENERATED_LANE_COUNT: usize = 512;

/// Default generated-matrix workgroup width for live CUDA/reference differential tests.
pub(crate) const GENERATED_WORKGROUP_SIZE_X: u32 = 128;

/// Pack node ids into canonical little-endian frontier words.
pub(crate) fn pack_nodes(nodes: &[u32], node_count: u32) -> Vec<u32> {
    let mut words = vec![0; node_count.div_ceil(32).max(1) as usize];
    for &node in nodes {
        words[node as usize / 32] |= 1 << (node % 32);
    }
    words
}

/// CUDA-backed optimizer dispatcher used by parity and self-optimizer tests.
pub(crate) struct CudaProgramDispatcher<'a> {
    /// Live CUDA backend borrowed for the duration of one test.
    pub(crate) backend: &'a CudaBackend,
}

impl<'a> ProgramDispatcher for CudaProgramDispatcher<'a> {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut config = DispatchConfig::default();
        config.grid_override = grid_override;
        self.backend
            .dispatch(program, inputs, &config)
            .map_err(|err| DispatchError::BackendError(err.to_string()))
    }
}

/// Acquire the live CUDA backend required by release-path GPU tests.
pub(crate) fn live_dispatcher() -> CudaBackend {
    CudaBackend::acquire().expect(
        "CudaBackend::acquire failed on a host that must have an NVIDIA GPU. \
         Fix: inspect driver visibility and adapter probing; live GPU tests must not silently skip.",
    )
}

/// Acquire the live CUDA backend for self-optimizer tests that use backend naming.
pub(crate) fn live_backend() -> CudaBackend {
    live_dispatcher()
}

/// Run a closure with the live CUDA backend required by release-path GPU tests.
pub(crate) fn with_live_backend<R>(_test_name: &str, run: impl FnOnce(&CudaBackend) -> R) -> R {
    let backend = live_dispatcher();
    run(&backend)
}

/// Run a closure with a live CUDA-backed optimizer dispatcher.
///
/// The backend must outlive the dispatcher, so this helper centralizes the
/// acquisition/lifetime pattern used by CUDA self-substrate parity tests.
pub(crate) fn with_cuda_optimizer_dispatcher<R>(
    _test_name: &str,
    run: impl FnOnce(&CudaProgramDispatcher<'_>) -> R,
) -> R {
    let backend = live_dispatcher();
    let dispatcher = CudaProgramDispatcher { backend: &backend };
    run(&dispatcher)
}

/// Run the pure Rust reference interpreter for byte-buffer CUDA test inputs.
pub(crate) fn reference_outputs(
    program: &Program,
    inputs: &[Vec<u8>],
    case_name: &str,
) -> Vec<Vec<u8>> {
    let values = inputs
        .iter()
        .map(|input| Value::Bytes(Arc::from(input.clone().into_boxed_slice())))
        .collect::<Vec<_>>();
    vyre_reference::reference_eval(program, &values)
        .unwrap_or_else(|error| {
            panic!("Fix: reference CUDA test case `{case_name}` failed: {error}")
        })
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Compile and dispatch through the authenticated CUDA artifact route.
pub(crate) fn compiled_cuda_outputs(
    backend: &CudaBackend,
    program: &Program,
    inputs: &[Vec<u8>],
    case_name: &str,
) -> Vec<Vec<u8>> {
    compiled_cuda_outputs_with_config(
        backend,
        program,
        inputs,
        &DispatchConfig::default(),
        case_name,
    )
}

/// Compile and dispatch through the authenticated CUDA artifact route with explicit geometry.
pub(crate) fn compiled_cuda_outputs_with_config(
    _backend: &CudaBackend,
    program: &Program,
    inputs: &[Vec<u8>],
    config: &DispatchConfig,
    case_name: &str,
) -> Vec<Vec<u8>> {
    let graph =
        vyre::ir::ProgramGraph::from_program(case_name, program.clone()).unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` graph failed: {error}")
        });
    let request = vyre_megakernel::CompileRequest::new(
        graph,
        vyre_megakernel::ExternalFacts::new(
            vyre_megakernel::Digest([0; 32]),
            std::collections::BTreeMap::new(),
        ),
        vyre_megakernel::SearchBudget::new(128, 128, 0, 0, 128),
        60_000,
    )
    .validate()
    .unwrap_or_else(|error| {
        panic!("Fix: CUDA generated case `{case_name}` compile request failed: {error}")
    });
    let artifact = vyre_megakernel::compile(&request).unwrap_or_else(|error| {
        panic!("Fix: CUDA generated case `{case_name}` compiler failed: {error}")
    });
    let registration = vyre_driver::backend_registration(vyre_driver_cuda::CUDA_BACKEND_ID)
        .unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` registration failed: {error}")
        });
    let compiler = registration.target_compiler().unwrap_or_else(|error| {
        panic!("Fix: CUDA generated case `{case_name}` target compiler failed: {error}")
    });
    let envelope =
        vyre_megakernel::attach_target(artifact, compiler.as_ref()).unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` target attachment failed: {error}")
        });
    let materializer = registration.materializer().unwrap_or_else(|error| {
        panic!("Fix: CUDA generated case `{case_name}` materializer acquisition failed: {error}")
    });
    let payload = envelope
        .target_payloads()
        .first()
        .expect("Fix: CUDA target attachment must produce one payload");
    let instance = materializer
        .materialize(envelope.neutral(), payload)
        .unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` materialization failed: {error}")
        });
    let plan = vyre_driver::BindingPlan::build(program).unwrap_or_else(|error| {
        panic!("Fix: CUDA generated case `{case_name}` binding plan failed: {error}")
    });
    let mut bindings = vyre_driver::BindingSet::new(envelope.neutral().digest());
    for binding in &plan.bindings {
        let Some(input_index) = binding.input_index else {
            continue;
        };
        let resource = envelope
            .neutral()
            .resources()
            .iter()
            .find(|resource| resource.name == binding.name.as_ref())
            .unwrap_or_else(|| {
                panic!(
                    "Fix: CUDA generated case `{case_name}` artifact omitted input `{}`",
                    binding.name
                )
            });
        bindings.insert(
            resource.value,
            vyre_driver::BoundResource::Host(inputs[input_index].clone()),
        );
    }
    if let Some(grid) = config.grid_override.or(config.dispatch_grid) {
        bindings.set_invocation_grid(grid).unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` invocation grid failed: {error}")
        });
    }
    let completion = instance
        .submit(bindings)
        .and_then(|submission| submission.wait())
        .unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` artifact submission failed: {error}")
        });
    let mut outputs = vec![Vec::new(); plan.output_indices.len()];
    for binding in &plan.bindings {
        let Some(output_index) = binding.output_index else {
            continue;
        };
        let resource = envelope
            .neutral()
            .resources()
            .iter()
            .find(|resource| resource.name == binding.name.as_ref())
            .unwrap_or_else(|| {
                panic!(
                    "Fix: CUDA generated case `{case_name}` artifact omitted output `{}`",
                    binding.name
                )
            });
        outputs[output_index] = completion
            .outputs
            .get(&resource.value)
            .or_else(|| completion.retained.get(&resource.value))
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "Fix: CUDA generated case `{case_name}` completion omitted writable value `{}`",
                    binding.name
                )
            });
    }
    outputs
}

/// Outputs from one generated CUDA/reference matrix case.
pub(crate) struct GeneratedCudaReferenceOutputs {
    pub(crate) direct_cuda: Vec<Vec<u8>>,
    pub(crate) compiled_cuda: Vec<Vec<u8>>,
    pub(crate) reference: Vec<Vec<u8>>,
}

/// Outputs from one generated CUDA resident/reference matrix case.
pub(crate) struct GeneratedResidentCudaReferenceOutputs {
    pub(crate) resident_cuda: Vec<Vec<u8>>,
    pub(crate) reference: Vec<Vec<u8>>,
}

/// Run one generated matrix case through direct CUDA, compiled CUDA, and reference paths.
pub(crate) fn cuda_reference_outputs(
    backend: &CudaBackend,
    program: &Program,
    inputs: &[Vec<u8>],
    case_name: &str,
) -> GeneratedCudaReferenceOutputs {
    cuda_reference_outputs_with_config(
        backend,
        program,
        inputs,
        &DispatchConfig::default(),
        case_name,
    )
}

/// Run one generated matrix case through direct CUDA, compiled CUDA, and reference paths with explicit config.
pub(crate) fn cuda_reference_outputs_with_config(
    backend: &CudaBackend,
    program: &Program,
    inputs: &[Vec<u8>],
    config: &DispatchConfig,
    case_name: &str,
) -> GeneratedCudaReferenceOutputs {
    let direct_cuda = backend
        .dispatch(program, inputs, config)
        .unwrap_or_else(|error| {
            panic!("Fix: CUDA generated case `{case_name}` direct dispatch failed: {error}")
        });
    let compiled_cuda =
        compiled_cuda_outputs_with_config(backend, program, inputs, config, case_name);
    let reference = reference_outputs(program, inputs, case_name);
    GeneratedCudaReferenceOutputs {
        direct_cuda,
        compiled_cuda,
        reference,
    }
}

/// Run one generated matrix case through CUDA-resident buffers and the Rust reference path.
pub(crate) fn resident_cuda_reference_outputs(
    backend: &CudaBackend,
    program: &Program,
    inputs: &[Vec<u8>],
    output_byte_lens: &[usize],
    case_name: &str,
) -> GeneratedResidentCudaReferenceOutputs {
    let mut handles = Vec::with_capacity(inputs.len() + output_byte_lens.len());
    for (index, input) in inputs.iter().enumerate() {
        let handle = backend.allocate_resident(input.len()).unwrap_or_else(|error| {
            panic!(
                "Fix: CUDA resident generated case `{case_name}` input {index} allocation failed: {error}"
            )
        });
        backend.upload_resident(handle, input).unwrap_or_else(|error| {
            panic!(
                "Fix: CUDA resident generated case `{case_name}` input {index} upload failed: {error}"
            )
        });
        handles.push(handle);
    }
    let output_start = handles.len();
    for (index, &byte_len) in output_byte_lens.iter().enumerate() {
        let handle = backend.allocate_resident(byte_len).unwrap_or_else(|error| {
            panic!(
                "Fix: CUDA resident generated case `{case_name}` output {index} allocation failed: {error}"
            )
        });
        handles.push(handle);
    }

    backend
        .dispatch_resident(program, &handles, &DispatchConfig::default())
        .unwrap_or_else(|error| {
            panic!("Fix: CUDA resident generated case `{case_name}` dispatch failed: {error}")
        });

    let mut resident_cuda = Vec::with_capacity(output_byte_lens.len());
    for (index, &handle) in handles[output_start..].iter().enumerate() {
        resident_cuda.push(backend.download_resident(handle).unwrap_or_else(|error| {
            panic!(
                "Fix: CUDA resident generated case `{case_name}` output {index} download failed: {error}"
            )
        }));
    }
    for handle in handles {
        backend.free_resident(handle).unwrap_or_else(|error| {
            panic!("Fix: CUDA resident generated case `{case_name}` cleanup failed: {error}")
        });
    }
    let reference = reference_outputs(program, inputs, case_name);
    GeneratedResidentCudaReferenceOutputs {
        resident_cuda,
        reference,
    }
}

/// Decode CUDA output bytes into little-endian `f32` lanes.
pub(crate) use vyre_primitives::wire::decode_f32_le_bytes_all as bytes_f32;
/// Pack little-endian `f32` lanes into the byte buffers expected by CUDA dispatch.
pub(crate) use vyre_primitives::wire::pack_f32_slice as f32_bytes;
/// Pack little-endian `i32` lanes into the byte buffers expected by CUDA dispatch.
pub(crate) use vyre_primitives::wire::pack_i32_slice as i32_bytes;
/// Pack little-endian `u16` lanes into the byte buffers expected by CUDA dispatch.
pub(crate) use vyre_primitives::wire::pack_u16_slice as u16_bytes;
/// Pack little-endian `u32` lanes into the byte buffers expected by CUDA dispatch.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

/// Pack Bool lanes using the stable CUDA storage ABI: one little-endian u32 word per lane.
pub(crate) fn bool_bytes(values: &[bool]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for &value in values {
        let word = u32::from(value);
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Decode CUDA output bytes into little-endian `u32` lanes.
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as bytes_u32;

/// Dispatch a one-input u32 program whose single output is a packed bitset.
///
/// This is the canonical CUDA predicate-parity shape: one u32 input buffer,
/// one zero-initialized bitset output buffer, and a grid sized directly from
/// the logical lane count. Keeping it here prevents predicate tests from
/// drifting on grid math or output truncation.
pub(crate) fn cuda_u32_bitset_output(
    backend: &CudaBackend,
    program: &Program,
    lanes: u32,
    input_words: &[u32],
    case_name: &str,
) -> Vec<u32> {
    let output_words = lanes.div_ceil(32).max(1);
    let inputs = vec![
        u32_bytes(input_words),
        vec![0u8; output_words as usize * std::mem::size_of::<u32>()],
    ];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = lanes.div_ceil(workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = backend
        .dispatch(program, &inputs, &config)
        .unwrap_or_else(|error| panic!("Fix: CUDA predicate case `{case_name}` failed: {error}"));
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(output_words as usize);
    out
}

/// Materialize a Bool expression into the stable generated-test u32 oracle word.
pub(crate) fn bool_word(value: Expr) -> Expr {
    Expr::select(value, Expr::u32(1), Expr::u32(0))
}

/// Materialize a binary comparison into the stable generated-test u32 oracle word.
pub(crate) fn compare_word(lhs: Expr, rhs: Expr, compare: fn(Expr, Expr) -> Expr) -> Expr {
    bool_word(compare(lhs, rhs))
}

pub(crate) fn eq_word(lhs: Expr, rhs: Expr) -> Expr {
    compare_word(lhs, rhs, Expr::eq)
}

pub(crate) fn ne_word(lhs: Expr, rhs: Expr) -> Expr {
    compare_word(lhs, rhs, Expr::ne)
}

pub(crate) fn lt_word(lhs: Expr, rhs: Expr) -> Expr {
    compare_word(lhs, rhs, Expr::lt)
}

pub(crate) fn le_word(lhs: Expr, rhs: Expr) -> Expr {
    compare_word(lhs, rhs, Expr::le)
}

pub(crate) fn gt_word(lhs: Expr, rhs: Expr) -> Expr {
    compare_word(lhs, rhs, Expr::gt)
}

pub(crate) fn ge_word(lhs: Expr, rhs: Expr) -> Expr {
    compare_word(lhs, rhs, Expr::ge)
}

/// A generated-matrix program: `reads` bound at 0..n, one `out` buffer last,
/// every buffer at the generated lane count and the generated workgroup width.
///
/// Binding order is the contract the reference interpreter and both CUDA paths
/// agree on, so it is fixed here rather than restated per matrix file.
pub(crate) fn generated_lane_program(
    reads: &[(&str, DataType)],
    output: DataType,
    body: Vec<Node>,
) -> Program {
    let lanes = GENERATED_LANE_COUNT as u32;
    let mut buffers: Vec<BufferDecl> = reads
        .iter()
        .enumerate()
        .map(|(binding, (name, ty))| {
            BufferDecl::read(*name, binding as u32, ty.clone()).with_count(lanes)
        })
        .collect();
    buffers.push(BufferDecl::output("out", reads.len() as u32, output).with_count(lanes));
    Program::wrapped(buffers, [GENERATED_WORKGROUP_SIZE_X, 1, 1], body)
}

/// Store `value` at the lane index in `out`, guarded so a launch rounded up to
/// the workgroup width writes nothing past the declared lane count.
pub(crate) fn guarded_generated_store(value: Expr) -> Vec<Node> {
    guarded_generated_store_at(Expr::var("idx"), value)
}

/// [`guarded_generated_store`] writing to `index` instead of the lane index, for
/// the permutation matrices whose whole point is a nonidentity destination.
pub(crate) fn guarded_generated_store_at(index: Expr, value: Expr) -> Vec<Node> {
    vec![
        Node::let_bind("idx", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("idx"), Expr::u32(GENERATED_LANE_COUNT as u32)),
            vec![Node::store("out", index, value)],
        ),
    ]
}

/// One case in a generated CUDA/reference matrix sweep.
pub(crate) struct GeneratedMatrixCase {
    /// Case name, reported by every diagnostic the sweep emits.
    pub(crate) name: &'static str,
    /// Program under test.
    pub(crate) program: Program,
    /// Bindings in declaration order.
    pub(crate) inputs: Vec<Vec<u8>>,
}

/// Diff every case against the reference interpreter on both the direct and the
/// compiled CUDA path, then prove the sweep compared every lane of every case.
///
/// The lane total is asserted rather than reported because
/// [`assert_u32_output_lanes`] returns the number of lanes it actually compared:
/// handed a truncated or empty output it compares zero and returns zero, which
/// would leave a sweep green while checking nothing.
pub(crate) fn assert_u32_matrix_sweep(
    backend: &CudaBackend,
    sweep: &str,
    coverage: &str,
    cases: impl ExactSizeIterator<Item = GeneratedMatrixCase>,
) {
    let expected_lanes = cases.len() * GENERATED_LANE_COUNT * 2;
    let mut checked_lanes = 0usize;
    for case in cases {
        let outputs = cuda_reference_outputs(backend, &case.program, &case.inputs, case.name);
        checked_lanes += assert_u32_output_lanes(
            case.name,
            GENERATED_LANE_COUNT,
            &outputs.direct_cuda,
            &outputs.reference,
        );
        checked_lanes += assert_u32_output_lanes(
            case.name,
            GENERATED_LANE_COUNT,
            &outputs.compiled_cuda,
            &outputs.reference,
        );
    }
    assert_eq!(
        checked_lanes, expected_lanes,
        "Fix: generated CUDA {sweep} matrix must keep {coverage} across direct and compiled paths."
    );
}

/// [`assert_u32_matrix_sweep`] for f32 outputs, compared with the strict edge
/// semantics [`assert_f32_output_lanes`] fixes at `max_ulp`.
pub(crate) fn assert_f32_matrix_sweep(
    backend: &CudaBackend,
    sweep: &str,
    coverage: &str,
    max_ulp: u32,
    cases: impl ExactSizeIterator<Item = GeneratedMatrixCase>,
) {
    let expected_lanes = cases.len() * GENERATED_LANE_COUNT * 2;
    let mut checked_lanes = 0usize;
    for case in cases {
        let outputs = cuda_reference_outputs(backend, &case.program, &case.inputs, case.name);
        checked_lanes += assert_f32_output_lanes(
            case.name,
            GENERATED_LANE_COUNT,
            max_ulp,
            &outputs.direct_cuda,
            &outputs.reference,
        );
        checked_lanes += assert_f32_output_lanes(
            case.name,
            GENERATED_LANE_COUNT,
            max_ulp,
            &outputs.compiled_cuda,
            &outputs.reference,
        );
    }
    assert_eq!(
        checked_lanes, expected_lanes,
        "Fix: generated CUDA {sweep} matrix must keep {coverage} across direct and compiled paths."
    );
}

/// Adversarial u32 corpus shared by generated cast/FMA matrices.
pub(crate) fn generated_u32_cast_values(lane_count: usize) -> Vec<u32> {
    (0..lane_count)
        .map(|lane| {
            let lane = lane as u32;
            match lane % 16 {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 127,
                4 => 128,
                5 => 255,
                6 => 1024,
                7 => 0x7fff_ffff,
                8 => 0x8000_0000,
                9 => u32::MAX,
                10 => 0x5555_5555,
                11 => 0xaaaa_aaaa,
                _ => lane.wrapping_mul(0x9e37_79b9).rotate_left((lane & 31) + 1),
            }
        })
        .collect()
}

/// Adversarial i32 corpus shared by generated cast/FMA matrices.
pub(crate) fn generated_i32_cast_values(lane_count: usize) -> Vec<i32> {
    generated_u32_cast_values(lane_count)
        .into_iter()
        .enumerate()
        .map(|(lane, word)| match lane % 14 {
            0 => 0,
            1 => 1,
            2 => -1,
            3 => 127,
            4 => -128,
            5 => 1024,
            6 => -1024,
            7 => i32::MAX,
            8 => i32::MIN,
            _ => word as i32,
        })
        .collect()
}

/// Adversarial f32 cast corpus shared by generated cast/FMA matrices.
pub(crate) fn generated_f32_cast_values(lane_count: usize) -> Vec<f32> {
    const BITS: &[u32] = &[
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x4000_0000,
        0xc000_0000,
        0x42fe_0000,
        0xc2fe_0000,
        0x4eff_ffff,
        0xceff_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
    ];
    (0..lane_count)
        .map(|lane| f32::from_bits(BITS[lane % BITS.len()]))
        .collect()
}

/// Adversarial Bool corpus shared by generated cast/FMA matrices.
pub(crate) fn generated_bool_cast_values(lane_count: usize) -> Vec<bool> {
    (0..lane_count)
        .map(|lane| {
            let lane = lane as u32;
            matches!(
                lane.wrapping_mul(0x045d_9f3b).rotate_left(lane & 7) & 0b1011,
                0b0001 | 0b0011 | 0b1001
            )
        })
        .collect()
}

/// Adversarial u32 corpus shared by resident generated matrices.
pub(crate) fn generated_mixed_u32_values(salt: u32) -> Vec<u32> {
    (0..GENERATED_LANE_COUNT)
        .map(|lane| {
            let lane = lane as u32;
            let mixed = lane.wrapping_mul(0x9e37_79b9).rotate_left((lane & 31) + 1)
                ^ salt.rotate_right(lane & 31);
            match lane % 16 {
                0 => 0,
                1 => 1,
                2 => u32::MAX,
                3 => 0x8000_0000,
                4 => 0x7fff_ffff,
                5 => 0x5555_5555,
                6 => 0xaaaa_aaaa,
                7 => 0x0123_4567,
                _ => mixed,
            }
        })
        .collect()
}

/// Adversarial Bool corpus shared by generated control and resident matrices.
pub(crate) fn generated_mixed_bool_values(salt: u32) -> Vec<bool> {
    (0..GENERATED_LANE_COUNT)
        .map(|lane| {
            let lane = lane as u32;
            let mixed = lane.wrapping_mul(0x045d_9f3b).rotate_left((lane & 7) + 1)
                ^ salt.rotate_right(lane & 31);
            (mixed & 0b1011) == 0b0001 || lane % 13 == 0
        })
        .collect()
}

/// Nonzero, noncontiguous word-sized output ranges for compact readback tests.
pub(crate) fn compact_word_ranges() -> [(usize, usize); 4] {
    let word = std::mem::size_of::<u32>();
    [
        (word, word),
        ((GENERATED_LANE_COUNT / 3) * word, 2 * word),
        ((GENERATED_LANE_COUNT / 2) * word, word),
        ((GENERATED_LANE_COUNT - 1) * word, word),
    ]
}

/// Overlapping and adjacent word-sized output ranges for fused readback tests.
pub(crate) fn overlapping_word_ranges() -> [(usize, usize); 4] {
    let word = std::mem::size_of::<u32>();
    [
        (0, 4 * word),
        (2 * word, 4 * word),
        (6 * word, 2 * word),
        ((GENERATED_LANE_COUNT - 2) * word, 2 * word),
    ]
}

/// Assert compact byte ranges match the same slices from a full reference output.
pub(crate) fn assert_compact_ranges_match(
    case_name: &str,
    actual: &[Vec<u8>],
    expected: &[u8],
    ranges: &[(usize, usize)],
) {
    assert_eq!(
        actual.len(),
        ranges.len(),
        "Fix: {case_name} must return one compact output buffer per requested range."
    );
    for (index, ((byte_offset, byte_len), bytes)) in ranges.iter().zip(actual.iter()).enumerate() {
        let end = byte_offset + byte_len;
        assert!(
            end <= expected.len(),
            "Fix: {case_name} range {index} exceeds reference output: {byte_offset}..{end} over {} bytes.",
            expected.len()
        );
        assert_eq!(
            bytes.len(),
            *byte_len,
            "Fix: {case_name} range {index} must compact exactly {byte_len} byte(s)."
        );
        assert_eq!(
            bytes.as_slice(),
            &expected[*byte_offset..end],
            "Fix: {case_name} compact range {index} must match the reference bytes."
        );
    }
}

/// Adversarial f32 FMA corpus shared by generated cast/FMA matrices.
pub(crate) fn generated_f32_fma_values(lane_count: usize, salt: u32) -> Vec<f32> {
    (0..lane_count)
        .map(|lane| {
            let lane = lane as u32;
            let bits = match lane % 12 {
                0 => 0x0000_0000,
                1 => 0x8000_0000,
                2 => 0x3f80_0000,
                3 => 0xbf80_0000,
                4 => 0x4000_0000,
                5 => 0xc000_0000,
                6 => 0x3f00_0000,
                7 => 0xbf00_0000,
                _ => (lane.wrapping_mul(0x0101_0101) ^ salt).rotate_left(lane & 15) & 0x7f7f_ffff,
            };
            f32::from_bits(bits)
        })
        .collect()
}

/// Assert one u32 output buffer matches the reference lane-for-lane.
pub(crate) fn assert_u32_output_lanes(
    case_name: &str,
    lane_count: usize,
    cuda_outputs: &[Vec<u8>],
    reference_outputs: &[Vec<u8>],
) -> usize {
    assert_eq!(
        cuda_outputs.len(),
        1,
        "Fix: CUDA generated case `{case_name}` must return exactly one output buffer."
    );
    assert_eq!(
        reference_outputs.len(),
        1,
        "Fix: reference generated case `{case_name}` must return exactly one output buffer."
    );
    let actual = bytes_u32(&cuda_outputs[0]);
    let expected = bytes_u32(&reference_outputs[0]);
    assert_eq!(
        actual.len(),
        lane_count,
        "Fix: CUDA generated case `{case_name}` output lane count changed."
    );
    assert_eq!(
        expected.len(),
        lane_count,
        "Fix: reference generated case `{case_name}` output lane count changed."
    );
    for lane in 0..lane_count {
        assert_eq!(
            actual[lane], expected[lane],
            "Fix: CUDA generated case `{case_name}` lane {lane} diverged from reference."
        );
    }
    lane_count
}

/// Assert one f32 output buffer matches the reference with strict edge semantics.
pub(crate) fn assert_f32_output_lanes(
    case_name: &str,
    lane_count: usize,
    max_ulp: u32,
    cuda_outputs: &[Vec<u8>],
    reference_outputs: &[Vec<u8>],
) -> usize {
    assert_eq!(
        cuda_outputs.len(),
        1,
        "Fix: CUDA f32 generated case `{case_name}` must return exactly one output buffer."
    );
    assert_eq!(
        reference_outputs.len(),
        1,
        "Fix: reference f32 generated case `{case_name}` must return exactly one output buffer."
    );
    let actual = bytes_f32(&cuda_outputs[0]);
    let expected = bytes_f32(&reference_outputs[0]);
    assert_eq!(actual.len(), lane_count);
    assert_eq!(expected.len(), lane_count);
    for lane in 0..lane_count {
        assert_f32_close(case_name, lane, max_ulp, actual[lane], expected[lane]);
    }
    lane_count
}

fn assert_f32_close(case_name: &str, lane: usize, max_ulp: u32, actual: f32, expected: f32) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "Fix: CUDA f32 generated case `{case_name}` lane {lane} expected NaN, got {actual:?}."
        );
        return;
    }
    if expected == 0.0 {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Fix: CUDA f32 generated case `{case_name}` lane {lane} changed signed-zero semantics."
        );
        return;
    }
    if expected.is_infinite() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Fix: CUDA f32 generated case `{case_name}` lane {lane} changed infinity sign."
        );
        return;
    }
    let ulp = f32_ulp_distance(actual, expected).unwrap_or(u32::MAX);
    assert!(
        ulp <= max_ulp,
        "Fix: CUDA f32 generated case `{case_name}` lane {lane} exceeded {max_ulp} ULP: actual={actual:?} expected={expected:?} ulp={ulp}."
    );
}

fn f32_ulp_distance(actual: f32, expected: f32) -> Option<u32> {
    if actual.to_bits() == expected.to_bits() {
        return Some(0);
    }
    if actual.is_nan() || expected.is_nan() {
        return None;
    }
    Some(ordered_f32_bits(actual).abs_diff(ordered_f32_bits(expected)))
}

pub(crate) fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

/// Workgroup width of [`cross_block_grid_sync_program`]. Lane count divided by
/// this is the block count, which is what makes the cross-block read in segment
/// 1 actually cross a block boundary.
pub(crate) const CROSS_BLOCK_GRID_SYNC_WORKGROUP: u32 = 256;

/// Pre-barrier accumulate iterations charged per block index in
/// [`cross_block_grid_sync_program`]. Block `b` runs `b * DELAY` dependent
/// read-modify-write iterations before the barrier, so block 0 runs none and the
/// last block runs the most.
///
/// This asymmetry is the entire detection mechanism, so it is not a tuning knob
/// to trim. A cooperative launch guarantees every block is co-resident, which
/// means all blocks START together: with uniform pre-barrier work every block
/// reaches the barrier within a few hundred nanoseconds of the others, and the
/// two `bar.sync` plus one global atomic on the barrier path already cost more
/// than that. A missing barrier is then invisible, because the value a block
/// would read has already landed by the time it can read it. That was measured,
/// not assumed: with uniform pre-barrier work this fixture passed with the
/// counter reset removed entirely, at 8 blocks and again at 512.
pub(crate) const CROSS_BLOCK_GRID_SYNC_DELAY_PER_BLOCK: u32 = 2;

/// Program whose correct answer requires a whole-grid barrier to actually block.
///
/// Shape, with `b = gid / workgroup` as the block index:
///   segment 0: `scratch[gid] += 1`, repeated `b * DELAY` times
///   barrier:   `MemoryOrdering::GridSync`
///   segment 1: `out[gid] = scratch[n - 1] + input[gid]`
///
/// Two properties make this a real detector rather than a race that usually
/// resolves benignly.
///
/// The accumulation is LOAD-BEARING: `scratch[n - 1]`'s accumulated value is
/// exactly what segment 1 reads, so the loop cannot be dropped as dead work and
/// its result is what the assertion checks. A fixture that overwrote the
/// accumulated slot with a constant afterwards would let an optimizer delete the
/// loop and silently lose all detection power.
///
/// The work is ASYMMETRIC and inverted against block scheduling order: the slot
/// every block reads belongs to the LAST lane, whose block does the MOST
/// pre-barrier work. Block 0 does none, so it reaches the read first needing a
/// value only the slowest block can produce. Without a barrier it observes that
/// slot mid-accumulation and the output is below the correct value.
///
/// Pair it with [`cross_block_grid_sync_inputs`] and
/// [`cross_block_grid_sync_expected`], which fix the one correct answer.
pub(crate) fn cross_block_grid_sync_program(n: u32) -> Program {
    assert!(
        n >= 2 * CROSS_BLOCK_GRID_SYNC_WORKGROUP && n % CROSS_BLOCK_GRID_SYNC_WORKGROUP == 0,
        "Fix: the cross-block grid-sync fixture needs a whole number of blocks and at least two \
         of them; got {n} lanes at workgroup {CROSS_BLOCK_GRID_SYNC_WORKGROUP}."
    );
    // Iterations for this lane: (gid / workgroup) * DELAY.
    let iterations = Expr::mul(
        Expr::div(Expr::gid_x(), Expr::u32(CROSS_BLOCK_GRID_SYNC_WORKGROUP)),
        Expr::u32(CROSS_BLOCK_GRID_SYNC_DELAY_PER_BLOCK),
    );
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(n),
            BufferDecl::read_write("scratch", 1, DataType::U32).with_count(n),
            BufferDecl::output("out", 2, DataType::U32).with_count(n),
        ],
        [CROSS_BLOCK_GRID_SYNC_WORKGROUP, 1, 1],
        vec![
            // segment 0: block-proportional dependent accumulate into this lane's
            // own slot. Later blocks take strictly longer to reach the barrier.
            Node::loop_for(
                "grid_sync_delay",
                Expr::u32(0),
                iterations,
                vec![Node::store(
                    "scratch",
                    Expr::gid_x(),
                    Expr::add(Expr::load("scratch", Expr::gid_x()), Expr::u32(1)),
                )],
            ),
            // whole-grid barrier: the LAST block's accumulation must be complete
            // and visible to every block, including the ones that finished first.
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            // segment 1: out[gid] = scratch[n - 1] + input[gid]
            Node::store(
                "out",
                Expr::gid_x(),
                Expr::add(
                    Expr::load("scratch", Expr::u32(n - 1)),
                    Expr::load("input", Expr::gid_x()),
                ),
            ),
        ],
    )
}

/// Inputs for [`cross_block_grid_sync_program`]: `input[gid] == gid`, and
/// `scratch` seeded to the same values so the accumulate has a known base.
///
/// `scratch` is read_write, so every launch MUST re-upload it. A launch that
/// inherited the previous launch's `scratch` would read an already-accumulated
/// `scratch[n - 1]` and pass even with no barrier at all, which would hide the
/// whole defect class this fixture exists to catch.
pub(crate) fn cross_block_grid_sync_inputs(n: u32) -> Vec<Vec<u8>> {
    let lanes: Vec<u32> = (0..n).collect();
    let input = u32_bytes(&lanes);
    let scratch = input.clone();
    vec![input, scratch]
}

/// The only correct `out` buffer for [`cross_block_grid_sync_program`] driven by
/// [`cross_block_grid_sync_inputs`].
///
/// The last lane accumulates up from `n - 1` by one per iteration over
/// `(blocks - 1) * DELAY` iterations, so after the barrier every lane reads
/// `scratch[n - 1] == (n - 1) + (blocks - 1) * DELAY` and stores that plus its
/// own `input[gid] == gid`.
///
/// A lane BELOW its expected value read `scratch[n - 1]` while the last block was
/// still accumulating, which is the precise signature of a grid barrier that did
/// not block. The shortfall says how far behind that block still was.
pub(crate) fn cross_block_grid_sync_expected(n: u32) -> Vec<u32> {
    let blocks = n / CROSS_BLOCK_GRID_SYNC_WORKGROUP;
    let last_slot = (n - 1) + (blocks - 1) * CROSS_BLOCK_GRID_SYNC_DELAY_PER_BLOCK;
    (0..n).map(|gid| last_slot + gid).collect()
}
