#![cfg(feature = "device-tests")]
// Integration test module for the containing Vyre package.
#![allow(dead_code, unused_imports)]

pub(crate) mod every_op_random_inputs;
pub(crate) mod self_optimizer;

use std::time::{Duration, Instant};
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::PendingDispatch;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_emit_naga::program::emit_module;

const LIVE_GPU_REQUIRED: &str =
    "WgpuBackend acquisition failed on a machine that must have a GPU. \
Fix: inspect WGPU adapter probing and driver visibility; live GPU tests must not silently skip.";

/// Acquire a fresh live WGPU backend for tests that need isolated backend state.
pub(crate) fn acquire_live_backend() -> WgpuBackend {
    WgpuBackend::acquire().expect(LIVE_GPU_REQUIRED)
}

/// Acquire the shared live WGPU backend for capability/adapter tests.
pub(crate) fn shared_live_backend() -> WgpuBackend {
    WgpuBackend::shared()
        .expect(LIVE_GPU_REQUIRED)
        .as_ref()
        .clone()
}

/// Resolve the underlying live WGPU adapter for the active backend.
pub(crate) fn selected_adapter(backend: &WgpuBackend) -> wgpu::Adapter {
    vyre_driver_wgpu::runtime::adapter_for_info(backend.adapter_info()).expect(
        "Fix: selected wgpu backend adapter must remain enumerable for live capability probing",
    )
}

/// Map IEEE-754 binary32 bit-patterns to ordered integer keys for ULP comparison.
pub(crate) fn f32_to_ordered(bits: u32) -> u32 {
    if (bits & 0x8000_0000) != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Pack little-endian `u32` lanes into backend dispatch bytes.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

/// Alias used by C parser integration tests.
pub(crate) fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    u32_bytes(words)
}

/// Decode backend output bytes into little-endian `u32` lanes.
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as bytes_u32;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

/// Alias used by C parser integration tests.
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;

/// Lower a test program with the canonical unit workgroup, validate it, and return WGSL.
pub(crate) fn emit_validated_wgsl(program: &Program) -> String {
    let module = emit_module(program, [1, 1, 1])
        .expect("Fix: test program must lower to a valid Naga module.");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("Fix: lowered test program must pass Naga validation.");
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .expect("Fix: validated test module must serialize to WGSL.")
}

pub(crate) fn add_one_program(words: u32) -> Program {
    let idx = Expr::gid_x();
    let in_bounds = Expr::lt(idx.clone(), Expr::u32(words));
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(words),
            BufferDecl::output("out", 1, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0..(words as usize * 4)),
        ],
        [64, 1, 1],
        vec![
            Node::if_then(
                in_bounds,
                vec![Node::store(
                    "out",
                    idx.clone(),
                    Expr::add(Expr::load("input", idx), Expr::u32(1)),
                )],
            ),
            Node::return_(),
        ],
    )
}

pub(crate) fn add_one_input(words: u32) -> Vec<u8> {
    (0..words).flat_map(u32::to_le_bytes).collect()
}

pub(crate) fn add_one_expected(words: u32) -> Vec<u8> {
    (1..=words).flat_map(u32::to_le_bytes).collect()
}

pub(crate) fn assert_dispatch_async_returns_before_gpu_completion() {
    let backend = acquire_live_backend();
    let program = add_one_program(512 * 1024);
    let input = add_one_input(512 * 1024);

    let start = Instant::now();
    let pending = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: dispatch_async must return a handle immediately without blocking on GPU completion");
    let return_time = start.elapsed();

    assert!(
        return_time < Duration::from_secs(1),
        "Fix: dispatch_async took {:?} to return; this suggests synchronous GPU blocking",
        return_time
    );

    let _ = pending.is_ready();

    let outputs = pending
        .await_result()
        .expect("Fix: await_result must resolve correctly");
    let expected = add_one_expected(512 * 1024);
    assert_eq!(outputs, vec![expected]);
}

pub(crate) fn assert_dispatch_async_ready_state_observable_for_non_trivial_work() {
    let backend = acquire_live_backend();
    let program = add_one_program(256 * 1024);
    let input = add_one_input(256 * 1024);

    let pending = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: dispatch_async must return a handle");

    let _ready_now = pending.is_ready();

    let outputs = pending
        .await_result()
        .expect("Fix: await_result must resolve");
    let expected = add_one_expected(256 * 1024);
    assert_eq!(outputs, vec![expected]);
}

pub(crate) fn assert_multiple_concurrent_async_dispatches_do_not_serialize() {
    let backend = acquire_live_backend();
    let program = add_one_program(128 * 1024);
    let input = add_one_input(128 * 1024);

    let _ = backend
        .dispatch(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: warm-up dispatch must succeed");

    let start = Instant::now();
    let p1 = backend
        .dispatch_async(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: dispatch_async #1 must start");
    let p2 = backend
        .dispatch_async(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: dispatch_async #2 must start");
    let p3 = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: dispatch_async #3 must start");
    let submit_time = start.elapsed();

    assert!(
        submit_time < Duration::from_millis(100),
        "Fix: three back-to-back dispatch_async calls took {:?}, suggesting blocking behavior",
        submit_time
    );

    let o1 = p1
        .await_result()
        .expect("Fix: async dispatch #1 must complete");
    let o2 = p2
        .await_result()
        .expect("Fix: async dispatch #2 must complete");
    let o3 = p3
        .await_result()
        .expect("Fix: async dispatch #3 must complete");
    assert_eq!(
        o1, o2,
        "Fix: identical async dispatches must produce identical outputs"
    );
    assert_eq!(o2, o3);
}

pub(crate) fn assert_pending_dispatch_from_wgpu_is_object_safe() {
    let backend = acquire_live_backend();
    let program = add_one_program(1024);
    let input = add_one_input(1024);

    let pending: Box<dyn PendingDispatch> = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: wgpu dispatch_async must produce object-safe PendingDispatch");

    let outputs = pending
        .await_result()
        .expect("Fix: object-safe await must succeed");
    let expected = add_one_expected(1024);
    assert_eq!(outputs, vec![expected]);
}

/// Assert that the backend is bound to a real hardware GPU, never a CPU fallback.
pub(crate) fn assert_non_cpu_backend(backend: &WgpuBackend) {
    let info = backend.adapter_info();
    assert!(
        !matches!(
            info.device_type,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
        ),
        "Fix: WgpuBackend must never silently fall back to a CPU adapter. Adapter `{}` has type {:?}.",
        info.name,
        info.device_type
    );
}

/// Assert that an error result contains an actionable `Fix:` message.
pub(crate) fn assert_actionable_error<T: std::fmt::Debug>(
    result: &Result<T, vyre_driver::BackendError>,
    msg: &str,
) {
    let err = result.as_ref().unwrap_err();
    let text = err.to_string();
    assert!(text.contains("Fix:"), "Fix: {msg}. Got: {text}");
}

/// Standard subgroup probe WGSL shader using subgroup builtins.
pub(crate) const SUBGROUP_PROBE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(1) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(32)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) width: u32,
) {
    if (params.x == 4294967295u) {
        return;
    }
    let seed = input[0];
    if (gid.x == 0u) {
        output[0] = seed + subgroupAdd(select(1u, 0u, lane >= width));
    }
}
"#;

/// Build a long-running program that requires measurable execution time.
///
/// One invocation per output word, so the element count also decides the grid.
/// WebGPU admits at most 65535 workgroups per axis and this program declares a
/// 1D launch, so the count has to stay inside that product or every dispatch of
/// it is refused before it reaches the device.
pub(crate) fn long_running_program() -> Program {
    const WORKGROUP_INVOCATIONS: u32 = 256;
    const MAX_WORKGROUPS_PER_AXIS: u32 = 65_535;
    const OUTPUT_WORDS: u32 = 8 * 1024 * 1024;
    const _: () = assert!(OUTPUT_WORDS.div_ceil(WORKGROUP_INVOCATIONS) <= MAX_WORKGROUPS_PER_AXIS);
    let mut body = Vec::with_capacity(515);
    body.push(Node::let_bind("idx", Expr::gid_x()));
    body.push(Node::let_bind("acc", Expr::var("idx")));
    for round in 0..512u32 {
        body.push(Node::assign(
            "acc",
            Expr::bitxor(
                Expr::mul(Expr::var("acc"), Expr::u32(1_664_525)),
                Expr::add(
                    Expr::var("idx"),
                    Expr::u32(1_013_904_223u32.wrapping_add(round)),
                ),
            ),
        ));
    }
    body.push(Node::if_then(
        Expr::lt(Expr::var("idx"), Expr::buf_len("out")),
        vec![Node::store("out", Expr::var("idx"), Expr::var("acc"))],
    ));
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(OUTPUT_WORDS)
            .with_output_byte_range(0..4)],
        [WORKGROUP_INVOCATIONS, 1, 1],
        body,
    )
}

/// Compute a 1D grid dispatch override for multi-dimensional workgroups in Cat-A fixtures.
pub(crate) fn cat_a_dispatch_config(program: &Program) -> DispatchConfig {
    let mut config = DispatchConfig::default();
    let workgroup = program.workgroup_size();
    if workgroup[1] == 1 && workgroup[2] == 1 {
        return config;
    }
    let lanes = u64::from(workgroup[0])
        .saturating_mul(u64::from(workgroup[1]))
        .saturating_mul(u64::from(workgroup[2]));
    let max_writable_count = program
        .buffers()
        .iter()
        .filter(|decl| {
            matches!(decl.access(), vyre::ir::BufferAccess::ReadWrite) || decl.is_output()
        })
        .map(|decl| u64::from(decl.count()))
        .max()
        .unwrap_or(1);
    assert!(
        max_writable_count <= lanes,
        "Fix: non-1D Cat-A program needs explicit multi-workgroup grid; workgroup={workgroup:?}, lanes={lanes}, writable={max_writable_count}"
    );
    config.grid_override = Some([1, 1, 1]);
    config
}

/// Pack a slice of bytes into little-endian u32 words.
pub(crate) fn byte_stream_input_bytes(bytes: &[u8]) -> Vec<u8> {
    let words: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    u32_bytes(&words)
}

/// Dispatch a program and extract its single u32 scalar output.
pub(crate) fn dispatch_single_u32_output(
    backend: &WgpuBackend,
    program: &Program,
    inputs: &[&[u8]],
    fix_msg: &str,
) -> u32 {
    let outputs = backend
        .dispatch_borrowed(program, inputs, &DispatchConfig::default())
        .expect(fix_msg);
    assert_eq!(outputs.len(), 1, "expected 1 output buffer");
    assert!(outputs[0].len() >= 4, "expected at least 4-byte output");
    u32::from_le_bytes([outputs[0][0], outputs[0][1], outputs[0][2], outputs[0][3]])
}
