//! Native Metal against wgpu-on-Metal on the same Program bytes.
//!
//! Two backends on one Apple GPU must agree byte for byte. This crate holds the
//! test because the subject is what native Metal produces; the wgpu side is the
//! second opinion. Each backend is first checked against an explicit byte oracle,
//! so two backends that agree on a wrong answer still fail.

use super::*;

use vyre_driver::DispatchConfig;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[test]
fn apple_native_metal_matches_wgpu_on_same_program_bytes() {
    let idx = Expr::var("idx");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32).with_count(8),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32).with_count(8),
            BufferDecl::storage("out", 2, BufferAccess::WriteOnly, DataType::U32)
                .with_count(8)
                .with_output_byte_range(0..32),
        ],
        [8, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(idx.clone(), Expr::u32(8)),
                vec![Node::store(
                    "out",
                    idx.clone(),
                    Expr::add(
                        Expr::load("a", idx.clone()),
                        Expr::mul(Expr::load("b", idx), Expr::u32(3)),
                    ),
                )],
            ),
        ],
    );
    let a = [1u32, 2, 3, 4, 5, 6, 7, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let b = [10u32, 11, 12, 13, 14, 15, 16, 17]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let expected = [31u32, 35, 39, 43, 47, 51, 55, 59]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let metal = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before differential dispatch.",
    );
    let wgpu = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU-on-Metal must acquire on the Apple GPU differential lane.");
    let config = DispatchConfig::default();
    let metal_outputs = metal
        .dispatch(&program, &[a.clone(), b.clone()], &config)
        .expect("Fix: native Metal must dispatch the differential Program.");
    let wgpu_outputs = wgpu
        .dispatch(&program, &[a, b], &config)
        .expect("Fix: WGPU-on-Metal must dispatch the same differential Program.");

    assert_eq!(
        metal_outputs,
        vec![expected.clone()],
        "Fix: native Metal output must match the explicit byte oracle before comparing backends."
    );
    assert_eq!(
        wgpu_outputs,
        vec![expected],
        "Fix: WGPU-on-Metal output must match the explicit byte oracle before comparing backends."
    );
    assert_eq!(
        metal_outputs, wgpu_outputs,
        "Fix: native Metal and WGPU-on-Metal must produce byte-identical outputs for the same Program."
    );
}

/// WHY: closes the class "a launch folded across grid axes computes something
/// other than the same launch on one axis". The two backends reach the same
/// elements through different arithmetic here, which is what makes the
/// comparison worth running: wgpu is given a per-axis workgroup ceiling low
/// enough that grid inference must fold the launch onto y, so its kernel reads
/// an element index linearized over the whole grid, while native Metal plans the
/// same program against its own device truth and dispatches one axis with a
/// per-axis index. A fold that mis-addresses one lane disagrees with both the
/// byte oracle and the other backend.
///
/// The ceiling is pinned rather than probed because the number under test is the
/// portability floor a conformant implementation guarantees, and this GPU
/// publishes a per-axis maximum no fixture-sized buffer reaches.
///
/// What it does not prove: a launch at this GPU's own ceiling. That needs a
/// buffer of tens of millions of elements and belongs to a capacity case, not to
/// a differential one.
#[test]
fn a_folded_wgpu_launch_matches_native_metal_on_one_axis() {
    // 4 workgroups of 256 lanes cover 1024 elements on x, so 4096 elements need
    // four rows of y once the ceiling is pinned to 4.
    const WORDS: u32 = 4096;
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(WORDS)
                .with_output_byte_range(0..(WORDS as usize * 4)),
        ],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(WORDS)),
            vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
        )],
    );
    let expected = (0..WORDS).flat_map(u32::to_le_bytes).collect::<Vec<_>>();

    let metal =
        acquire().expect("Fix: Apple Metal builds must acquire the system default MTLDevice.");
    let wgpu = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU-on-Metal must acquire on the Apple GPU differential lane.");

    let mut folded = DispatchConfig::default();
    folded.max_workgroups_per_axis = Some([4, 65_535, 65_535]);
    let wgpu_outputs = wgpu
        .dispatch(&program, &[], &folded)
        .expect("Fix: WGPU-on-Metal must fold a launch past the pinned per-axis ceiling onto y rather than refuse it.");
    let metal_outputs = metal
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal must dispatch the same Program against its own device truth.");

    assert_eq!(
        wgpu_outputs,
        vec![expected.clone()],
        "Fix: a folded WGPU launch must match the explicit byte oracle before comparing backends. \
         Every lane stores its own element index, so a mismatch names the lane the linearized \
         index mis-addressed."
    );
    assert_eq!(
        metal_outputs,
        vec![expected],
        "Fix: native Metal output must match the explicit byte oracle before comparing backends."
    );
    assert_eq!(
        wgpu_outputs, metal_outputs,
        "Fix: a launch folded across grid axes must produce the same buffer as the same launch on \
         one axis."
    );
}
