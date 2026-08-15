//! Native Metal against wgpu-on-Metal on the same Program bytes.
//!
//! Two backends on one Apple GPU must agree byte for byte. This crate holds the
//! test because the subject is what native Metal produces; the wgpu side is the
//! second opinion. Each backend is first checked against an explicit byte oracle,
//! so two backends that agree on a wrong answer still fail.

use crate::*;

use vyre_driver::{DispatchConfig, VyreBackend as _};
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
