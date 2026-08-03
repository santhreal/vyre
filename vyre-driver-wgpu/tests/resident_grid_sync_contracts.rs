//! Live WGPU contracts for resident programs with dispatch-level grid synchronization.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::MemoryOrdering;

fn ordered_grid_sync_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("state", 0, BufferAccess::ReadWrite, DataType::U32).with_count(2)],
        [1, 1, 1],
        vec![
            Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(0)),
                vec![Node::store("state", Expr::u32(0), Expr::u32(40))],
            ),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(0)),
                vec![Node::store(
                    "state",
                    Expr::u32(1),
                    Expr::add(Expr::load("state", Expr::u32(0)), Expr::u32(1)),
                )],
            ),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(0)),
                vec![Node::store(
                    "state",
                    Expr::u32(0),
                    Expr::add(Expr::load("state", Expr::u32(1)), Expr::u32(1)),
                )],
            ),
        ],
    )
}

/// Resident WGPU dispatch must split grid barriers into launches before an oversubscribed grid can deadlock.
#[test]
fn resident_grid_sync_completes_and_preserves_segment_order_on_oversubscribed_grid() {
    let backend = WgpuBackend::acquire().expect(
        "Fix: live WGPU backend required for resident grid-sync regression coverage; missing GPU is a configuration bug.",
    );
    let state = backend
        .allocate_resident(2 * size_of::<u32>())
        .expect("Fix: WGPU must allocate resident grid-sync state.");
    let result = (|| {
        backend.upload_resident(&state, &[0; 8])?;
        let mut config = DispatchConfig::default();
        config.grid_override = Some([4_096, 1, 1]);
        let timed = backend.dispatch_resident_timed(
            &ordered_grid_sync_program(),
            &[state.clone()],
            &config,
        )?;
        assert_eq!(
            timed.outputs,
            vec![[42u32.to_le_bytes(), 41u32.to_le_bytes()].concat()],
            "resident grid-sync segments must observe every prior segment's device-resident writes",
        );
        Ok::<(), vyre::BackendError>(())
    })();
    let cleanup = backend.free_resident(state);
    result.expect("Fix: WGPU resident grid-sync dispatch must split before device submission.");
    cleanup.expect("Fix: resident grid-sync test cleanup must free its state buffer.");
}
