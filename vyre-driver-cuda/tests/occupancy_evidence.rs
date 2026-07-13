//! W3-6 per-kernel occupancy evidence, on a real GPU launch.
//!
//! Every kernel launch now records its DRIVER-measured achieved occupancy (via
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor`, cached per kernel shape) into
//! the telemetry snapshot as an aggregate mean plus measured/unmeasured launch
//! counts. This test proves the evidence is real on the RTX 5090: it runs a
//! dispatch loop and asserts the snapshot reports measured launches with a
//! plausible occupancy fraction and NO unmeasured launches.

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// A parallel Program: 256 threads each storing their index. A real grid/block
/// so the occupancy query returns a meaningful active-block count (a single-thread
/// kernel would still measure, but this exercises a realistic block size).
fn parallel_store_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(256)],
        [256, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
    )
}

#[test]
fn steady_state_launches_report_per_kernel_occupancy_evidence() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");

    let program = parallel_store_program();
    let inputs: Vec<Vec<u8>> = vec![vec![0u8; 256 * 4]];
    let config = DispatchConfig::default();

    // Warm the module + occupancy cache, then measure a clean epoch.
    backend
        .dispatch(&program, &inputs, &config)
        .expect("Fix: CUDA dispatch must succeed while warming.");
    backend.reset_telemetry();

    const LAUNCHES: usize = 16;
    for _ in 0..LAUNCHES {
        backend
            .dispatch(&program, &inputs, &config)
            .expect("Fix: CUDA dispatch must succeed in the occupancy loop.");
    }

    let snapshot = backend.telemetry_snapshot();
    let measured = snapshot.occupancy_measured_launches;
    let unmeasured = snapshot.occupancy_unmeasured_launches;
    let mean_bps = snapshot.mean_occupancy_bps();
    println!(
        "occupancy over {LAUNCHES} launches: measured={measured} unmeasured={unmeasured} mean={mean_bps}bps"
    );

    // Every launch in the loop was measured (the kernel loaded, the driver
    // occupancy query succeeded) (no silent measurement gap).
    assert!(
        measured >= LAUNCHES as u64,
        "each dispatch must record a measured occupancy; measured={measured} over {LAUNCHES} launches"
    );
    assert_eq!(
        unmeasured, 0,
        "no launch of a well-formed kernel should be occupancy-unmeasured; got {unmeasured}"
    );

    // A real occupancy fraction: strictly positive (the kernel runs) and at most
    // full (10000 bps). A broken occupancy query would land outside this range.
    assert!(
        mean_bps > 0 && mean_bps <= 10_000,
        "mean occupancy must be a real fraction in (0, 10000] bps; got {mean_bps}"
    );

    // The evidence is operator-visible in the Prometheus exposition.
    let text = snapshot.to_prometheus_text();
    assert!(
        text.contains("vyre_cuda_mean_occupancy_bps")
            && text.contains("vyre_cuda_occupancy_measured_launches_total"),
        "occupancy evidence must be exported in the Prometheus text"
    );

    // The mean is exactly consistent with the raw sum/count it derives from.
    assert_eq!(
        mean_bps,
        (snapshot.launch_occupancy_bps_sum / snapshot.occupancy_measured_launches) as u32,
        "reported mean must equal sum/measured"
    );
}
