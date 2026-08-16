//! Cooperative grid-sync contracts for resident and concurrent CUDA dispatch.
//!
//! Every launch must reset and serialize the module-scope barrier counter.
//! Tests assert exact cross-block outputs because an early barrier release
//! returns populated buffers containing wrong values rather than an error.

mod harness;
use harness::{
    bytes_u32, cross_block_grid_sync_expected, cross_block_grid_sync_inputs,
    cross_block_grid_sync_program, CROSS_BLOCK_GRID_SYNC_WORKGROUP,
};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_cuda::{cuda_factory, CudaBackend};

/// Lane count for every case here. 512 blocks of 256 threads sits well inside
/// the cooperative residency ceiling on a grid-sync capable device while still
/// spreading block start times widely enough that segment 1's read of the LAST
/// lane's slot genuinely depends on the barrier. An 8-block grid does NOT: every
/// block starts within a few microseconds, so the output is correct even with the
/// barrier released early, and the gate passes while the defect is present. That
/// was measured, not assumed.
const LANES: u32 = 512 * CROSS_BLOCK_GRID_SYNC_WORKGROUP;

/// Launch repetitions for the reuse gates. A stale counter is released early on
/// the SECOND launch, so 2 is the minimum that can fail; going further catches a
/// counter that drifts only after several reuses, such as a reset that lands on
/// the wrong stream and races rather than being absent.
const REUSE_LAUNCHES: usize = 8;

/// Acquire the registered CUDA backend used by shared-backend concurrency tests.
fn live_registered_backend() -> Box<dyn VyreBackend> {
    let backend = cuda_factory()
        .expect("Fix: CUDA backend factory must succeed on the GPU-required test host.");
    assert!(
        backend.supports_grid_sync(),
        "Fix: this host reports no native cooperative grid-sync lowering, so every gate in this \
         suite would pass vacuously. Native grid-sync is required on the CUDA test fleet."
    );
    backend
}

/// Assert one `out` buffer holds exactly the expected value for every lane, and
/// report the first offending lane with both values when it does not.
///
/// The failure message separates the two ways a lane goes wrong, because they
/// point at different code. A lane BELOW its expected value read the last lane's
/// accumulator before the last block finished, so the barrier did not block. A
/// lane above it, or off by something unrelated to the accumulator, is a
/// segment-0 or readback fault rather than a barrier fault.
fn assert_grid_sync_output(out_bytes: &[u8], context: &str) {
    let actual = bytes_u32(out_bytes);
    let expected = cross_block_grid_sync_expected(LANES);
    assert_eq!(
        actual.len(),
        expected.len(),
        "Fix: {context} returned {} lanes, expected {}. The `out` buffer must cover every lane.",
        actual.len(),
        expected.len()
    );
    if let Some((lane, (&got, &want))) = actual
        .iter()
        .zip(expected.iter())
        .enumerate()
        .find(|(_, (got, want))| got != want)
    {
        let diagnosis = if got < want {
            format!(
                "that lane read scratch[n - 1] while the LAST block was still {} iterations from \
                 finishing, so the whole-grid barrier did not block. On a reused launch this is \
                 the signature of a stale _vyre_grid_barrier counter that already sat at or past \
                 this barrier's release target, releasing it on arrival.",
                want - got
            )
        } else {
            "that lane is ABOVE its expected value, which the barrier cannot cause: the \
             accumulator is monotonic and capped by the last block's iteration count. Look at \
             segment 0's accumulate or the output readback, not the barrier."
                .to_string()
        };
        panic!(
            "Fix: {context} produced out[{lane}] = {got}, expected {want}. Block {} of {}. \
             {diagnosis}",
            u32::try_from(lane).unwrap_or(u32::MAX) / CROSS_BLOCK_GRID_SYNC_WORKGROUP,
            LANES / CROSS_BLOCK_GRID_SYNC_WORKGROUP
        );
    }
}

/// The last output buffer of the fixture program is `out`. Taking it positionally
/// keeps the assertion honest if a buffer is ever added ahead of it.
fn out_buffer(outputs: &[Vec<u8>], context: &str) -> Vec<u8> {
    outputs
        .last()
        .unwrap_or_else(|| {
            panic!("Fix: {context} returned no output buffers; the fixture declares `out`.")
        })
        .clone()
}

/// The resident launch route must reset the barrier counter between launches.
///
/// This is the route that was actually broken. `dispatch_resident` did not zero
/// `_vyre_grid_barrier` at all, and it did not force cooperative launch for a
/// grid-sync program either, so a second resident launch of one grid-sync
/// program released its first barrier immediately. A compiled pipeline reaches
/// this code through its persistent-handle entry points, which is why the
/// ahead-of-time refusal was load-bearing until this was fixed.
///
/// `scratch` is re-uploaded before each launch so the second launch starts from
/// the same state as the first. Without that, the second launch would read an
/// already-incremented `scratch[0]` and pass even with no barrier at all.
#[test]
fn resident_grid_sync_dispatch_synchronizes_on_its_second_launch() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let program = cross_block_grid_sync_program(LANES);
    let config = DispatchConfig::default();
    let inputs = cross_block_grid_sync_inputs(LANES);
    let out_bytes = inputs[0].len();

    let mut handles = Vec::with_capacity(3);
    for input in &inputs {
        let handle = backend
            .allocate_resident(input.len())
            .expect("Fix: resident input allocation must succeed.");
        handles.push(handle);
    }
    let out_handle = backend
        .allocate_resident(out_bytes)
        .expect("Fix: resident output allocation must succeed.");
    handles.push(out_handle);

    for launch in 1..=REUSE_LAUNCHES {
        // Re-upload every input, including the read_write `scratch`, so each
        // launch starts from identical state and only the barrier counter
        // carries over.
        for (index, input) in inputs.iter().enumerate() {
            backend
                .upload_resident(handles[index], input)
                .unwrap_or_else(|error| {
                    panic!("Fix: resident input {index} upload for launch {launch} failed: {error}")
                });
        }
        backend
            .dispatch_resident(&program, &handles, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: resident grid-sync dispatch launch {launch} failed: {error}")
            });
        let out = backend
            .download_resident(out_handle)
            .unwrap_or_else(|error| {
                panic!("Fix: resident output download for launch {launch} failed: {error}")
            });
        assert_grid_sync_output(
            &out,
            &format!(
                "resident grid-sync launch {launch} of {REUSE_LAUNCHES} (the route that did not \
                 reset the counter at all)"
            ),
        );
    }

    for handle in handles {
        backend
            .free_resident(handle)
            .expect("Fix: resident handle cleanup must succeed.");
    }
}

/// Concurrent cooperative dispatches of ONE program through ONE backend must all
/// be correct.
///
/// This is a permanent property, not a tripwire. `CudaBackend` is `Clone` and
/// every clone shares one `Arc<CudaModuleCache>`, so all threads dispatching one
/// program share a single loaded module and therefore a single
/// `_vyre_grid_barrier`. Sharing one backend across a thread pool is the obvious
/// and only realistic way to use the API, so if per-launch zeroing of a
/// module-scope counter is unsafe under concurrency, it is reachable from the
/// shipped surface by ordinary code and this test is what notices.
///
/// Note precisely what this does NOT cover: separate `CudaBackend::acquire`
/// calls build separate module caches and therefore separate counters, so a
/// per-thread-backend test cannot provoke aliasing no matter how many threads it
/// runs. That is why this test shares one backend deliberately.
#[test]
fn concurrent_cooperative_dispatches_through_one_shared_backend_stay_correct() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 16;

    let backend = live_registered_backend();
    let backend: &dyn VyreBackend = backend.as_ref();
    let program = cross_block_grid_sync_program(LANES);
    let config = DispatchConfig::default();
    let inputs = cross_block_grid_sync_inputs(LANES);

    let failures = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|thread| {
                let program = &program;
                let config = &config;
                let inputs = &inputs;
                scope.spawn(move || {
                    let mut local_failures = Vec::new();
                    for iteration in 0..PER_THREAD {
                        match backend.dispatch(program, inputs, config) {
                            Ok(outputs) => match outputs.last() {
                                Some(out) => {
                                    let actual = bytes_u32(out);
                                    let expected = cross_block_grid_sync_expected(LANES);
                                    if actual != expected {
                                        let lane = actual
                                            .iter()
                                            .zip(expected.iter())
                                            .position(|(got, want)| got != want)
                                            .unwrap_or(0);
                                        local_failures.push(format!(
                                            "thread {thread} iteration {iteration}: out[{lane}] = \
                                             {} expected {}",
                                            actual[lane], expected[lane]
                                        ));
                                    }
                                }
                                None => local_failures.push(format!(
                                    "thread {thread} iteration {iteration}: no output buffers"
                                )),
                            },
                            Err(error) => local_failures.push(format!(
                                "thread {thread} iteration {iteration}: dispatch failed: {error}"
                            )),
                        }
                    }
                    local_failures
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| {
                worker
                    .join()
                    .expect("Fix: a concurrent cooperative dispatch worker panicked.")
            })
            .collect::<Vec<_>>()
    });

    assert!(
        failures.is_empty(),
        "Fix: {} of {} concurrent cooperative grid-sync dispatches through ONE shared backend \
         produced wrong results. All threads share one loaded module and therefore one \
         module-scope _vyre_grid_barrier counter, so a per-launch zeroing of that counter races \
         against another thread's in-flight grid: one thread's reset lands while another's blocks \
         are still spinning, that grid's wait predicate goes false on arrival, and it runs \
         unsynchronized. Scope the counter per launch or serialize cooperative launches that \
         share a module. First failures:\n{}",
        failures.len(),
        THREADS * PER_THREAD,
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Concurrent cooperative dispatches on INDEPENDENT backends must all be
/// correct, and this test records why that is a weaker statement.
///
/// Each `CudaBackend::acquire` builds a fresh `Arc<CudaModuleCache>`, so each
/// backend loads its own CUmodule and gets its own `_vyre_grid_barrier` at its
/// own device address. Threads that each own a backend therefore cannot alias
/// the counter regardless of thread count. Keeping this case explicit stops a
/// future reader from mistaking a green per-thread-backend test for coverage of
/// the shared-backend property above, which is the mistake that hid this
/// question in the first place.
#[test]
fn concurrent_cooperative_dispatches_on_independent_backends_stay_correct() {
    const THREADS: usize = 4;
    const PER_THREAD: usize = 8;

    let program = cross_block_grid_sync_program(LANES);
    let config = DispatchConfig::default();
    let inputs = cross_block_grid_sync_inputs(LANES);
    let expected = cross_block_grid_sync_expected(LANES);

    let failures = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|thread| {
                let program = &program;
                let config = &config;
                let inputs = &inputs;
                let expected = &expected;
                scope.spawn(move || {
                    let backend = cuda_factory()
                        .expect("Fix: per-thread CUDA backend acquisition must succeed.");
                    let mut local_failures = Vec::new();
                    for iteration in 0..PER_THREAD {
                        match backend.dispatch(program, inputs, config) {
                            Ok(outputs) => match outputs.last().map(|out| bytes_u32(out)) {
                                Some(actual) if &actual == expected => {}
                                Some(actual) => local_failures.push(format!(
                                    "thread {thread} iteration {iteration}: first wrong lane {:?}",
                                    actual
                                        .iter()
                                        .zip(expected.iter())
                                        .enumerate()
                                        .find(|(_, (got, want))| got != want)
                                )),
                                None => local_failures.push(format!(
                                    "thread {thread} iteration {iteration}: no output buffers"
                                )),
                            },
                            Err(error) => local_failures.push(format!(
                                "thread {thread} iteration {iteration}: dispatch failed: {error}"
                            )),
                        }
                    }
                    local_failures
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| {
                worker
                    .join()
                    .expect("Fix: an independent-backend cooperative worker panicked.")
            })
            .collect::<Vec<_>>()
    });

    assert!(
        failures.is_empty(),
        "Fix: {} of {} concurrent cooperative grid-sync dispatches on INDEPENDENT backends \
         produced wrong results. Independent backends hold independent module caches and \
         therefore independent barrier counters, so a failure here is NOT counter aliasing: look \
         at per-launch device state such as the transient allocation pool or the stream pool. \
         First failures:\n{}",
        failures.len(),
        THREADS * PER_THREAD,
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
