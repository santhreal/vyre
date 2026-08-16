//! CUDA backend: device lifecycle, buffer management, and kernel dispatch.

use std::sync::{Arc, Condvar, Mutex};

use dashmap::DashMap;

use cudarc::driver::CudaContext;
use smallvec::SmallVec;
use vyre_driver::trap_record::{decode_trap_record, TRAP_RECORD_BYTES};
use vyre_driver::validation::ValidationCache;
use vyre_driver::SpeculationMode;
use vyre_driver::{resolve_fixpoint_iterations, BackendError, DispatchConfig, LaunchPlan};
use vyre_driver::{BindingPlan, BindingRole};
use vyre_foundation::ir::Program;

use super::allocations::{DeviceAllocationPool, PinnedHostAllocationPool};
use super::module_cache::{
    CudaModuleCache, CudaPtxSourceCache, CudaPtxSourceCacheSnapshot, ModuleCacheKey, ModuleGlobals,
    PtxSourceCacheKey, TrapSidecar,
};
use super::plan::{compute_ordered_output_indices, CudaDispatchPlan};
use super::ptx_target::select_loadable_ptx_target_sm;
use super::resident::{
    CudaDispatchBinding, CudaResidentBuffer, CudaResidentStore, ResidentBufferView,
};
use super::resident_dispatch::next_dispatch_binding;
use super::staging_reserve::reserve_smallvec;
use super::telemetry::{CudaTelemetry, CudaTelemetrySnapshot};
use crate::device::{CudaDeviceCaps, CudaDeviceHandle};

const TRANSIENT_ALLOCATION_POOL_BYTES: usize = 256 * 1024 * 1024;
const PINNED_HOST_POOL_BYTES: usize = 128 * 1024 * 1024;
const CUDA_LAUNCH_RESOURCE_CACHE: usize = 128;

// Inline: covers `acquire`, `drop`, `enqueue_barrier_reset`, which no integration test can name.
#[cfg(test)]
mod tests {
    /// A counter at exactly TWICE the ceiling is the missed-reset signature, and
    /// it must fire.
    ///
    /// This is the literal shape of the bug that shipped. Launch 1 starts from a
    /// zero counter and drives it to `barriers * gridSize`. Launch 2 on the same
    /// module without a reset starts from that value: every barrier finds its
    /// release target already satisfied, becomes a pass-through, and the kernel
    /// returns success with wrong cross-block data. Because a CTA records its
    /// arrival BEFORE it spins, the pass-through arrivals still count, so the
    /// counter lands at exactly 2x. That is the value asserted here. If this stops
    /// firing, the silent-wrong-answer mode is back and the only symptom is bad
    /// output: no error, no log line, no failed launch.
    #[test]
    fn counter_at_exactly_twice_the_ceiling_is_the_missed_reset_signature_and_fires() {
        // One barrier over 4 blocks: ceiling 4, a second un-reset launch reaches 8.
        let error = super::verify_arrival_count(8, 4).expect_err(
            "Fix: 8 arrivals against a ceiling of 4 is a missed reset and must be refused.",
        );
        let message = error.to_string();
        assert!(
            message.contains('8') && message.contains('4'),
            "Fix: the refusal must name the observed count and the ceiling, because the whole \
             reason this bug survived is that the failure was silent. Got: {message}"
        );
        assert!(
            message.contains("_vyre_grid_barrier"),
            "Fix: the refusal must name the counter symbol so a reader can find it. Got: {message}"
        );
        assert!(
            message.contains("enqueue_barrier_reset"),
            "Fix: the refusal must name the call that was skipped, so the fix does not require \
             reading this module. Got: {message}"
        );
        // A fourth un-reset launch lands at 4x and must also fire: the check is
        // not an equality test that only recognizes the doubled case.
        assert!(
            super::verify_arrival_count(16, 4).is_err(),
            "Fix: any multiple of the ceiling is a missed reset and must fire, not just 2x."
        );
    }

    /// One arrival past the ceiling must fire.
    ///
    /// The boundary matters because a PARTIAL reset (a memset that landed on the
    /// wrong stream, or a reset racing a still-spinning grid from a previous
    /// launch) does not produce a clean multiple. Slack here would let exactly
    /// that case through, and it is the case the per-module gate exists to
    /// prevent, so the audit must not soften the boundary the gate defends.
    #[test]
    fn counter_one_past_the_ceiling_fires() {
        assert!(
            super::verify_arrival_count(5, 4).is_err(),
            "Fix: exceeding the ceiling by one must fire; slack would hide a partial reset."
        );
        assert!(
            super::verify_arrival_count(1021, 1020).is_err(),
            "Fix: the boundary must hold at the real cooperative grid width measured on this host \
             (1020 blocks at workgroup 256), not only at toy values."
        );
    }

    /// A counter exactly AT the ceiling must pass.
    ///
    /// This is what a healthy launch produces: every block arrives at every
    /// barrier exactly once. An audit that fired here would refuse every correct
    /// cooperative dispatch, which is not a conservative failure: it would force
    /// whoever hit it to delete the check, and the silent bug would come back with
    /// nothing left watching for it.
    #[test]
    fn counter_exactly_at_the_ceiling_passes() {
        assert!(
            super::verify_arrival_count(4, 4).is_ok(),
            "Fix: a healthy launch lands exactly at the ceiling and must pass."
        );
        assert!(
            super::verify_arrival_count(8160, 8160).is_ok(),
            "Fix: a healthy launch must pass at a realistic width too (8 barriers over 1020 \
             blocks), where an accidental narrower integer type would also show up."
        );
    }

    /// A counter BELOW the ceiling must pass, and this is the case that documents
    /// why exact equality was rejected.
    ///
    /// A grid-uniform early exit legitimately skips later barriers. `exatok`'s
    /// convergence loop takes that exit on every converged encode: the grid agrees
    /// it is done and returns before reaching the remaining barrier sites, leaving
    /// the counter below `barriers * gridSize`. Exact equality would refuse that
    /// on the normal path, and a check that cries wolf on the normal path gets
    /// deleted rather than debugged. The bound is therefore one-sided ON PURPOSE,
    /// and it still catches the real bug because a missed reset overshoots.
    #[test]
    fn counter_below_the_ceiling_passes_because_a_grid_uniform_early_exit_is_legitimate() {
        assert!(
            super::verify_arrival_count(2, 4).is_ok(),
            "Fix: an early exit that skips later barriers lands below the ceiling and is correct."
        );
        assert!(
            super::verify_arrival_count(1020, 8160).is_ok(),
            "Fix: a converged encode that clears only the first of 8 barrier sites over 1020 \
             blocks is correct and must not be refused."
        );
    }

    /// A zero counter must pass, and it IS reachable.
    ///
    /// Two ways to reach it, both correct. A grid-uniform early exit that returns
    /// before the FIRST barrier leaves the counter at the value the pre-launch
    /// memset wrote, which is zero. And a program whose barrier sits inside a
    /// region no block enters (a loop with a zero trip count on this input) never
    /// arrives at all. Neither is a bug, so zero must not fire. It is worth
    /// stating because zero is also what a BROKEN read would return, and the
    /// distinction is that a broken read is caught by `d2h_sync_checked`
    /// propagating its driver error rather than by this comparison.
    #[test]
    fn zero_counter_passes_and_is_reachable_by_an_early_exit_before_the_first_barrier() {
        assert!(
            super::verify_arrival_count(0, 4).is_ok(),
            "Fix: a grid that exits before its first barrier leaves the counter at zero, which is \
             correct and must not fire."
        );
        assert!(
            super::verify_arrival_count(0, 8160).is_ok(),
            "Fix: zero must pass regardless of how large the ceiling is."
        );
    }

    /// The ceiling for a REAL multi-barrier program must be the observed barrier
    /// count times the block count, asserted as an exact number.
    ///
    /// This closes the failure mode that a synthetic-PTX test cannot see: a
    /// ceiling computed too HIGH never fires, and the audit silently becomes
    /// decoration. That is the same failure class as the original no-op barrier,
    /// where everything reported success and only the data was wrong. So the count
    /// is taken from PTX the real emitter produced for a real program rather than
    /// from a hand-written string.
    ///
    /// `persistent_fixpoint_grid` emits two grid barriers per wave (one after the
    /// transfer body, one after the per-word compare and ping-pong), so a
    /// four-wave program carries eight barrier sites. If the emitter ever merges
    /// or drops barrier markers, the marker count and this ceiling disagree with
    /// the barrier count the kernel actually executes, and a missed reset stops
    /// being detectable at the launch that causes it.
    #[test]
    fn four_wave_fixpoint_ceiling_is_eight_barriers_times_the_block_count() {
        use vyre_foundation::ir::{Expr, Node};

        const WORDS: u32 = 64;
        const WAVES: u32 = 4;
        let transfer_body = vec![Node::if_then(
            Expr::lt(Expr::InvocationId { axis: 0 }, Expr::u32(WORDS)),
            vec![Node::store(
                "next",
                Expr::InvocationId { axis: 0 },
                Expr::bitor(
                    Expr::load("current", Expr::InvocationId { axis: 0 }),
                    Expr::u32(1),
                ),
            )],
        )];
        let program = vyre_libs::fixpoint::persistent_fixpoint::persistent_fixpoint_grid(
            transfer_body,
            "current",
            "next",
            "changed",
            WORDS,
            WAVES,
        );

        let mut config = vyre_driver::DispatchConfig::default();
        config.cooperative = true;
        let ptx = crate::codegen::program_to_ptx_for_sm(&program, &config, 90)
            .expect("Fix: the four-wave persistent fixpoint program must emit PTX.");

        let barriers = ptx.matches(super::GRID_BARRIER_PTX_MARKER).count();
        assert_eq!(
            barriers, 8,
            "Fix: four waves emit two grid barriers each, so eight barrier markers must appear. \
             Got {barriers}. A lower count means the ceiling under-counts arrivals and the audit \
             would refuse healthy launches; a higher count means it over-counts and the audit \
             stops detecting a missed reset."
        );

        // 1020 blocks is the cooperative block ceiling measured on this host at
        // workgroup 256, so this is the real geometry the audit runs against.
        let ceiling = super::grid_barrier_arrival_ceiling(&ptx, [1020, 1, 1])
            .expect("Fix: a program with barrier markers must yield a ceiling.");
        assert_eq!(
            ceiling, 8160,
            "Fix: eight barriers over 1020 blocks admit exactly 8160 arrivals (8 * 1020). A \
             ceiling that ignored the barrier count would compute 1020 and refuse this program's \
             every healthy launch; one that over-counted would never fire at all."
        );
        // Same program, narrower grid: the ceiling must track the grid, not latch.
        assert_eq!(
            super::grid_barrier_arrival_ceiling(&ptx, [4, 1, 1])
                .expect("a 4-block grid yields a ceiling"),
            32,
            "Fix: the ceiling must scale with the launch grid; eight barriers over 4 blocks admit \
             32 arrivals."
        );
    }

    /// The ceiling must be `barriers * blocks`, computed from real PTX text.
    ///
    /// A ceiling that ignored the barrier count would refuse every legitimate
    /// multi-barrier launch; one that ignored the grid would refuse every launch
    /// wider than one block. Both are false positives that end with the check
    /// deleted, so the arithmetic is asserted against exact values.
    #[test]
    fn arrival_ceiling_is_barrier_count_times_block_count() {
        let one = "// grid.sync barrier #0 target\nbar.sync 0;\n";
        assert_eq!(
            super::grid_barrier_arrival_ceiling(one, [4, 1, 1])
                .expect("one barrier over 4 blocks is representable"),
            4,
            "Fix: one barrier over 4 blocks admits exactly 4 arrivals."
        );
        let three = "// grid.sync barrier #0\n// grid.sync barrier #1\n// grid.sync barrier #2\n";
        assert_eq!(
            super::grid_barrier_arrival_ceiling(three, [4, 1, 1])
                .expect("three barriers over 4 blocks is representable"),
            12,
            "Fix: three barriers over 4 blocks admits 12 arrivals; a ceiling pinned to one \
             barrier would refuse this launch at 8 arrivals."
        );
        // Blocks multiply across all three grid dimensions.
        assert_eq!(
            super::grid_barrier_arrival_ceiling(one, [4, 3, 2])
                .expect("4x3x2 blocks is representable"),
            24,
            "Fix: the block count is the product of all three grid dimensions."
        );
        // A 1020-block cooperative grid, the measured exatok ceiling at
        // workgroup 256 on this host, with one barrier.
        assert_eq!(
            super::grid_barrier_arrival_ceiling(one, [1020, 1, 1])
                .expect("1020 blocks is representable"),
            1020,
            "Fix: the real cooperative grid width must produce a ceiling equal to its block count."
        );
    }

    /// PTX with no barrier marker must be REFUSED, not silently unbounded.
    ///
    /// The audit derives its bound by counting an emitter comment marker. If that
    /// marker is renamed, a zero count would make the ceiling zero and every
    /// launch would look infinitely healthy, quietly disabling the check that
    /// exists to catch a silent bug. Failing closed here means a rename shows up
    /// as a loud refusal instead of as lost coverage.
    #[test]
    fn missing_barrier_marker_fails_closed_instead_of_disabling_the_audit() {
        let error = super::grid_barrier_arrival_ceiling("bar.sync 0;\n", [4, 1, 1]).expect_err(
            "Fix: PTX with no barrier marker must refuse, because a zero ceiling would silently \
             disable the arrival audit.",
        );
        let message = error.to_string();
        assert!(
            message.contains(super::GRID_BARRIER_PTX_MARKER),
            "Fix: the refusal must name the marker it looked for so the fix is obvious. \
             Got: {message}"
        );
    }

    /// An overflowing ceiling must refuse rather than wrap.
    ///
    /// A wrapped ceiling can land BELOW a healthy arrival count and refuse
    /// correct work, or land absurdly high and accept a stale counter. Neither is
    /// acceptable for a check whose job is to be trusted.
    #[test]
    fn overflowing_arrival_ceiling_is_refused_rather_than_wrapped() {
        let mut ptx = String::new();
        for index in 0..4 {
            ptx.push_str(&format!("// grid.sync barrier #{index}\n"));
        }
        // u32::MAX blocks per dimension across three dimensions overflows u64
        // when multiplied by 4 barriers.
        let grid = [u32::MAX, u32::MAX, u32::MAX];
        assert!(
            super::grid_barrier_arrival_ceiling(&ptx, grid).is_err(),
            "Fix: an overflowing ceiling must refuse; a wrapped ceiling would either refuse \
             healthy launches or accept a stale counter."
        );
    }

    /// Is the gate's busy flag set right now?
    fn gate_is_busy(gate: &std::sync::Arc<super::ModuleGlobalsGate>) -> bool {
        *gate
            .busy
            .lock()
            .expect("Fix: the gate mutex must not be poisoned inside this test.")
    }

    /// A lease holding the gate but addressing no global.
    ///
    /// Exercises the gate lifecycle without a CUDA context: the release
    /// short-circuits before any stream work, so ordering and freeing are
    /// observable on the host.
    fn gate_only_lease(
        gate: &std::sync::Arc<super::ModuleGlobalsGate>,
    ) -> super::ModuleGlobalsLease {
        let guard = super::ModuleGlobalsGate::acquire(gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        super::ModuleGlobalsLease {
            barrier: None,
            trap: None,
            guard: Some(guard),
            arrival_ceiling: 0,
        }
    }

    fn test_error(message: &str) -> super::BackendError {
        super::BackendError::DispatchFailed {
            code: None,
            message: message.to_string(),
        }
    }

    /// The gate must still be HELD while the synchronize runs.
    ///
    /// # The bug this locks out
    ///
    /// Freeing the gate before the synchronize returns lets the next sequence
    /// acquire the counter and memset it to zero while this launch's grid is
    /// still running, so that grid's remaining barriers wait for a release target
    /// that can no longer be reached. The symptom is a HANG rather than an error,
    /// and it reproduces only under cooperative launch. Asserting the gate ends
    /// up free is not enough: that holds for the broken order too. This asserts
    /// the gate is still held DURING the wait.
    #[test]
    fn release_in_order_holds_the_gate_until_the_synchronize_returns() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let guard = super::ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        assert!(
            gate_is_busy(&gate),
            "Fix: acquiring the gate must set the busy flag, or the whole exclusion is inert."
        );
        let held_during_synchronize = std::cell::Cell::new(false);
        let result = super::release_in_order(
            Some(guard),
            || {
                held_during_synchronize.set(gate_is_busy(&gate));
                Ok(())
            },
            || Ok(()),
        );
        assert!(result.is_ok(), "Fix: a clean release must report success.");
        assert!(
            held_during_synchronize.get(),
            "Fix: the gate must remain held while the stream synchronize runs. Releasing first \
             lets the next launch reset the counter under a live grid, which hangs instead of \
             erroring."
        );
        assert!(
            !gate_is_busy(&gate),
            "Fix: the gate must be free once the release returns, or every later cooperative \
             launch of this module blocks forever."
        );
    }

    /// Synchronize, then audit, then free: exactly that order.
    ///
    /// # The bug this locks out
    ///
    /// The audit reads the counter over the bus with a blocking copy, so it is
    /// only meaningful once the stream is quiescent. Auditing before the
    /// synchronize reads a counter that arrivals are still landing in, which
    /// under-reports and makes a missed reset invisible; auditing after the gate
    /// is freed races the next sequence's memset and can read that instead.
    #[test]
    fn release_in_order_audits_after_the_synchronize_and_while_the_gate_is_held() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let guard = super::ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        let steps = std::cell::RefCell::new(Vec::new());
        let audit_saw_gate_held = std::cell::Cell::new(false);
        let result = super::release_in_order(
            Some(guard),
            || {
                steps.borrow_mut().push("synchronize");
                Ok(())
            },
            || {
                steps.borrow_mut().push("audit");
                audit_saw_gate_held.set(gate_is_busy(&gate));
                Ok(())
            },
        );
        assert!(result.is_ok(), "Fix: a clean release must report success.");
        assert_eq!(
            *steps.borrow(),
            vec!["synchronize", "audit"],
            "Fix: the audit must read the counter only after the stream is quiescent, or a \
             missed reset goes unnoticed."
        );
        assert!(
            audit_saw_gate_held.get(),
            "Fix: the audit must run while the gate is still held, or it races the next \
             sequence's memset of the same counter."
        );
    }

    /// A failed synchronize must free the gate and skip the audit.
    ///
    /// # The bug this locks out
    ///
    /// The audit's device-to-host read is only sound once the stream is
    /// quiescent, which a failed synchronize has not established, so running it
    /// anyway reads a counter with launches possibly still in flight and can
    /// refuse healthy work. Returning early instead would leave the busy flag set
    /// and block every later cooperative launch of the module.
    #[test]
    fn release_in_order_frees_the_gate_and_skips_the_audit_when_the_synchronize_fails() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let guard = super::ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        let audited = std::cell::Cell::new(false);
        let result = super::release_in_order(
            Some(guard),
            || Err(test_error("synchronize failed")),
            || {
                audited.set(true);
                Ok(())
            },
        );
        let message = match result {
            Err(super::BackendError::DispatchFailed { message, .. }) => message,
            other => panic!("Fix: a failed synchronize must surface its error. Got: {other:?}"),
        };
        assert_eq!(
            message, "synchronize failed",
            "Fix: the synchronize error must reach the caller unchanged, not be replaced by an \
             audit error from an unsynchronized read."
        );
        assert!(
            !audited.get(),
            "Fix: the audit must not run after a failed synchronize; its counter read is only \
             sound once the stream is quiescent."
        );
        assert!(
            !gate_is_busy(&gate),
            "Fix: a failed synchronize must still free the gate, or one transient stream error \
             wedges every later cooperative launch of this module."
        );
    }

    /// A failed audit must free the gate before reporting.
    ///
    /// # The bug this locks out
    ///
    /// The audit fires exactly when a stale counter is detected, which is already
    /// a bad day. Propagating that refusal without freeing the gate would turn a
    /// diagnosable wrong-data error into a permanent block on every later
    /// cooperative launch of the same module, hiding the message that names the
    /// real defect.
    #[test]
    fn release_in_order_frees_the_gate_when_the_audit_refuses() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let guard = super::ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        let result =
            super::release_in_order(Some(guard), || Ok(()), || Err(test_error("stale counter")));
        assert!(
            result.is_err(),
            "Fix: an audit refusal must reach the caller; a stale counter means the kernel's \
             cross-block reads were wrong even though the launch reported success."
        );
        assert!(
            !gate_is_busy(&gate),
            "Fix: an audit refusal must still free the gate, or the error that names the defect \
             is followed by a hang that hides it."
        );
    }

    /// The launch runs while the gate is held, and its value is returned.
    ///
    /// # The bug this locks out
    ///
    /// The lease exists to give one launch sequence exclusive use of a module's
    /// counter. A helper that released before running the launch, or that dropped
    /// the launch's return value, would satisfy the type checker while removing
    /// the exclusion the counter depends on.
    #[test]
    fn launch_then_release_runs_the_launch_while_the_gate_is_held() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let lease = gate_only_lease(&gate);
        let held_during_launch = std::cell::Cell::new(false);
        let launched =
            lease.launch_then_release(std::ptr::null_mut(), "gate lifetime unit test", |_lease| {
                held_during_launch.set(gate_is_busy(&gate));
                Ok(7_u32)
            });
        assert_eq!(
            launched.expect("Fix: a successful launch closure must return its value."),
            7,
            "Fix: the launch closure's value must reach the caller unchanged."
        );
        assert!(
            held_during_launch.get(),
            "Fix: the launch must run while the lease holds the gate, or two sequences share one \
             module's arrival counter."
        );
        assert!(
            !gate_is_busy(&gate),
            "Fix: the gate must be free once the helper returns."
        );
    }

    /// A failed launch must still end the lease, and must still be reported.
    ///
    /// # The bug this locks out
    ///
    /// This is the case the hand-written shape at the four call sites got wrong
    /// by construction: `launched?` placed before the release returns early, the
    /// guard drops through `Drop` instead of through the release, and the gate is
    /// freed WITHOUT the stream synchronize. The next sequence then resets the
    /// counter under a grid that may still be spinning. Swallowing the error
    /// instead would be worse still: the launch failure would vanish and the
    /// dispatch would report success.
    #[test]
    fn launch_then_release_reports_a_failed_launch_and_still_frees_the_gate() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let lease = gate_only_lease(&gate);
        let result: Result<(), super::BackendError> =
            lease.launch_then_release(std::ptr::null_mut(), "gate lifetime unit test", |_lease| {
                Err(test_error("launch failed"))
            });
        let message = match result {
            Err(super::BackendError::DispatchFailed { message, .. }) => message,
            other => panic!("Fix: a failed launch must surface its error. Got: {other:?}"),
        };
        assert_eq!(
            message, "launch failed",
            "Fix: the launch error must reach the caller unchanged; swallowing it would report a \
             failed dispatch as a success."
        );
        assert!(
            !gate_is_busy(&gate),
            "Fix: a failed launch must still end the lease through the release, or the gate stays \
             set and every later cooperative launch of this module blocks."
        );
    }

    /// A second lease on the same gate must wait for the first to be released.
    ///
    /// # The bug this locks out
    ///
    /// The counter is a module-scope symbol at a fixed address, so two concurrent
    /// sequences launching the same module share one counter: each one's
    /// arrivals inflate the other's barrier targets and its reset zeroes the
    /// other's progress. If the gate did not actually exclude, that corruption
    /// would be invisible until a multi-threaded caller appeared.
    #[test]
    fn a_released_gate_is_acquirable_again_and_a_held_one_is_not() {
        let gate = std::sync::Arc::new(super::ModuleGlobalsGate::default());
        let first = super::ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        assert!(
            gate_is_busy(&gate),
            "Fix: a held gate must report busy, or the exclusion does nothing."
        );
        drop(first);
        assert!(
            !gate_is_busy(&gate),
            "Fix: dropping the guard must clear the busy flag, including on unwind."
        );
        let second = super::ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a released gate must be acquirable by the next sequence.");
        assert!(
            gate_is_busy(&gate),
            "Fix: re-acquiring must set the busy flag again."
        );
        drop(second);
    }
}

/// A live CUDA backend handle bound to a specific device.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    /// Probed device capabilities over the hardware limit.
    pub caps: CudaDeviceCaps,
    pub(crate) ptx_target_sm: u32,
    pub(crate) launch_resources: Arc<crate::stream::CudaLaunchResourcePool>,
    pub(crate) transient_pool: Arc<DeviceAllocationPool>,
    pub(crate) host_pool: Arc<PinnedHostAllocationPool>,
    pub(crate) ptx_source_cache: Arc<CudaPtxSourceCache>,
    module_cache: Arc<CudaModuleCache>,
    pub(crate) resident_store: Arc<CudaResidentStore>,
    pub(crate) validation_cache: Arc<ValidationCache>,
    pub(crate) graph_capture_lock: Arc<Mutex<()>>,
    pub(crate) async_upload_stream: Arc<Mutex<Option<crate::stream::CudaStream>>>,
    pub(crate) telemetry: Arc<CudaTelemetry>,
    /// Cache of driver-measured active-blocks-per-SM keyed by
    /// `(CUfunction as usize, threads_per_block)`: occupancy is constant per
    /// kernel shape, so this makes the per-launch occupancy-evidence query a map
    /// lookup after the first launch instead of repeated FFI (Law 7).
    pub(crate) occupancy_blocks_cache: Arc<DashMap<(usize, u32), u32>>,
    /// Serializing gate per module-cache key for launches that hold a
    /// module-scope global, created on first use. Keyed exactly like the module
    /// cache because that is the aliasing set: one key means one loaded CUmodule
    /// and therefore one `_vyre_grid_barrier` counter and one trap record. See
    /// [`ModuleGlobalsGate`].
    module_globals_gates: Arc<DashMap<ModuleCacheKey, Arc<ModuleGlobalsGate>>>,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaBackend {
    /// Acquire the default CUDA device (ordinal 0).
    pub fn acquire() -> Result<Self, String> {
        Self::acquire_ordinal(0)
    }

    /// Acquire a specific CUDA device by ordinal.
    ///
    /// # Errors
    ///
    /// Returns an error when the CUDA driver cannot initialize, the ordinal is
    /// out of range, or required device attributes cannot be queried.
    pub fn acquire_ordinal(ordinal: usize) -> Result<Self, String> {
        // E4 + E5: enable the CUDA driver's persistent disk JIT cache
        // before any module load so the first dispatch this process
        // does on a previously-seen kernel hits the on-disk cuBIN
        // instead of re-JITing. Idempotent and respectful of operator
        // overrides via the CUDA_CACHE_* env vars.
        crate::jit_cache::configure_jit_cache_default()?;
        let device = CudaDeviceHandle::acquire_ordinal(ordinal)?;
        let caps = device.caps;
        let ptx_target_sm = select_loadable_ptx_target_sm(caps.ptx_target_sm())?;
        let ctx = device.ctx;
        let resident_store = CudaResidentStore::new().map_err(|error| error.to_string())?;
        Ok(Self {
            caps,
            ptx_target_sm,
            launch_resources: Arc::new(crate::stream::CudaLaunchResourcePool::new(
                CUDA_LAUNCH_RESOURCE_CACHE,
            )),
            transient_pool: Arc::new(DeviceAllocationPool::new(TRANSIENT_ALLOCATION_POOL_BYTES)),
            host_pool: Arc::new(PinnedHostAllocationPool::new(PINNED_HOST_POOL_BYTES)),
            ptx_source_cache: Arc::new(CudaPtxSourceCache::new()),
            module_cache: Arc::new(CudaModuleCache::new()),
            resident_store: Arc::new(resident_store),
            validation_cache: Arc::new(ValidationCache::default()),
            graph_capture_lock: Arc::new(Mutex::new(())),
            async_upload_stream: Arc::new(Mutex::new(None)),
            telemetry: Arc::new(CudaTelemetry::default()),
            occupancy_blocks_cache: Arc::new(DashMap::new()),
            module_globals_gates: Arc::new(DashMap::new()),
            ctx,
        })
    }

    pub(crate) fn prepare_launch_plan(
        &self,
        program: &Program,
        bindings: &BindingPlan,
        config: &DispatchConfig,
    ) -> Result<LaunchPlan, BackendError> {
        self.enforce_config_caps(config)?;
        LaunchPlan::from_bindings(program, &bindings.bindings, config, self.launch_limits())
    }

    pub(crate) fn prepare_host_dispatch(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<CudaDispatchPlan, BackendError> {
        let bindings = BindingPlan::from_borrowed_inputs(program, inputs)?;
        let launch = self.prepare_launch_plan(program, &bindings, config)?;
        self.validate_program_cached(program)?;
        let cooperative = self.resolve_cooperative_flag_for_program(program, config)?;
        let output_binding_indices = compute_ordered_output_indices(&bindings)?;
        let fixpoint_iterations = resolve_fixpoint_iterations(config, "CUDA")?;
        Ok(CudaDispatchPlan {
            bindings,
            output_binding_indices,
            launch,
            cooperative,
            fixpoint_iterations,
        })
    }

    /// Cooperative-launch flag for `program`, accounting for its grid-sync
    /// content.
    ///
    /// A program that still contains `MemoryOrdering::GridSync` barriers when it
    /// reaches a launch path has been lowered to PTX with native in-kernel grid
    /// barriers (the resident-fixpoint and host-split paths split the barriers
    /// out before lowering, so they never arrive here). Such a kernel MUST be
    /// launched cooperatively, every CTA co-resident, or the in-kernel grid
    /// barrier deadlocks. Force cooperative and fail closed when the device
    /// cannot run cooperative launch, rather than silently launching a kernel
    /// that would hang.
    ///
    /// Every prepare entrypoint routes through this, not only the borrowed-host
    /// one. A compiled pipeline plans through `prepare_static_dispatch` and its
    /// persistent-handle routes plan through `prepare_resident_dispatch`, so a
    /// grid-sync program compiled without `DispatchConfig::cooperative` set would
    /// otherwise plan a plain `cuLaunchKernel` for a kernel whose barriers
    /// require every CTA to be co-resident.
    fn resolve_cooperative_flag_for_program(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        if vyre_driver::grid_sync::contains_grid_sync(program) {
            if !self.hardware_supports_grid_sync() {
                return Err(BackendError::UnsupportedFeature {
                    name: format!(
                        "cuda_native_grid_sync (compute_capability={:?}, cooperative_launch={})",
                        self.caps.compute_capability, self.caps.cooperative_launch
                    ),
                    backend: crate::CUDA_BACKEND_ID.to_string(),
                });
            }
            return Ok(true);
        }
        self.resolve_cooperative_flag(config)
    }

    pub(crate) fn prepare_static_dispatch(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<CudaDispatchPlan, BackendError> {
        let bindings = BindingPlan::build(program)?;
        let launch = self.prepare_launch_plan(program, &bindings, config)?;
        self.validate_program_cached(program)?;
        let cooperative = self.resolve_cooperative_flag_for_program(program, config)?;
        let output_binding_indices = compute_ordered_output_indices(&bindings)?;
        let fixpoint_iterations = resolve_fixpoint_iterations(config, "CUDA")?;
        Ok(CudaDispatchPlan {
            bindings,
            output_binding_indices,
            launch,
            cooperative,
            fixpoint_iterations,
        })
    }

    pub(crate) fn prepare_resident_dispatch(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
    ) -> Result<CudaDispatchPlan, BackendError> {
        let static_bindings = BindingPlan::build(program)?;
        let required_bindings = static_bindings
            .bindings
            .len()
            .checked_sub(static_bindings.shared_indices.len())
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident binding plan has {} binding(s) but {} shared binding index(es). Rebuild the dispatch plan before launching.",
                    static_bindings.bindings.len(),
                    static_bindings.shared_indices.len()
                ),
            })?;
        if bindings.len() != required_bindings {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident dispatch expected {required_bindings} bound resource(s) but received {}.",
                    bindings.len()
                ),
            });
        }

        let mut input_lengths = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut input_lengths,
            static_bindings.input_indices.len(),
            "resident dispatch input lengths",
        )?;
        input_lengths.extend(std::iter::repeat_n(0, static_bindings.input_indices.len()));
        let mut next_binding = 0usize;
        for binding in &static_bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let source = next_dispatch_binding(
                bindings,
                &mut next_binding,
                "resident dispatch input-length derivation",
            )?;
            let byte_len = match source {
                CudaDispatchBinding::Resident(handle) => self.resident_store.view(handle)?.byte_len,
                CudaDispatchBinding::Borrowed(bytes) => bytes.len(),
            };
            if let Some(input_index) = binding.input_index {
                let Some(input_len) = input_lengths.get_mut(input_index) else {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident dispatch input binding index {input_index} has no matching input-length slot after deriving {} resident input length(s). Rebuild the binding plan before resident launch.",
                            input_lengths.len()
                        ),
                    });
                };
                *input_len = byte_len;
            }
        }

        let bindings = BindingPlan::from_input_lengths(program, &input_lengths)?;
        let launch = self.prepare_launch_plan(program, &bindings, config)?;
        self.validate_program_cached(program)?;
        let cooperative = self.resolve_cooperative_flag_for_program(program, config)?;
        let output_binding_indices = compute_ordered_output_indices(&bindings)?;
        let fixpoint_iterations = resolve_fixpoint_iterations(config, "CUDA")?;
        Ok(CudaDispatchPlan {
            bindings,
            output_binding_indices,
            launch,
            cooperative,
            fixpoint_iterations,
        })
    }

    /// Validate that the caller's cooperative-launch request is consistent
    /// with the device's reported capabilities. Returns the resolved flag
    /// (always `false` when the caller didn't ask) or an `UnsupportedFeature`
    /// error when the caller asked for cooperative launch on a device that
    /// can't run it.
    ///
    /// This method gates *only* the host-side launch API, NOT the codegen
    /// emission of in-kernel grid-sync barriers. The barrier emission is
    /// still controlled by `lowers_grid_sync()`. Callers that opt into
    /// cooperative launch but whose program does not contain any GridSync
    /// barriers get the cooperative API call (resident grid) but no
    /// in-kernel sync sequence  -  the launcher still runs faster on programs
    /// that benefit from a resident grid even without explicit grid-sync.
    fn resolve_cooperative_flag(&self, config: &DispatchConfig) -> Result<bool, BackendError> {
        if !config.cooperative {
            return Ok(false);
        }
        if !self.hardware_supports_grid_sync() {
            return Err(BackendError::UnsupportedFeature {
                name: format!(
                    "cuda_cooperative_launch (compute_capability={:?}, cooperative_launch={})",
                    self.caps.compute_capability, self.caps.cooperative_launch
                ),
                backend: crate::CUDA_BACKEND_ID.to_string(),
            });
        }
        Ok(true)
    }

    fn enforce_config_caps(&self, config: &DispatchConfig) -> Result<(), BackendError> {
        if matches!(config.speculation, Some(SpeculationMode::Force)) {
            return Err(BackendError::UnsupportedFeature {
                name: "speculative dispatch".to_string(),
                backend: crate::CUDA_BACKEND_ID.to_string(),
            });
        }
        Ok(())
    }

    /// Pre-warmup: ensures the CUDA context is active.
    pub fn warmup(&self) -> Result<(), BackendError> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| BackendError::DispatchFailed {
                code: None,
                message: format!("CUDA context bind failed: {e}"),
            })
    }

    /// Cleanup: sync and release cached modules.
    pub fn cleanup(&self) -> Result<(), BackendError> {
        self.warmup()?;
        self.ptx_source_cache.clear();
        self.module_cache.clear();
        self.resident_store.clear()?;
        self.transient_pool.clear()?;
        self.host_pool.clear()?;
        self.launch_resources.clear()?;
        Ok(())
    }

    pub(crate) fn with_resident<T>(
        &self,
        handle: CudaResidentBuffer,
        f: impl FnOnce(ResidentBufferView) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        self.warmup()?;
        let buffer = self.resident_store.view(handle)?;
        f(buffer)
    }

    pub(crate) fn resident_handles_from_resources(
        &self,
        resources: &[vyre_driver::Resource],
    ) -> Result<SmallVec<[CudaResidentBuffer; 8]>, BackendError> {
        self.resident_store.handles_from_resources(resources)
    }

    /// Resolve a dispatch resource list into mixed resident/borrowed bindings.
    pub(crate) fn resident_bindings_from_resources<'a>(
        &self,
        resources: &'a [vyre_driver::Resource],
    ) -> Result<SmallVec<[CudaDispatchBinding<'a>; 8]>, BackendError> {
        self.resident_store.bindings_from_resources(resources)
    }

    pub(crate) fn resident_handle_from_resource(
        &self,
        resource: &vyre_driver::Resource,
    ) -> Result<CudaResidentBuffer, BackendError> {
        self.resident_store.handle_from_resource(resource)
    }

    pub(crate) fn module_cache_key_for_ptx_source_key(
        &self,
        ptx_source_key: PtxSourceCacheKey,
    ) -> Result<ModuleCacheKey, BackendError> {
        self.module_cache
            .key_for_ptx_source_key(ptx_source_key, self.caps.compute_capability)
    }

    pub(crate) fn module_cache_key_for_raw_ptx_artifact(
        &self,
        raw_ptx_source: &str,
    ) -> Result<ModuleCacheKey, BackendError> {
        self.module_cache
            .key_for_raw_ptx_artifact(raw_ptx_source, self.caps.compute_capability)
    }

    pub(crate) fn module_for_ptx_with_key(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
    ) -> Result<cudarc::driver::sys::CUfunction, BackendError> {
        self.module_cache
            .function_for_ptx(ptx_src, key, self.ptx_target_sm())
    }

    /// The module-scope globals this PTX module exposes to the host.
    pub(crate) fn module_globals_with_key(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
    ) -> Result<ModuleGlobals, BackendError> {
        self.module_cache
            .module_globals_for_ptx(ptx_src, key, self.ptx_target_sm())
    }

    /// The module-scope `_vyre_grid_barrier` counter this launch must start from
    /// zero, or `None` when the launch needs no reset.
    ///
    /// A native grid-sync kernel drives the counter up to `N * gridSize` for `N`
    /// in-kernel barriers, and each barrier's release target is a compile-time
    /// multiple of `gridSize`. A launch that starts from a stale value therefore
    /// releases its first barrier before every CTA has arrived. Every cooperative
    /// launch site takes a [`ModuleGlobalsLease`] over it, which zeroes it before
    /// each launch, so the borrowed-host path, the resident paths, and the
    /// compiled pipeline that reuses them share ONE reset instead of drifting
    /// copies.
    ///
    /// A grid-sync program whose loaded module declares no counter is a codegen
    /// failure, not a launch to attempt quietly.
    fn grid_barrier_reset_target(
        &self,
        program: &Program,
        prepared: &CudaDispatchPlan,
        globals: &ModuleGlobals,
    ) -> Result<Option<(u64, usize)>, BackendError> {
        if !prepared.cooperative || !vyre_driver::grid_sync::contains_grid_sync(program) {
            return Ok(None);
        }
        match globals.grid_barrier {
            Some(global) => Ok(Some(global)),
            None => Err(BackendError::InvalidProgram {
                fix:
                    "Fix: CUDA cooperative grid-sync launch found no `_vyre_grid_barrier` counter in the loaded module although the program contains grid-sync barriers. Ensure the PTX emitter declares the module-scope counter for grid-sync kernels."
                        .to_string(),
            }),
        }
    }

    /// Exclusive lease on this module's host-visible module-scope globals for one
    /// launch sequence, or an inert lease when the module has none.
    ///
    /// Acquiring the lease BLOCKS while another launch sequence on the same module
    /// is still in flight, which is what makes a per-module global safe to share
    /// across concurrent dispatches. Hold it across the resets and launches, then
    /// end it with [`ModuleGlobalsLease::launch_then_release`].
    ///
    /// A trap-declaring module takes the lease whether or not the launch is
    /// cooperative, because the trap record is per-module exactly as the counter
    /// is: two overlapping launches would zero each other's record and the second
    /// one's trap would be reported against the first one's launch, or lost.
    pub(crate) fn lease_module_globals(
        &self,
        program: &Program,
        prepared: &CudaDispatchPlan,
        ptx_src: &str,
        module_key: ModuleCacheKey,
    ) -> Result<ModuleGlobalsLease, BackendError> {
        let globals = self.module_globals_with_key(ptx_src, module_key)?;
        let barrier = self.grid_barrier_reset_target(program, prepared, &globals)?;
        let trap = globals.trap;
        if barrier.is_none() && trap.is_none() {
            return Ok(ModuleGlobalsLease {
                barrier: None,
                trap: None,
                guard: None,
                arrival_ceiling: 0,
            });
        }
        let arrival_ceiling = if barrier.is_some() {
            grid_barrier_arrival_ceiling(ptx_src, prepared.launch.grid)?
        } else {
            0
        };
        let gate = Arc::clone(
            self.module_globals_gates
                .entry(module_key)
                .or_insert_with(|| Arc::new(ModuleGlobalsGate::default()))
                .value(),
        );
        let guard = ModuleGlobalsGate::acquire(&gate)?;
        Ok(ModuleGlobalsLease {
            barrier,
            trap,
            guard: Some(guard),
            arrival_ceiling,
        })
    }

    /// Number of loaded CUDA modules currently held in the warm cache.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the cache lock is poisoned.
    pub fn cached_module_count(&self) -> Result<usize, BackendError> {
        Ok(self.module_cache.len())
    }

    /// Compiled module cache counters for honest compile telemetry.
    #[must_use]
    pub fn pipeline_cache_snapshot(&self) -> vyre_driver::PipelineCacheSnapshot {
        self.module_cache.snapshot()
    }

    /// PTX source cache counters for pre-module-load lowering telemetry.
    #[must_use]
    pub fn ptx_source_cache_snapshot(&self) -> CudaPtxSourceCacheSnapshot {
        self.ptx_source_cache.snapshot()
    }

    /// Runtime CUDA telemetry counters for launches, copies, readbacks, and syncs.
    ///
    /// The transient device-allocation-pool hit/miss counters are overlaid here
    /// from the pool itself (their source of truth). `CudaTelemetry` does not hold
    /// the pool, so a bare `CudaTelemetry::snapshot` reports them as zero and this
    /// boundary fills in the real values (ONE PLACE for the count, read once here).
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CudaTelemetrySnapshot {
        let mut snapshot = self.telemetry.snapshot();
        snapshot.device_pool_hits = self.transient_pool.hits();
        snapshot.device_pool_misses = self.transient_pool.misses();
        snapshot
    }

    /// Reset runtime CUDA telemetry counters without clearing caches or resident buffers.
    pub fn reset_telemetry(&self) {
        self.telemetry.reset();
        // Reset the pool hit/miss counters into the same epoch so the hit rate
        // reflects the window measured after the reset, not lifetime-of-process.
        self.transient_pool.reset_hit_counters();
    }

    /// Bytes of transient CUDA device memory retained for dispatch reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the allocation-pool lock is poisoned.
    pub fn cached_transient_allocation_bytes(&self) -> Result<usize, BackendError> {
        self.transient_pool.cached_bytes()
    }

    /// Bytes of transient CUDA device memory currently owned by the transient pool.
    ///
    /// This includes both checked-out allocations and cached allocations retained for reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if allocation accounting cannot be read.
    pub fn allocated_transient_allocation_bytes(&self) -> Result<usize, BackendError> {
        self.transient_pool.allocated_bytes()
    }

    /// Cached CUDA streams/events retained for dispatch reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if a launch-resource pool lock is poisoned.
    pub fn cached_launch_resource_counts(&self) -> Result<(usize, usize), BackendError> {
        self.launch_resources.cached_counts()
    }

    /// Detailed cached CUDA launch resources retained for dispatch reuse,
    /// including timing-enabled events used by CUDA graph replay telemetry.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if launch-resource accounting cannot be read.
    pub fn cached_launch_resource_counts_detailed(
        &self,
    ) -> Result<crate::CudaLaunchResourceCounts, BackendError> {
        self.launch_resources.cached_counts_detailed()
    }

    /// Snapshot the driver-tier observability surface
    /// ([`vyre_driver::observability::DriverObservability`]) plus the
    /// cuda module-cache count as a single backend metric.
    ///
    /// Operators scrape this in addition to per-substrate Prometheus
    /// counters when correlating substrate activity with backend
    /// resource usage.
    #[must_use]
    pub fn observability_snapshot(&self) -> vyre_driver::observability::DriverObservability {
        vyre_driver::observability::DriverObservability::snapshot()
    }

    /// PTX disk-cache directory path. Reuses the shared on-disk pipeline-cache
    /// layout, keyed by the VSA fingerprint.
    ///
    /// P-CUDA-2: PTX/CUBIN blobs persist across runs in this directory
    /// so first-run compile cost amortizes over the cluster.
    pub fn ptx_disk_cache_dir() -> Result<std::path::PathBuf, BackendError> {
        if let Some(path) = std::env::var_os("VYRE_PTX_CACHE_DIR") {
            let path = std::path::PathBuf::from(path);
            if path.as_os_str().is_empty() {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: VYRE_PTX_CACHE_DIR is empty. Set it to a writable persistent directory or unset it so XDG/HOME cache discovery can run."
                        .to_string(),
                });
            }
            return Ok(path);
        }
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            return Ok(std::path::PathBuf::from(xdg).join("vyre").join("ptx-cache"));
        }
        if let Some(home) = std::env::var_os("HOME") {
            return Ok(std::path::PathBuf::from(home)
                .join(".cache")
                .join("vyre")
                .join("ptx-cache"));
        }
        Err(BackendError::InvalidProgram {
            fix: "Fix: CUDA PTX disk cache has no VYRE_PTX_CACHE_DIR, XDG_CACHE_HOME, or HOME. Configure a writable persistent cache root; temporary fallback is forbidden for production compile performance."
                .to_string(),
        })
    }
}

/// Serializes launch sequences that share one loaded module's host-visible
/// module-scope globals: the `_vyre_grid_barrier` arrival counter and the trap
/// record.
///
/// # Why serializing is required rather than merely tidy
///
/// The counter is a MODULE-scope global, and a barrier releases when it reaches
/// `(barrier_index + 1) * gridSize`. Two launches of one module that are in
/// flight together therefore interfere in both directions, and both were
/// measured on an RTX 5090 before this gate existed:
///
/// - Their arrivals MIX. Barrier 0's target is one `gridSize`, so it is reached
///   after `gridSize` arrivals drawn from either grid, releasing both before
///   either has fully arrived. Blocks then read each other's pre-barrier state
///   and the output is wrong with no error raised.
/// - Their resets CLOBBER. One launch's zeroing lands while the other's blocks
///   are spinning, dropping the counter below a target that was already met, so
///   that grid spins forever. That is a hang, not a wrong answer.
///
/// This is reachable with no threads at all: a compiled pipeline's batched
/// dispatch enqueues each batch element as its own async dispatch on its own
/// stream, so the elements overlap on the device and alias the counter.
///
/// # The cost, stated plainly
///
/// This is a serialization, not a way to make concurrent cooperative launches
/// work. Cooperative grid-sync launches that share a module now run one at a
/// time, and the lease is held until the launch COMPLETES because the counter
/// stays live for the kernel's whole execution. Launches on different modules,
/// on independent backends, and every launch of a module with neither global are
/// untouched.
///
/// The throughput this gives up is small in practice and unavailable in
/// principle: a cooperative launch requires every block co-resident, so a
/// grid-sync grid is sized to fill the device and a second one had nowhere to run
/// concurrently anyway. Removing the serialization requires giving each launch
/// its OWN counter, which means the counter's address has to become a launch
/// input rather than a module-scope symbol. That is an emitter and kernel-ABI
/// change, not a driver-side one.
///
/// # The trap record has the same shape
///
/// The trap record is also a module-scope global, zeroed before a launch and read
/// after it. Two overlapping launches of one trap-declaring module would zero
/// each other's record: the second launch's zeroing erases a trap the first
/// launch already recorded, so a launch that trapped is reported as successful
/// and its output is read as an answer. That is the same class of failure as a
/// clobbered counter and it takes the same gate, which is why one gate per module
/// covers both rather than two gates racing for the same module.
///
/// A trap-declaring launch is serialized whether or not it is cooperative, and
/// unlike a cooperative launch it has no residency argument saying a second one
/// had nowhere to run. That cost is real: concurrent launches of one trapping
/// module now run one at a time. It buys the ability to report the trap at all,
/// which is the difference between a refusal and a wrong answer.
#[derive(Debug, Default)]
pub(crate) struct ModuleGlobalsGate {
    busy: Mutex<bool>,
    free: Condvar,
}

impl ModuleGlobalsGate {
    /// Block until no other launch sequence holds this module's globals.
    fn acquire(gate: &Arc<Self>) -> Result<ModuleGlobalsGuard, BackendError> {
        let mut busy = gate.busy.lock().map_err(|_| BackendError::DispatchFailed {
            code: None,
            message:
                "CUDA module-globals gate mutex was poisoned by a panicking launch. Fix: a launch panicked while holding the module's grid-barrier counter or trap record; treat the earlier panic as the defect."
                    .to_string(),
        })?;
        while *busy {
            busy = gate.free.wait(busy).map_err(|_| BackendError::DispatchFailed {
                code: None,
                message:
                    "CUDA module-globals gate mutex was poisoned while waiting for the module's globals. Fix: a launch panicked while holding them; treat the earlier panic as the defect."
                        .to_string(),
            })?;
        }
        *busy = true;
        drop(busy);
        Ok(ModuleGlobalsGuard {
            gate: Arc::clone(gate),
        })
    }
}

/// Releases its [`ModuleGlobalsGate`] on drop, including on unwind, so a
/// panicking launch cannot wedge every later launch of the same module.
#[derive(Debug)]
pub(crate) struct ModuleGlobalsGuard {
    gate: Arc<ModuleGlobalsGate>,
}

impl Drop for ModuleGlobalsGuard {
    fn drop(&mut self) {
        // Recover the poisoned guard rather than propagating: failing to clear
        // the flag here would block every future launch on this module, which is
        // strictly worse than continuing after someone else's panic.
        let mut busy = self
            .gate
            .busy
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *busy = false;
        drop(busy);
        self.gate.free.notify_one();
    }
}

/// One launch sequence's exclusive hold on a module's host-visible module-scope
/// globals, plus the globals themselves.
///
/// An inert lease (a module that declares neither the grid-barrier counter nor a
/// trap record) holds nothing and enqueues nothing, so an ordinary launch pays
/// only two moved `Option`s.
#[derive(Debug)]
pub(crate) struct ModuleGlobalsLease {
    /// The cooperative grid-barrier counter, when this launch must reset one.
    barrier: Option<(u64, usize)>,
    /// The trap record, when the loaded module declares one.
    trap: Option<Arc<TrapSidecar>>,
    guard: Option<ModuleGlobalsGuard>,
    /// Arrivals ONE launch of this kernel can contribute: static barrier count
    /// times grid block count. See [`grid_barrier_arrival_ceiling`]. Zero when no
    /// counter is held.
    arrival_ceiling: u64,
}

impl ModuleGlobalsLease {
    /// Zero the grid-barrier counter on `stream`, ahead of the launch it belongs
    /// to.
    ///
    /// The memset is stream-ordered, so it lands before the kernel that reads the
    /// counter without any host synchronization. Call this before EVERY launch in
    /// the sequence, not once per sequence: each launch drives the counter up to
    /// `barriers * gridSize` and the next launch must start from zero.
    ///
    /// The trap record is deliberately NOT reset here. See
    /// [`ModuleGlobalsLease::launch_then_release`].
    ///
    /// # Safety
    ///
    /// `stream` must be live until the memset completes.
    pub(crate) unsafe fn enqueue_barrier_reset(
        &self,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), BackendError> {
        let Some((counter_ptr, counter_len)) = self.barrier else {
            return Ok(());
        };
        // SAFETY: the pointer came from cuModuleGetGlobal on the module this
        // lease was taken against; stream lifetime is the caller's guarantee.
        unsafe { super::copy::memset_d8_async_checked(counter_ptr, 0, counter_len, stream) }
    }

    /// Zero the trap record on `stream`, once, ahead of the whole sequence.
    ///
    /// # Safety
    ///
    /// `stream` must be live until the memset completes.
    unsafe fn enqueue_trap_reset(
        &self,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), BackendError> {
        let Some(trap) = self.trap.as_deref() else {
            return Ok(());
        };
        // SAFETY: the pointer came from cuModuleGetGlobal on the module this lease
        // was taken against and the byte count was validated against the record
        // size at load; stream lifetime is the caller's guarantee.
        unsafe {
            super::copy::memset_d8_async_checked(trap.device_ptr(), 0, trap.byte_count(), stream)
        }
    }

    /// Zero the trap record, run `launch` under this lease, then end the lease, in
    /// the ONE order that is safe.
    ///
    /// The launch closure's error is captured rather than propagated, so the
    /// release always runs, and the release SYNCHRONIZES `stream` before the gate
    /// is freed. That synchronize is why this order cannot be rearranged and why
    /// the release cannot be moved into the closure or made cheap: the globals are
    /// live for the kernel's whole EXECUTION, not merely until the launch call
    /// returns. Free the gate before the grid finishes and the next sequence
    /// memsets `_vyre_grid_barrier` to zero underneath a still-running kernel,
    /// whose remaining barriers then wait for a release target that can no longer
    /// be reached.
    ///
    /// # Why the trap record is zeroed here and not per launch
    ///
    /// One lease can cover many launches: fixpoint iterations, and every element
    /// of a batched dispatch, all enqueued on one stream and read back once at
    /// release. The device claims the record with a compare-and-swap, so the FIRST
    /// trap in the sequence is the one kept. Zeroing per launch would instead erase
    /// the record an earlier launch had already written, and the read at release
    /// would find zero: a sequence whose second element trapped after its first
    /// element trapped would be reported as successful. Once per sequence makes
    /// "any launch under this lease trapped" the thing the readback answers, which
    /// is the question the caller is asking. The cost is that `lane` identifies the
    /// trapping lane within the sequence, not which launch it belonged to.
    ///
    /// Taking `self` by value is deliberate: it makes skipping the release
    /// unrepresentable at a call site. Hand-writing the sequence instead, as
    /// `let launched = (|| { .. })(); release(..)?; launched?;`, works but puts
    /// the ordering back in four places, and getting it wrong there COMPILES and
    /// passes every test whose program neither traps nor grid-syncs.
    ///
    /// `stream` must be live for the whole sequence, which every caller already
    /// guarantees by owning the stream across the launch.
    pub(crate) fn launch_then_release<T>(
        self,
        stream: cudarc::driver::sys::CUstream,
        label: &'static str,
        launch: impl FnOnce(&Self) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        // SAFETY: the caller owns `stream` across the launch sequence this lease
        // covers, so it outlives the memset enqueued here.
        let reset = unsafe { self.enqueue_trap_reset(stream) };
        if let Err(error) = reset {
            // The gate is freed by dropping the lease. Nothing was launched, so
            // there is nothing to synchronize or read back.
            return Err(error);
        }
        let launched = launch(&self);
        self.release_after_launch(stream, label)?;
        launched
    }

    /// End the lease once the launches have completed.
    ///
    /// When a global is held this SYNCHRONIZES `stream` first, and that is
    /// load-bearing rather than defensive on both counts: the counter is live for
    /// the kernel's whole execution, so releasing at enqueue time would let the
    /// next sequence reset it under a still-running grid, which is the hang
    /// described on [`ModuleGlobalsGate`]; and the trap record is only complete
    /// once the kernel has finished, so reading it before the synchronize would
    /// report no trap on a launch that was about to write one. An inert lease
    /// synchronizes nothing.
    ///
    /// The trap is read BEFORE the arrival audit. A trapped kernel exits early, so
    /// it legitimately skips later barriers and its arrival count is not evidence
    /// of anything; reporting a stale-counter suspicion instead of the trap would
    /// name the wrong defect.
    ///
    /// Private on purpose: [`ModuleGlobalsLease::launch_then_release`] is the only
    /// caller, so no launch site can open-code the release and drift out of
    /// order.
    fn release_after_launch(
        self,
        stream: cudarc::driver::sys::CUstream,
        label: &'static str,
    ) -> Result<(), BackendError> {
        let Self {
            barrier,
            trap,
            guard,
            arrival_ceiling,
        } = self;
        if barrier.is_none() && trap.is_none() {
            return Ok(());
        }
        release_in_order(
            guard,
            || crate::stream::synchronize_raw_stream(stream, label),
            || {
                read_trap_record(trap.as_deref())?;
                audit_arrivals(barrier, arrival_ceiling)
            },
        )
    }
}

/// Read the trap record and refuse the launch if a lane wrote one.
///
/// # The bug this locks out
///
/// The device cannot return an error. A trapping kernel branches to its exit
/// label, so the launch reports success, `cuStreamSynchronize` reports success,
/// and the output buffers hold whatever the lanes had written before the guard
/// fired. Without this read the caller gets that as an answer. This target
/// advertises `supports_trap_propagation`, so a program with a trap is admitted
/// rather than refused up front, and this read is what makes the advertisement
/// true.
///
/// The caller MUST have synchronized `stream` first: a record read before the
/// kernel finishes says nothing about whether it trapped.
fn read_trap_record(trap: Option<&TrapSidecar>) -> Result<(), BackendError> {
    let Some(trap) = trap else {
        return Ok(());
    };
    let mut record = [0_u8; TRAP_RECORD_BYTES];
    // SAFETY: the pointer came from cuModuleGetGlobal for this module and
    // addresses at least TRAP_RECORD_BYTES, checked when the module was loaded.
    // The caller synchronized the stream, so the kernel's writes have landed and
    // no launch is in flight against this record.
    unsafe {
        super::copy::d2h_sync_checked(
            record.as_mut_ptr().cast::<std::ffi::c_void>(),
            trap.device_ptr(),
            TRAP_RECORD_BYTES,
        )?;
    }
    match decode_trap_record(&record)? {
        None => Ok(()),
        Some(decoded) => Err(BackendError::DispatchFailed {
            code: None,
            message: format!(
                "cuda dispatch trapped: {}",
                decoded.describe(|code| trap.tag_for_code(code))
            ),
        }),
    }
}

/// Synchronize, audit, and only THEN free the gate.
///
/// # The bug this locks out
///
/// Dropping the guard before the synchronize has returned frees the gate while
/// the grid may still be running, and the next sequence's reset then lands under
/// a live kernel: its remaining barriers wait on a target that can no longer be
/// reached, so the symptom is a hang rather than an error. Both fallible steps
/// therefore run BEFORE the drop, and their error is returned AFTER it, so a
/// failed synchronize or a failed audit still frees the gate instead of leaving
/// the busy flag set.
fn release_in_order(
    guard: Option<ModuleGlobalsGuard>,
    synchronize: impl FnOnce() -> Result<(), BackendError>,
    audit: impl FnOnce() -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let audited = synchronize().and_then(|()| audit());
    drop(guard);
    audited
}

/// Verify the counter did not exceed what ONE launch can contribute.
///
/// # The bug this locks out
///
/// A grid barrier releases when the module-scope counter reaches
/// `(index + 1) * gridSize`. A launch that starts from a STALE counter finds
/// that already satisfied and every barrier becomes a pass-through no-op: the
/// kernel returns success, the driver reports no error, and the only symptom is
/// wrong data. That is what shipped on the resident dispatch paths until the
/// per-launch reset landed, and it cost a day of chasing a flake.
///
/// A CTA records its arrival BEFORE it spins, so a no-op barrier still counts.
/// The counter after one reset-then-launch sequence therefore cannot exceed
/// `barriers * gridSize`, and a missed reset shows up as a MULTIPLE of that
/// bound. Checking the upper bound rather than exact equality is deliberate: a
/// grid-uniform early exit legitimately skips later barriers and leaves the
/// counter BELOW the bound, which is correct and must not be flagged.
fn audit_arrivals(target: Option<(u64, usize)>, arrival_ceiling: u64) -> Result<(), BackendError> {
    let Some((counter_ptr, _)) = target else {
        return Ok(());
    };
    if arrival_ceiling == 0 {
        return Ok(());
    }
    let mut observed = 0_u32;
    // SAFETY: the pointer came from cuModuleGetGlobal for this module and
    // addresses the 4-byte counter; the caller synchronized the stream, so every
    // arrival from this sequence has landed and no launch is in flight.
    unsafe {
        super::copy::d2h_sync_checked(
            std::ptr::from_mut(&mut observed).cast::<std::ffi::c_void>(),
            counter_ptr,
            std::mem::size_of::<u32>(),
        )?;
    }
    verify_arrival_count(observed, arrival_ceiling)
}

/// Marker the PTX emitter writes once per emitted grid-sync barrier.
///
/// The static barrier count is not carried on the program or the plan, and the
/// emitted PTX is the only place it exists by the time a launch site needs it.
/// Pinned by a test so the coupling breaks loudly if the emitter's comment
/// changes rather than silently disabling the audit.
pub(crate) const GRID_BARRIER_PTX_MARKER: &str = "grid.sync barrier #";

/// Arrivals ONE launch can contribute: static barrier count times grid blocks.
///
/// Every CTA records one arrival per barrier it reaches, and grid-sync barriers
/// are top-level, so a launch of a `b`-barrier kernel over `g` blocks contributes
/// at most `b * g`.
fn grid_barrier_arrival_ceiling(ptx_src: &str, grid: [u32; 3]) -> Result<u64, BackendError> {
    let barriers = ptx_src.matches(GRID_BARRIER_PTX_MARKER).count();
    if barriers == 0 {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA cooperative grid-sync launch found no `{GRID_BARRIER_PTX_MARKER}` marker in PTX that declares the _vyre_grid_barrier counter, so the barrier-arrival audit cannot bound the counter. Keep the emitter's per-barrier comment marker, or carry the barrier count on the dispatch plan instead."
            ),
        });
    }
    // Checked throughout: the block product alone overflows u64 for a maximal
    // three-dimensional grid, and this runs on the launch path, so a panicking
    // multiply here would turn an audit into a crash.
    u64::from(grid[0])
        .checked_mul(u64::from(grid[1]))
        .and_then(|blocks| blocks.checked_mul(u64::from(grid[2])))
        .and_then(|blocks| u64::try_from(barriers).ok()?.checked_mul(blocks))
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA grid-sync barrier arrival ceiling overflowed for {barriers} barrier(s) over grid {grid:?}. Reduce the grid or the barrier count."
            ),
        })
}

/// Refuse an arrival count above what ONE launch can contribute.
///
/// Split from the device read so the refusal itself is unit-testable: proving
/// the audit FIRES otherwise means deliberately skipping a counter reset in
/// production code, and the audit exists precisely because that failure is
/// silent.
fn verify_arrival_count(observed: u32, ceiling: u64) -> Result<(), BackendError> {
    if u64::from(observed) <= ceiling {
        return Ok(());
    }
    Err(BackendError::DispatchFailed {
        code: None,
        message: format!(
            "CUDA cooperative grid-sync launch left the module-scope _vyre_grid_barrier counter at {observed} arrivals, above the {ceiling} that one launch of this kernel can contribute (static barrier count times grid blocks). The counter was not zeroed before this launch, so its barriers released on arrival instead of waiting and the kernel's cross-block reads are WRONG even though the launch reported success. Fix: every cooperative launch site must hold a ModuleGlobalsLease and call enqueue_barrier_reset before EACH launch; a new launch path that skips it reintroduces silent wrong answers from the second launch onward."
        ),
    })
}
