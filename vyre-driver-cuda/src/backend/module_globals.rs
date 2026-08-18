use std::sync::{Arc, Condvar, Mutex};

use vyre_driver::trap_record::{decode_trap_record, TRAP_RECORD_BYTES};
use vyre_driver::BackendError;

use super::module_cache::TrapSidecar;

/// Marker the PTX emitter writes once per emitted grid-sync barrier.
///
/// The static barrier count is not carried on the program or the plan, and the
/// emitted PTX is the only place it exists by the time a launch site needs it.
/// Pinned by a test so the coupling breaks loudly if the emitter's comment
/// changes rather than silently disabling the audit.
pub(crate) const GRID_BARRIER_PTX_MARKER: &str = "grid.sync barrier #";

/// Serializes launch sequences that share one loaded module's host-visible
/// module-scope globals: the `_vyre_grid_barrier` arrival counter and the trap
/// record.
///
/// # Why serializing is required rather than merely tidy
///
/// The counter is a MODULE-scope global, and a barrier releases when it reaches
/// `(barrier_index + 1) * gridSize`. Two launches of one module that are in
/// flight together therefore interfere in both directions, and both were
/// observed during multi-stream verification before this serialization existed:
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
    pub(crate) fn acquire(gate: &Arc<Self>) -> Result<ModuleGlobalsGuard, BackendError> {
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
    pub(crate) barrier: Option<(u64, usize)>,
    /// The trap record, when the loaded module declares one.
    pub(crate) trap: Option<Arc<TrapSidecar>>,
    pub(crate) guard: Option<ModuleGlobalsGuard>,
    /// Arrivals ONE launch of this kernel can contribute: static barrier count
    /// times grid block count. See [`grid_barrier_arrival_ceiling`]. Zero when no
    /// counter is held.
    pub(crate) arrival_ceiling: u64,
}

impl ModuleGlobalsLease {
    pub(crate) fn new(
        barrier: Option<(u64, usize)>,
        trap: Option<Arc<TrapSidecar>>,
        guard: Option<ModuleGlobalsGuard>,
        arrival_ceiling: u64,
    ) -> Self {
        Self {
            barrier,
            trap,
            guard,
            arrival_ceiling,
        }
    }

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
    pub(crate) fn launch_then_release<T>(
        self,
        stream: cudarc::driver::sys::CUstream,
        label: &'static str,
        launch: impl FnOnce(&Self) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        // SAFETY: the caller owns `stream` across the launch sequence this lease
        // covers, so it outlives the memset enqueued here.
        unsafe { self.enqueue_trap_reset(stream) }?;
        let launched = launch(&self);
        self.release_after_launch(stream, label)?;
        launched
    }

    /// Run the launches and hand the lease back to be ended at completion.
    pub(crate) fn launch_then_defer_release<T>(
        self,
        stream: cudarc::driver::sys::CUstream,
        label: &'static str,
        launch: impl FnOnce(&Self) -> Result<T, BackendError>,
    ) -> Result<(T, Self), BackendError> {
        // SAFETY: the caller owns `stream` across the launch sequence this lease
        // covers, so it outlives the memset enqueued here.
        unsafe { self.enqueue_trap_reset(stream) }?;
        match launch(&self) {
            Ok(value) => Ok((value, self)),
            Err(error) => {
                self.release_after_launch(stream, label)?;
                Err(error)
            }
        }
    }

    /// End the lease once the completion event has been awaited.
    pub(crate) fn release_after_completion(self) -> Result<(), BackendError> {
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
            || Ok(()),
            || {
                read_trap_record(trap.as_deref())?;
                audit_arrivals(barrier, arrival_ceiling)
            },
        )
    }

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

pub(crate) fn read_trap_record(trap: Option<&TrapSidecar>) -> Result<(), BackendError> {
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

pub(crate) fn release_in_order(
    guard: Option<ModuleGlobalsGuard>,
    synchronize: impl FnOnce() -> Result<(), BackendError>,
    audit: impl FnOnce() -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let audited = synchronize().and_then(|()| audit());
    drop(guard);
    audited
}

pub(crate) fn audit_arrivals(
    target: Option<(u64, usize)>,
    arrival_ceiling: u64,
) -> Result<(), BackendError> {
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

pub(crate) fn grid_barrier_arrival_ceiling(
    ptx_src: &str,
    grid: [u32; 3],
) -> Result<u64, BackendError> {
    let barriers = ptx_src.matches(GRID_BARRIER_PTX_MARKER).count();
    if barriers == 0 {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA cooperative grid-sync launch found no `{GRID_BARRIER_PTX_MARKER}` marker in PTX that declares the _vyre_grid_barrier counter, so the barrier-arrival audit cannot bound the counter. Keep the emitter's per-barrier comment marker, or carry the barrier count on the dispatch plan instead."
            ),
        });
    }
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

pub(crate) fn verify_arrival_count(observed: u32, ceiling: u64) -> Result<(), BackendError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_at_exactly_twice_the_ceiling_is_the_missed_reset_signature_and_fires() {
        let error = verify_arrival_count(8, 4).expect_err(
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
        assert!(
            verify_arrival_count(16, 4).is_err(),
            "Fix: any multiple of the ceiling is a missed reset and must fire, not just 2x."
        );
    }

    #[test]
    fn counter_one_past_the_ceiling_fires() {
        assert!(
            verify_arrival_count(5, 4).is_err(),
            "Fix: exceeding the ceiling by one must fire; slack would hide a partial reset."
        );
        assert!(
            verify_arrival_count(1021, 1020).is_err(),
            "Fix: the boundary must hold at the real cooperative grid width measured on this host \
             (1020 blocks at workgroup 256), not only at toy values."
        );
    }

    #[test]
    fn counter_exactly_at_the_ceiling_passes() {
        assert!(
            verify_arrival_count(4, 4).is_ok(),
            "Fix: a healthy launch lands exactly at the ceiling and must pass."
        );
        assert!(
            verify_arrival_count(8160, 8160).is_ok(),
            "Fix: a healthy launch must pass at a realistic width too (8 barriers over 1020 \
             blocks), where an accidental narrower integer type would also show up."
        );
    }

    #[test]
    fn counter_below_the_ceiling_passes_because_a_grid_uniform_early_exit_is_legitimate() {
        assert!(
            verify_arrival_count(2, 4).is_ok(),
            "Fix: an early exit that skips later barriers lands below the ceiling and is correct."
        );
        assert!(
            verify_arrival_count(1020, 8160).is_ok(),
            "Fix: a converged encode that clears only the first of 8 barrier sites over 1020 \
             blocks is correct and must not be refused."
        );
    }

    #[test]
    fn zero_counter_passes_and_is_reachable_by_an_early_exit_before_the_first_barrier() {
        assert!(
            verify_arrival_count(0, 4).is_ok(),
            "Fix: a grid that exits before its first barrier leaves the counter at zero, which is \
             correct and must not fire."
        );
        assert!(
            verify_arrival_count(0, 8160).is_ok(),
            "Fix: zero must pass regardless of how large the ceiling is."
        );
    }

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

        let barriers = ptx.matches(GRID_BARRIER_PTX_MARKER).count();
        assert_eq!(
            barriers, 8,
            "Fix: four waves emit two grid barriers each, so eight barrier markers must appear. \
             Got {barriers}. A lower count means the ceiling under-counts arrivals and the audit \
             would refuse healthy launches; a higher count means it over-counts and the audit \
             stops detecting a missed reset."
        );

        let ceiling = grid_barrier_arrival_ceiling(&ptx, [1020, 1, 1])
            .expect("Fix: a program with barrier markers must yield a ceiling.");
        assert_eq!(
            ceiling, 8160,
            "Fix: eight barriers over 1020 blocks admit exactly 8160 arrivals (8 * 1020). A \
             ceiling that ignored the barrier count would compute 1020 and refuse this program's \
             every healthy launch; one that over-counted would never fire at all."
        );
        assert_eq!(
            grid_barrier_arrival_ceiling(&ptx, [4, 1, 1]).expect("a 4-block grid yields a ceiling"),
            32,
            "Fix: the ceiling must scale with the launch grid; eight barriers over 4 blocks admit \
             32 arrivals."
        );
    }

    #[test]
    fn arrival_ceiling_is_barrier_count_times_block_count() {
        let one = "// grid.sync barrier #0 target\nbar.sync 0;\n";
        assert_eq!(
            grid_barrier_arrival_ceiling(one, [4, 1, 1])
                .expect("one barrier over 4 blocks is representable"),
            4,
            "Fix: one barrier over 4 blocks admits exactly 4 arrivals."
        );
        let three = "// grid.sync barrier #0\n// grid.sync barrier #1\n// grid.sync barrier #2\n";
        assert_eq!(
            grid_barrier_arrival_ceiling(three, [4, 1, 1])
                .expect("three barriers over 4 blocks is representable"),
            12,
            "Fix: three barriers over 4 blocks admits 12 arrivals; a ceiling pinned to one \
             barrier would refuse this launch at 8 arrivals."
        );
        assert_eq!(
            grid_barrier_arrival_ceiling(one, [4, 3, 2]).expect("4x3x2 blocks is representable"),
            24,
            "Fix: the block count is the product of all three grid dimensions."
        );
        assert_eq!(
            grid_barrier_arrival_ceiling(one, [1020, 1, 1]).expect("1020 blocks is representable"),
            1020,
            "Fix: the real cooperative grid width must produce a ceiling equal to its block count."
        );
    }

    #[test]
    fn missing_barrier_marker_fails_closed_instead_of_disabling_the_audit() {
        let error = grid_barrier_arrival_ceiling("bar.sync 0;\n", [4, 1, 1]).expect_err(
            "Fix: PTX with no barrier marker must refuse, because a zero ceiling would silently \
             disable the arrival audit.",
        );
        let message = error.to_string();
        assert!(
            message.contains(GRID_BARRIER_PTX_MARKER),
            "Fix: the refusal must name the marker it looked for so the fix is obvious. \
             Got: {message}"
        );
    }

    #[test]
    fn overflowing_arrival_ceiling_is_refused_rather_than_wrapped() {
        let mut ptx = String::new();
        for index in 0..4 {
            ptx.push_str(&format!("// grid.sync barrier #{index}\n"));
        }
        let grid = [u32::MAX, u32::MAX, u32::MAX];
        assert!(
            grid_barrier_arrival_ceiling(&ptx, grid).is_err(),
            "Fix: an overflowing ceiling must refuse; a wrapped ceiling would either refuse \
             healthy launches or accept a stale counter."
        );
    }

    fn gate_is_busy(gate: &Arc<ModuleGlobalsGate>) -> bool {
        *gate
            .busy
            .lock()
            .expect("Fix: the gate mutex must not be poisoned inside this test.")
    }

    fn gate_only_lease(gate: &Arc<ModuleGlobalsGate>) -> ModuleGlobalsLease {
        let guard = ModuleGlobalsGate::acquire(gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        ModuleGlobalsLease {
            barrier: None,
            trap: None,
            guard: Some(guard),
            arrival_ceiling: 0,
        }
    }

    fn test_error(message: &str) -> BackendError {
        BackendError::DispatchFailed {
            code: None,
            message: message.to_string(),
        }
    }

    #[test]
    fn release_in_order_holds_the_gate_until_the_synchronize_returns() {
        let gate = Arc::new(ModuleGlobalsGate::default());
        let guard = ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        assert!(
            gate_is_busy(&gate),
            "Fix: acquiring the gate must set the busy flag, or the whole exclusion is inert."
        );
        let held_during_synchronize = std::cell::Cell::new(false);
        let result = release_in_order(
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

    #[test]
    fn release_in_order_audits_after_the_synchronize_and_while_the_gate_is_held() {
        let gate = Arc::new(ModuleGlobalsGate::default());
        let guard = ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        let steps = std::cell::RefCell::new(Vec::new());
        let audit_saw_gate_held = std::cell::Cell::new(false);
        let result = release_in_order(
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

    #[test]
    fn release_in_order_frees_the_gate_and_skips_the_audit_when_the_synchronize_fails() {
        let gate = Arc::new(ModuleGlobalsGate::default());
        let guard = ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        let audited = std::cell::Cell::new(false);
        let result = release_in_order(
            Some(guard),
            || Err(test_error("synchronize failed")),
            || {
                audited.set(true);
                Ok(())
            },
        );
        let message = match result {
            Err(BackendError::DispatchFailed { message, .. }) => message,
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

    #[test]
    fn release_in_order_frees_the_gate_when_the_audit_refuses() {
        let gate = Arc::new(ModuleGlobalsGate::default());
        let guard = ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a fresh gate must be acquirable by the first caller.");
        let result = release_in_order(Some(guard), || Ok(()), || Err(test_error("stale counter")));
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

    #[test]
    fn launch_then_release_runs_the_launch_while_the_gate_is_held() {
        let gate = Arc::new(ModuleGlobalsGate::default());
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

    #[test]
    fn launch_then_release_reports_a_failed_launch_and_still_frees_the_gate() {
        let gate = Arc::new(ModuleGlobalsGate::default());
        let lease = gate_only_lease(&gate);
        let result: Result<(), BackendError> =
            lease.launch_then_release(std::ptr::null_mut(), "gate lifetime unit test", |_lease| {
                Err(test_error("launch failed"))
            });
        let message = match result {
            Err(BackendError::DispatchFailed { message, .. }) => message,
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

    #[test]
    fn a_released_gate_is_acquirable_again_and_a_held_one_is_not() {
        let gate = Arc::new(ModuleGlobalsGate::default());
        let first = ModuleGlobalsGate::acquire(&gate)
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
        let second = ModuleGlobalsGate::acquire(&gate)
            .expect("Fix: a released gate must be acquirable by the next sequence.");
        assert!(
            gate_is_busy(&gate),
            "Fix: re-acquiring must set the busy flag again."
        );
        drop(second);
    }
}
