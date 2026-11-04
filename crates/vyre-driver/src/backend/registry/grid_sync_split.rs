//! Shared grid-sync wrapper for backends without native cooperative barriers.

use vyre_foundation::ir::Program;

use crate::backend::forward::forward_vyre_backend_support;
use crate::backend::resident_sequence::{dispatch_resident_steps, read_resident_ranges_into};
use crate::backend::{
    BackendError, DeviceBuffer, DispatchConfig, OutputBuffers, PendingDispatch,
    ResidentDispatchStep, ResidentReadRange, Resource, TimedDispatchResult, VyreBackend,
};

pub(super) fn wrap_grid_sync_split(backend: Box<dyn VyreBackend>) -> Box<dyn VyreBackend> {
    Box::new(GridSyncSplitBackend { inner: backend })
}

struct GridSyncSplitBackend {
    inner: Box<dyn VyreBackend>,
}

impl crate::backend::sealed::Sealed for GridSyncSplitBackend {}

/// Only the `Program`-carrying half of the contract is written here. Everything
/// else, identity through lifecycle, comes from the one forwarding owner: this
/// wrapper previously restated it by hand and dropped seven methods onto the
/// trait defaults, which reported the inner backend as having no device-buffer
/// support, no distributed collectives, and no cooperative grid-sync fit.
///
/// Two dispatch entry points are deliberately left on their trait defaults
/// because those defaults route back through `self`, so they take the split
/// decision through the overrides below rather than around them:
/// `dispatch_resident_async` and
/// `dispatch_resident_sequence_read_ranges_timed_into` both call
/// `self.dispatch_resident_timed`. `tests/vyre_backend_forwarding_closure.rs`
/// pins that list, so a new dispatch entry point is red until somebody records
/// which of the two it is.
impl VyreBackend for GridSyncSplitBackend {
    forward_vyre_backend_support!();

    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if crate::grid_sync::contains_grid_sync(program) {
            let borrowed = borrowed_inputs_from_owned(inputs)?;
            if self.should_split_grid_sync_for(program, &borrowed, config)? {
                return crate::grid_sync::dispatch_with_grid_sync_split(
                    self.inner.as_ref(),
                    program,
                    &borrowed,
                    config,
                );
            }
        }
        self.inner.dispatch(program, inputs, config)
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync_for(program, inputs, config)? {
            return crate::grid_sync::dispatch_with_grid_sync_split(
                self.inner.as_ref(),
                program,
                inputs,
                config,
            );
        }
        self.inner.dispatch_borrowed(program, inputs, config)
    }

    fn dispatch_borrowed_timed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync_for(program, inputs, config)? {
            return crate::grid_sync::dispatch_with_grid_sync_split_timed(
                self.inner.as_ref(),
                program,
                inputs,
                config,
            );
        }
        self.inner.dispatch_borrowed_timed(program, inputs, config)
    }

    fn dispatch_borrowed_into(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync_for(program, inputs, config)? {
            return crate::grid_sync::dispatch_with_grid_sync_split_into(
                self.inner.as_ref(),
                program,
                inputs,
                config,
                outputs,
            );
        }
        self.inner
            .dispatch_borrowed_into(program, inputs, config, outputs)
    }

    fn dispatch_resident_timed(
        &self,
        program: &Program,
        resources: &[Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync(program) {
            return crate::grid_sync::dispatch_resident_with_grid_sync_split_timed(
                self.inner.as_ref(),
                program,
                resources,
                config,
            );
        }
        self.inner
            .dispatch_resident_timed(program, resources, config)
    }

    fn dispatch_resident_sequence_read_ranges_into(
        &self,
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        if steps
            .iter()
            .any(|step| self.should_split_grid_sync(step.program))
        {
            dispatch_resident_steps(self, steps)?;
            return read_resident_ranges_into(self, read_ranges, outputs);
        }
        self.inner
            .dispatch_resident_sequence_read_ranges_into(steps, read_ranges, outputs)
    }

    fn dispatch_resident_repeated_sequence_read_ranges_into(
        &self,
        prefix_steps: &[ResidentDispatchStep<'_>],
        repeated_steps: &[ResidentDispatchStep<'_>],
        repeat_count: u32,
        read_ranges: &[ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        if prefix_steps
            .iter()
            .chain(repeated_steps)
            .any(|step| self.should_split_grid_sync(step.program))
        {
            dispatch_resident_steps(self, prefix_steps)?;
            for _ in 0..repeat_count {
                dispatch_resident_steps(self, repeated_steps)?;
            }
            return read_resident_ranges_into(self, read_ranges, outputs);
        }
        self.inner
            .dispatch_resident_repeated_sequence_read_ranges_into(
                prefix_steps,
                repeated_steps,
                repeat_count,
                read_ranges,
                outputs,
            )
    }

    fn dispatch_async(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        config: &DispatchConfig,
    ) -> Result<Box<dyn PendingDispatch>, BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync(program) {
            let borrowed = borrowed_inputs_from_owned(inputs)?;
            let outputs = crate::grid_sync::dispatch_with_grid_sync_split(
                self.inner.as_ref(),
                program,
                &borrowed,
                config,
            )?;
            return Ok(Box::new(super::super::pending_dispatch::ReadyPending {
                outputs,
            }));
        }
        self.inner.dispatch_async(program, inputs, config)
    }

    fn dispatch_borrowed_async(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Box<dyn PendingDispatch>, BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync(program) {
            let outputs = crate::grid_sync::dispatch_with_grid_sync_split(
                self.inner.as_ref(),
                program,
                inputs,
                config,
            )?;
            return Ok(Box::new(super::super::pending_dispatch::ReadyPending {
                outputs,
            }));
        }
        self.inner.dispatch_borrowed_async(program, inputs, config)
    }

    fn dispatch_with_device_buffers(
        &self,
        program: &Program,
        inputs: &[&dyn DeviceBuffer],
        outputs: &mut [&mut dyn DeviceBuffer],
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        self.require_lowered_float_mode(program, config)?;
        if self.should_split_grid_sync(program) {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: program contains a grid-sync barrier and `{}` has no native cooperative launch, so it needs the host-side split, which carries each segment's state through host byte buffers. The device-buffer path exposes no readback between segments. Dispatch this program through dispatch_borrowed, or select a backend that reports supports_grid_sync().",
                    self.inner.id()
                ),
            });
        }
        self.inner
            .dispatch_with_device_buffers(program, inputs, outputs, config)
    }
}

impl GridSyncSplitBackend {
    fn should_split_grid_sync(&self, program: &Program) -> bool {
        crate::grid_sync::contains_grid_sync(program)
            && !self.inner.supports_grid_sync()
            && self.inner.allows_host_grid_sync_split()
    }

    /// Refuse a dispatch whose float lowering mode the wrapped backend does not
    /// lower.
    ///
    /// Every dispatch that carries a `DispatchConfig` passes through here. A
    /// backend that does not implement a mode would otherwise emit the mode it
    /// does implement and return contracted arithmetic under a request for one
    /// rounding per operation, which is a wrong answer rather than a slow one.
    fn require_lowered_float_mode(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        let mode = config.float_lowering;
        if self.inner.honors_float_lowering(mode) {
            return Ok(());
        }
        // `fp_parity::blocked_contraction_feature` is the single definition of
        // the name a contraction-blocking mode refuses under. A mode that
        // permits contraction has no operation set behind it, so a backend
        // refusing one refuses the mode alone.
        let name = vyre_foundation::fp_parity::blocked_contraction_feature(program, mode)
            .unwrap_or_else(|| format!("float lowering mode `{}`", mode.cache_label()));
        Err(BackendError::UnsupportedFeature {
            name,
            backend: self.inner.id().to_string(),
        })
    }

    /// Whether this dispatch must take the host split rather than a native
    /// cooperative launch.
    ///
    /// [`VyreBackend::supports_grid_sync`] answers whether the backend lowers a
    /// whole-grid barrier at all. It does not answer whether THIS launch fits:
    /// a cooperative grid must be fully co-resident, so a block count above the
    /// device's cooperative residency is rejected by the driver rather than run
    /// slowly. Routing on availability alone sent every oversized grid into a
    /// launch that could only fail, which is what
    /// [`crate::backend::ErrorCode::CooperativeResidencyExceeded`] reports.
    ///
    /// [`VyreBackend::allows_host_grid_sync_split`] refuses ONE of the two
    /// reasons a native launch is unavailable. A backend that cannot lower a
    /// grid barrier at all is asking the wrapper to emulate a primitive the
    /// device does not have, and a backend that answers `false` wants that
    /// surfaced as an unsupported feature rather than as a quietly slower
    /// multi-launch path. A backend that CAN lower the barrier and merely
    /// cannot make this grid resident is a different question: there is no
    /// native route to prefer, so refusing the split does not preserve a fast
    /// path, it converts a runnable dispatch into a hard error. `ErrorCode::
    /// CooperativeResidencyExceeded` documents that case as a fallback rather
    /// than a failure, and the fallback is this split.
    fn should_split_grid_sync_for(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        if !crate::grid_sync::contains_grid_sync(program) {
            return Ok(false);
        }
        if !self.inner.supports_grid_sync() {
            return Ok(self.inner.allows_host_grid_sync_split());
        }
        Ok(!self
            .inner
            .cooperative_grid_sync_fits(program, inputs, config)?)
    }
}

fn borrowed_inputs_from_owned(inputs: &[Vec<u8>]) -> Result<Vec<&[u8]>, BackendError> {
    let mut borrowed = Vec::new();
    if borrowed.capacity() < inputs.len() {
        borrowed
            .try_reserve_exact(inputs.len() - borrowed.len())
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: failed to reserve {} borrowed grid-sync input views for registry wrapper dispatch: {error}. Use borrowed dispatch directly or shard the host-side split.",
                    inputs.len()
                ),
            })?;
    }
    borrowed.extend(inputs.iter().map(Vec::as_slice));
    Ok(borrowed)
}
