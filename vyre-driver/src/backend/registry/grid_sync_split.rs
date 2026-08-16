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
        if self.should_split_grid_sync(program) {
            let borrowed = borrowed_inputs_from_owned(inputs)?;
            return crate::grid_sync::dispatch_with_grid_sync_split(
                self.inner.as_ref(),
                program,
                &borrowed,
                config,
            );
        }
        self.inner.dispatch(program, inputs, config)
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        if self.should_split_grid_sync(program) {
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
        if self.should_split_grid_sync(program) {
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
        if self.should_split_grid_sync(program) {
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

#[cfg(test)]
mod tests {
    use super::wrap_grid_sync_split;
    use crate::backend::forward::{forward_vyre_backend_dispatch, forward_vyre_backend_support};
    use crate::backend::registry::registered_backends;
    use crate::{
        BackendError, DeviceProfile, DeviceTimingQuality, DispatchConfig, ResidentDispatchStep,
        ResidentReadRange, Resource, VyreBackend,
    };
    use smallvec::SmallVec;
    use std::sync::{Arc, Mutex};
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::{BufferDecl, DataType, Node, Program};

    /// Rejects the owned dispatch entry point for a probe that serves the borrowed one.
    ///
    /// The probe implements [`VyreBackend::dispatch_borrowed`] itself; overriding
    /// the owned default with a rejection is what proves a caller reached the
    /// borrowed path rather than being staged into owned rows.
    macro_rules! reject_owned_dispatch {
        ($why:literal) => {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _config: &DispatchConfig,
            ) -> Result<Vec<Vec<u8>>, BackendError> {
                Err(BackendError::new($why))
            }
        };
    }

    /// Rejects both dispatch entry points for a probe that observes another method.
    ///
    /// [`VyreBackend::dispatch_borrowed`] is required, so a probe that dispatches
    /// no program at all still declares it, and the owned default would otherwise
    /// forward into it.
    macro_rules! reject_dispatch {
        ($why:literal) => {
            fn dispatch_borrowed(
                &self,
                _program: &Program,
                _inputs: &[&[u8]],
                _config: &DispatchConfig,
            ) -> Result<Vec<Vec<u8>>, BackendError> {
                Err(BackendError::new($why))
            }
        };
    }

    #[test]
    fn neutral_driver_alone_sees_no_backends() {
        assert!(
            registered_backends()
                .expect("valid empty backend registry")
                .is_empty(),
            "the neutral driver crate links no concrete backend registrations. \
             Fix: if a concrete backend crate was added as a dependency, move this \
             assertion into that crate's test suite."
        );
    }

    #[derive(Default)]
    struct SegmentRecorder {
        calls: Mutex<Vec<(bool, Vec<Vec<u8>>)>>,
    }

    impl crate::backend::sealed::Sealed for SegmentRecorder {}

    impl VyreBackend for SegmentRecorder {
        fn id(&self) -> &'static str {
            "segment-recorder"
        }

        fn device_profile(&self) -> DeviceProfile {
            let mut profile = DeviceProfile::conservative(self.id());
            profile.timing_quality = DeviceTimingQuality::DeviceTimestamps;
            profile.supports_device_timestamps = true;
            profile
        }

        reject_owned_dispatch!("owned dispatch should not run for split borrowed path. Fix: keep grid-sync split on the borrowed segment dispatcher.");

        fn dispatch_borrowed(
            &self,
            program: &Program,
            inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            let mut calls = self.calls.lock().map_err(BackendError::poisoned_lock)?;
            let has_grid_sync = crate::grid_sync::contains_grid_sync(program);
            let captured = inputs
                .iter()
                .map(|input| input.to_vec())
                .collect::<Vec<_>>();
            calls.push((has_grid_sync, captured));
            Ok(vec![vec![calls.len() as u8]])
        }
    }

    #[test]
    fn grid_sync_wrapper_preserves_the_concrete_device_profile() {
        let backend = wrap_grid_sync_split(Box::new(SegmentRecorder::default()));
        let profile = backend.device_profile();

        assert_eq!(
            profile.timing_quality,
            DeviceTimingQuality::DeviceTimestamps,
            "Fix: backend decorators must preserve the concrete backend timing quality."
        );
        assert!(
            profile.supports_device_timestamps,
            "Fix: backend decorators must preserve concrete device-timestamp capability."
        );
    }

    fn grid_sync_program() -> Program {
        Program::wrapped(
            vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::Return,
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                Node::Return,
            ],
        )
    }

    #[test]
    fn registered_backend_wrapper_splits_grid_sync_without_recursing() {
        let recorder = Arc::new(SegmentRecorder::default());
        let backend = wrap_grid_sync_split(Box::new(ArcBackend {
            inner: Arc::clone(&recorder),
        }));
        let inputs = [vec![0u8]];
        let borrowed: SmallVec<[&[u8]; 8]> = inputs.iter().map(Vec::as_slice).collect();

        let outputs = backend
            .dispatch_borrowed(&grid_sync_program(), &borrowed, &DispatchConfig::default())
            .expect("Fix: grid-sync split wrapper must dispatch every segment");

        assert_eq!(outputs, vec![vec![2]]);
        let calls = recorder
            .calls
            .lock()
            .expect("Fix: segment recorder mutex must not be poisoned");
        assert_eq!(calls.len(), 2);
        assert!(
            calls.iter().all(|(has_grid_sync, _)| !*has_grid_sync),
            "split segment dispatches must not contain GridSync barriers"
        );
        assert_eq!(calls[0].1, vec![vec![0]]);
        assert_eq!(
            calls[1].1,
            vec![vec![1]],
            "second segment must receive the first segment's ReadWrite output"
        );
    }

    struct NativeGridSyncProbe {
        calls: Mutex<usize>,
    }

    impl crate::backend::sealed::Sealed for NativeGridSyncProbe {}

    impl VyreBackend for NativeGridSyncProbe {
        fn id(&self) -> &'static str {
            "native-grid-sync-probe"
        }

        reject_owned_dispatch!(
            "owned dispatch should not run for this test. Fix: keep the borrowed path selected."
        );

        fn dispatch_borrowed(
            &self,
            program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            assert!(
                crate::grid_sync::contains_grid_sync(program),
                "native grid-sync backends must receive the original unsplit Program"
            );
            *self.calls.lock().map_err(BackendError::poisoned_lock)? += 1;
            Ok(vec![vec![9]])
        }

        fn supports_grid_sync(&self) -> bool {
            true
        }
    }

    #[test]
    fn registered_backend_wrapper_preserves_native_grid_sync_dispatch() {
        let probe = Arc::new(NativeGridSyncProbe {
            calls: Mutex::new(0),
        });
        let backend = wrap_grid_sync_split(Box::new(ArcBackend {
            inner: Arc::clone(&probe),
        }));

        let outputs = backend
            .dispatch_borrowed(&grid_sync_program(), &[], &DispatchConfig::default())
            .expect("Fix: native grid-sync backend should receive original dispatch");

        assert_eq!(outputs, vec![vec![9]]);
        assert_eq!(
            *probe
                .calls
                .lock()
                .expect("Fix: native probe mutex must not be poisoned"),
            1
        );
    }

    struct ResidentUploadProbe {
        uploads: Mutex<Vec<(u64, usize, usize)>>,
    }

    impl crate::backend::sealed::Sealed for ResidentUploadProbe {}

    impl VyreBackend for ResidentUploadProbe {
        fn id(&self) -> &'static str {
            "resident-upload-probe"
        }

        reject_dispatch!("resident upload forwarding test must not dispatch programs.");

        fn upload_resident_at_many(
            &self,
            uploads: &[(&Resource, usize, &[u8])],
        ) -> Result<(), BackendError> {
            let mut captured = self.uploads.lock().map_err(BackendError::poisoned_lock)?;
            for &(resource, offset, bytes) in uploads {
                let Resource::Resident(handle) = resource else {
                    return Err(BackendError::new(
                        "resident upload forwarding test expected resident handles.",
                    ));
                };
                captured.push((handle.id(), offset, bytes.len()));
            }
            Ok(())
        }
    }

    #[test]
    fn registered_backend_wrapper_forwards_ranged_resident_uploads() {
        let probe = Arc::new(ResidentUploadProbe {
            uploads: Mutex::new(Vec::new()),
        });
        let backend = wrap_grid_sync_split(Box::new(ArcBackend {
            inner: Arc::clone(&probe),
        }));

        let owner = crate::ResidentOwner::new().expect("Fix: owner ids must be available");
        backend
            .upload_resident_at_many(&[(&Resource::Resident(owner.handle(7)), 12, &[1, 2, 3])])
            .expect("Fix: grid-sync split wrapper must forward resident ranged uploads");

        assert_eq!(
            probe
                .uploads
                .lock()
                .expect("Fix: resident upload probe mutex must not be poisoned")
                .as_slice(),
            &[(7, 12, 3)]
        );
    }

    struct ResidentSequenceProbe {
        calls: Mutex<Vec<(usize, usize, u32, usize)>>,
    }

    impl crate::backend::sealed::Sealed for ResidentSequenceProbe {}

    impl VyreBackend for ResidentSequenceProbe {
        fn id(&self) -> &'static str {
            "resident-sequence-probe"
        }

        reject_dispatch!("resident sequence forwarding test must not dispatch any inputs.");

        fn dispatch_resident_repeated_sequence_read_ranges_into(
            &self,
            prefix_steps: &[ResidentDispatchStep<'_>],
            repeated_steps: &[ResidentDispatchStep<'_>],
            repeat_count: u32,
            read_ranges: &[ResidentReadRange<'_>],
            outputs: &mut [&mut Vec<u8>],
        ) -> Result<(), BackendError> {
            self.calls
                .lock()
                .map_err(BackendError::poisoned_lock)?
                .push((
                    prefix_steps.len(),
                    repeated_steps.len(),
                    repeat_count,
                    read_ranges.len(),
                ));
            for (index, output) in outputs.iter_mut().enumerate() {
                output.clear();
                output.push(index as u8 + 10);
            }
            Ok(())
        }
    }

    #[test]
    fn registered_backend_wrapper_forwards_resident_repeated_sequences() {
        let probe = Arc::new(ResidentSequenceProbe {
            calls: Mutex::new(Vec::new()),
        });
        let backend = wrap_grid_sync_split(Box::new(ArcBackend {
            inner: Arc::clone(&probe),
        }));
        let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
        let owner = crate::ResidentOwner::new().expect("Fix: owner ids must be available");
        let resources = [Resource::Resident(owner.handle(9))];
        let prefix_steps = [ResidentDispatchStep {
            program: &program,
            resources: &resources,
            grid_override: None,
            workgroup_override: None,
        }];
        let repeated_steps = [ResidentDispatchStep {
            program: &program,
            resources: &resources,
            grid_override: Some([3, 1, 1]),
            workgroup_override: None,
        }];
        let read_ranges = [
            ResidentReadRange {
                resource: &resources[0],
                byte_offset: 0,
                byte_len: 1,
            },
            ResidentReadRange {
                resource: &resources[0],
                byte_offset: 4,
                byte_len: 1,
            },
        ];
        let mut first = Vec::new();
        let mut second = Vec::new();

        backend
            .dispatch_resident_repeated_sequence_read_ranges_into(
                &prefix_steps,
                &repeated_steps,
                4,
                &read_ranges,
                &mut [&mut first, &mut second],
            )
            .expect("Fix: grid-sync split wrapper must forward resident repeated sequences");

        assert_eq!(first, vec![10]);
        assert_eq!(second, vec![11]);
        assert_eq!(
            probe
                .calls
                .lock()
                .expect("Fix: resident sequence probe mutex must not be poisoned")
                .as_slice(),
            &[(1, 1, 4, 2)]
        );
    }

    struct ArcBackend<T: VyreBackend + 'static> {
        inner: Arc<T>,
    }

    impl<T: VyreBackend + 'static> crate::backend::sealed::Sealed for ArcBackend<T> {}

    /// Forwards the WHOLE contract, so a probe below observes what a real
    /// backend behind the wrapper would, rather than the trait defaults.
    impl<T: VyreBackend + 'static> VyreBackend for ArcBackend<T> {
        forward_vyre_backend_support!();
        forward_vyre_backend_dispatch!();
    }

    struct GridSyncSplitOptOutProbe {
        calls: Mutex<usize>,
    }

    impl crate::backend::sealed::Sealed for GridSyncSplitOptOutProbe {}

    impl VyreBackend for GridSyncSplitOptOutProbe {
        fn id(&self) -> &'static str {
            "grid-sync-split-opt-out-probe"
        }

        reject_owned_dispatch!(
            "owned dispatch should not run for this test. Fix: keep the borrowed path selected."
        );

        fn dispatch_borrowed(
            &self,
            program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            assert!(
                crate::grid_sync::contains_grid_sync(program),
                "split opt-out backends must receive the original GridSync program"
            );
            *self.calls.lock().map_err(BackendError::poisoned_lock)? += 1;
            Ok(vec![vec![13]])
        }

        fn allows_host_grid_sync_split(&self) -> bool {
            false
        }
    }

    #[test]
    fn registered_backend_wrapper_preserves_grid_sync_when_backend_opts_out_of_host_split() {
        let probe = Arc::new(GridSyncSplitOptOutProbe {
            calls: Mutex::new(0),
        });
        let backend = wrap_grid_sync_split(Box::new(ArcBackend {
            inner: Arc::clone(&probe),
        }));

        let outputs = backend
            .dispatch_borrowed(&grid_sync_program(), &[], &DispatchConfig::default())
            .expect("Fix: split opt-out backend must receive original dispatch");

        assert_eq!(outputs, vec![vec![13]]);
        assert_eq!(
            *probe
                .calls
                .lock()
                .expect("Fix: split opt-out probe mutex must not be poisoned"),
            1
        );
    }

    /// A capability query the wrapper forgot to forward answers for the wrapper,
    /// not for the backend inside it, and reports a real capability as absent.
    /// These four were the ones it forgot.
    struct CapabilityProbe;

    impl crate::backend::sealed::Sealed for CapabilityProbe {}

    impl VyreBackend for CapabilityProbe {
        fn id(&self) -> &'static str {
            "capability-probe"
        }

        reject_dispatch!("capability forwarding test must not dispatch programs.");

        fn cooperative_grid_sync_fits(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<bool, BackendError> {
            Ok(true)
        }

        fn supports_distributed_collectives(&self) -> bool {
            true
        }

        fn allocate_device_buffer(
            &self,
            byte_len: usize,
        ) -> Result<Box<dyn crate::DeviceBuffer>, BackendError> {
            Err(BackendError::new(format!(
                "capability-probe reached allocate_device_buffer with {byte_len} bytes.",
            )))
        }
    }

    #[test]
    fn registered_backend_wrapper_forwards_capability_queries_to_the_inner_backend() {
        let backend = wrap_grid_sync_split(Box::new(CapabilityProbe));
        let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());

        assert!(
            backend
                .cooperative_grid_sync_fits(&program, &[], &DispatchConfig::default())
                .expect("Fix: cooperative fit query must reach the inner backend"),
            "the wrapper answered the cooperative-fit query itself. Fix: forward it."
        );
        assert!(
            backend.supports_distributed_collectives(),
            "the wrapper answered the collectives capability itself. Fix: forward it."
        );
        let error = backend
            .allocate_device_buffer(64)
            .expect_err("Fix: the probe rejects the allocation, so the call must reach it")
            .to_string();
        assert!(
            error.contains("capability-probe reached allocate_device_buffer with 64 bytes"),
            "the wrapper answered the device-buffer allocation itself, hiding a capable \
             backend behind UnsupportedFeature: {error}"
        );
    }

    #[test]
    fn registered_backend_wrapper_refuses_device_buffer_dispatch_that_needs_the_host_split() {
        let backend = wrap_grid_sync_split(Box::new(CapabilityProbe));
        let error = backend
            .dispatch_with_device_buffers(
                &grid_sync_program(),
                &[],
                &mut [],
                &DispatchConfig::default(),
            )
            .expect_err("Fix: an unsplittable grid-sync dispatch must fail closed")
            .to_string();
        assert!(
            error.contains("host-side split"),
            "the refusal must name why the device-buffer path cannot carry the split: {error}"
        );
        assert!(error.contains("Fix:"), "unexpected message: {error}");
    }
}
