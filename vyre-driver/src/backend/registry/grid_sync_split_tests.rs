//! Tests for the grid-sync split registry wrapper.
//!
//! Crate-internal: `wrap_grid_sync_split` is `pub(super)`, so no integration
//! test can name it.

use super::grid_sync_split::wrap_grid_sync_split;
use crate::backend::forward::{forward_vyre_backend_dispatch, forward_vyre_backend_support};
use crate::backend::registry::registered_backends;
use crate::{
    BackendError, DeviceProfile, DeviceTimingQuality, DispatchConfig, LaunchDirective,
    ResidentDispatchStep, ResidentReadRange, Resource, VyreBackend,
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

/// A backend that lowers every float mode, so the same sweep proves the
/// wrapper forwards a declared mode instead of refusing everything.
struct StrictCapableProbe;

impl crate::backend::sealed::Sealed for StrictCapableProbe {}

impl VyreBackend for StrictCapableProbe {
    fn id(&self) -> &'static str {
        "strict-capable-probe"
    }

    fn honors_float_lowering(&self, _mode: vyre_foundation::fp_parity::FloatLoweringMode) -> bool {
        true
    }

    fn dispatch_borrowed(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![vec![7]])
    }
}

fn plain_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Return],
    )
}

/// WHY: `DispatchConfig::float_lowering` selects the rounding a caller
/// requires, and a backend that does not lower a mode would otherwise emit
/// the mode it does lower: contracted arithmetic answering a request for one
/// rounding per operation, which is a wrong result rather than a slow one.
/// The sweep runs the whole mode roster, so a mode added later is judged
/// here without this case being edited.
#[test]
fn a_mode_the_wrapped_backend_does_not_lower_is_refused_by_name() {
    let inputs = [vec![0u8]];
    let borrowed: SmallVec<[&[u8]; 8]> = inputs.iter().map(Vec::as_slice).collect();
    for &mode in vyre_foundation::fp_parity::FloatLoweringMode::EVERY {
        let mut config = DispatchConfig::default();
        config.float_lowering = mode;

        let declining = wrap_grid_sync_split(Box::new(SegmentRecorder::default()));
        let verdict = declining.dispatch_borrowed(&plain_program(), &borrowed, &config);
        if mode.blocks_contraction() {
            match verdict {
                Err(BackendError::UnsupportedFeature { name, backend }) => {
                    assert!(
                        name.contains(mode.cache_label()),
                        "Fix: name the refused mode; got `{name}`"
                    );
                    assert_eq!(backend, "segment-recorder");
                }
                other => panic!(
                    "Fix: a backend that does not lower `{}` must refuse the dispatch, got {other:?}",
                    mode.cache_label()
                ),
            }
        } else {
            verdict.expect("Fix: the mode every backend lowers must dispatch.");
        }

        let capable = wrap_grid_sync_split(Box::new(StrictCapableProbe));
        assert_eq!(
            capable
                .dispatch_borrowed(&plain_program(), &borrowed, &config)
                .expect("Fix: a backend that declares the mode must be dispatched, not refused."),
            vec![vec![7]],
            "Fix: the wrapper must forward a mode the wrapped backend lowers."
        );
    }
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

/// WHY: `allows_host_grid_sync_split() == false` means "do not emulate a
/// barrier my device does not have". It was also read as "never split",
/// which sent a launch the device could not make resident into a native
/// cooperative launch that can only return
/// `CooperativeResidencyExceeded`. That is the whole failure: the routing
/// asked whether the split was *allowed* before asking whether the native
/// route *existed*, so the one dispatch with no native route was the one
/// dispatch denied the fallback. Both directions are pinned here, because a
/// rule that always splits would silently discard the native route this
/// backend asked to keep.
#[test]
fn a_native_backend_that_cannot_make_the_grid_resident_still_takes_the_split() {
    for (fits, expected_segments, expect_grid_sync) in [(false, 2, false), (true, 1, true)] {
        let recorder = Arc::new(SegmentRecorder::default());
        let backend = wrap_grid_sync_split(Box::new(GridSyncProbe::native_but_oversized(
            Arc::clone(&recorder),
            fits,
        )));
        let inputs = [vec![0u8]];
        let borrowed: SmallVec<[&[u8]; 8]> = inputs.iter().map(Vec::as_slice).collect();

        backend
            .dispatch_borrowed(&grid_sync_program(), &borrowed, &DispatchConfig::default())
            .expect("Fix: the wrapper must route this dispatch somewhere that runs");

        let calls = recorder
            .calls
            .lock()
            .expect("Fix: segment recorder mutex must not be poisoned");
        assert_eq!(
            calls.len(),
            expected_segments,
            "Fix: with cooperative_grid_sync_fits={fits} the wrapper must dispatch \
             {expected_segments} time(s), not {}.",
            calls.len()
        );
        assert!(
            calls
                .iter()
                .all(|(has_grid_sync, _)| *has_grid_sync == expect_grid_sync),
            "Fix: with cooperative_grid_sync_fits={fits} the backend must receive a program \
             whose GridSync presence is {expect_grid_sync}."
        );
    }
}

/// A backend described entirely by its capability answers. Every routing
/// decision the wrapper makes reads those four answers, so one probe with
/// four fields covers the native barrier, the split opt-out, and the
/// native backend whose grid does not fit. A probe carrying a recorder
/// forwards its dispatches instead of counting them, which is what the
/// segment-level assertions need.
struct GridSyncProbe {
    id: &'static str,
    marker: u8,
    native: bool,
    allows_split: bool,
    fits: bool,
    calls: Mutex<usize>,
    recorder: Option<Arc<SegmentRecorder>>,
}

impl GridSyncProbe {
    fn native() -> Self {
        Self::marked("native-grid-sync-probe", 9, true, true)
    }

    fn split_opt_out() -> Self {
        Self::marked("grid-sync-split-opt-out-probe", 13, false, false)
    }

    /// `fits` follows `native` so the residency answer stays the trait
    /// default for a probe that does not set one.
    fn marked(id: &'static str, marker: u8, native: bool, allows_split: bool) -> Self {
        Self {
            id,
            marker,
            native,
            allows_split,
            fits: native,
            calls: Mutex::new(0),
            recorder: None,
        }
    }

    /// Lowers a grid barrier natively, opts out of the host split, and
    /// answers the residency preflight from `fits`. The two refusals
    /// `allows_host_grid_sync_split` used to be asked for are separable
    /// only on a backend that has both.
    fn native_but_oversized(recorder: Arc<SegmentRecorder>, fits: bool) -> Self {
        Self {
            id: "native-but-oversized-probe",
            marker: 0,
            native: true,
            allows_split: false,
            fits,
            calls: Mutex::new(0),
            recorder: Some(recorder),
        }
    }
}

impl crate::backend::sealed::Sealed for GridSyncProbe {}

impl VyreBackend for GridSyncProbe {
    fn id(&self) -> &'static str {
        self.id
    }

    reject_owned_dispatch!(
        "owned dispatch should not run for this test. Fix: keep the borrowed path selected."
    );

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        if let Some(recorder) = &self.recorder {
            return recorder.dispatch_borrowed(program, inputs, config);
        }
        assert!(
            crate::grid_sync::contains_grid_sync(program),
            "a backend that keeps the barrier must receive the original unsplit Program"
        );
        *self.calls.lock().map_err(BackendError::poisoned_lock)? += 1;
        Ok(vec![vec![self.marker]])
    }

    fn supports_grid_sync(&self) -> bool {
        self.native
    }

    fn allows_host_grid_sync_split(&self) -> bool {
        self.allows_split
    }

    fn cooperative_grid_sync_fits(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        Ok(self.fits)
    }
}

#[test]
fn registered_backend_wrapper_preserves_native_grid_sync_dispatch() {
    let probe = Arc::new(GridSyncProbe::native());
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

/// A backend that overrides `supports_grid_sync` and leaves the residency
/// check on its default must still take the native route. The two defaults
/// disagreed: `cooperative_grid_sync_fits` answered a bare `false`, the
/// wrapper read that as "the native launch does not fit", and every
/// dispatch went through the host split the backend had just said it did
/// not need. The test above only sees it because the split then rejects the
/// input shape; this one asserts the contradiction directly, so overriding
/// one of the pair without the other stays red on its own.
#[test]
fn a_native_grid_sync_claim_answers_the_cooperative_fit_query() {
    /// Overrides `supports_grid_sync` and NOTHING else, so
    /// `cooperative_grid_sync_fits` below is the trait's own default rather
    /// than a copy of it. `GridSyncProbe` cannot stand in here: it answers
    /// the query from a field, which would make this assertion compare two
    /// values the constructor already set equal.
    struct DefaultCooperativeFitProbe;

    impl crate::backend::sealed::Sealed for DefaultCooperativeFitProbe {}

    impl VyreBackend for DefaultCooperativeFitProbe {
        fn id(&self) -> &'static str {
            "default-cooperative-fit-probe"
        }

        reject_dispatch!("the cooperative-fit default test must not dispatch programs.");

        fn supports_grid_sync(&self) -> bool {
            true
        }
    }

    let probe = DefaultCooperativeFitProbe;

    assert_eq!(
        probe
            .cooperative_grid_sync_fits(&grid_sync_program(), &[], &DispatchConfig::default())
            .expect("Fix: the default cooperative-fit query must not error"),
        probe.supports_grid_sync(),
        "the default cooperative-fit answer contradicts supports_grid_sync, so the wrapper \
         emulates a barrier the backend lowers natively"
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
        launch: None,
    }];
    let repeated_steps = [ResidentDispatchStep {
        program: &program,
        resources: &resources,
        launch: Some(
            LaunchDirective::stated_for(&program, [3, 1, 1])
                .expect("the fixture launch is positive"),
        ),
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

#[test]
fn registered_backend_wrapper_preserves_grid_sync_when_backend_opts_out_of_host_split() {
    let probe = Arc::new(GridSyncProbe::split_opt_out());
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
