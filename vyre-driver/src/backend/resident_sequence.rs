//! The backend-independent reading of a resident dispatch sequence.
//!
//! A resident sequence is a list of programs launched against resources that
//! stay bound, followed by a list of byte ranges read back from them. Every
//! backend that does not fuse the sequence onto one queue falls back to the
//! same two decisions: what launch configuration each step gets, and how the
//! requested ranges become one readback call. Both the `VyreBackend` defaults
//! and the grid-sync split decorator route through here, so the fallback cannot
//! drift between the trait and a wrapper that overrides it.

use smallvec::SmallVec;
use vyre_foundation::ir::Program;

use crate::backend::{BackendError, DispatchConfig, Resource, VyreBackend};

/// One backend-resident program dispatch in an ordered sequence.
pub struct ResidentDispatchStep<'a> {
    /// Program to dispatch.
    pub program: &'a Program,
    /// Resident resources in binding order.
    pub resources: &'a [Resource],
    /// The launch this step runs, when it states one.
    ///
    /// A grid is sized for a workgroup, so the two travel together or not at
    /// all: a step that stated a grid and lost its workgroup to a shared loop
    /// launched a grid that under-covered the work and dropped findings.
    pub launch: Option<crate::launch_directive::LaunchDirective>,
}

/// One compact byte range to read from a backend-resident resource.
pub struct ResidentReadRange<'a> {
    /// Resident resource to read from.
    pub resource: &'a Resource,
    /// Inclusive start byte offset within the resident resource.
    pub byte_offset: usize,
    /// Byte length to read back.
    pub byte_len: usize,
}

/// Timing captured for an ordered resident dispatch sequence.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResidentSequenceTiming {
    /// Host wall-clock time in nanoseconds across the entire sequence.
    pub wall_ns: u64,
    /// Total device execution time in nanoseconds, if reported.
    pub device_ns: Option<u64>,
    /// Total queue/stream enqueue latency in nanoseconds, if reported.
    pub enqueue_ns: Option<u64>,
    /// Total host wait latency in nanoseconds, if reported.
    pub wait_ns: Option<u64>,
}

/// Measure elapsed wall time in nanoseconds for a resident sequence.
pub(crate) fn elapsed_resident_sequence_wall_ns(
    started: std::time::Instant,
) -> Result<u64, BackendError> {
    u64::try_from(started.elapsed().as_nanos()).map_err(|error| BackendError::InvalidProgram {
        fix: format!(
            "Fix: resident sequence wall timing cannot fit u64 nanoseconds: {error}. Split telemetry windows or report per-step timing."
        ),
    })
}

/// Default implementation for downloading a resident byte range into a new vector.
pub(crate) fn download_resident_range_default<B>(
    backend: &B,
    resource: &Resource,
    byte_offset: usize,
    byte_len: usize,
) -> Result<Vec<u8>, BackendError>
where
    B: VyreBackend + ?Sized,
{
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: resident ranged download could not reserve {byte_len} output byte(s): {error}. Split the readback range before dispatch."
            ),
        }
    })?;
    backend.download_resident_range_into(resource, byte_offset, byte_len, &mut bytes)?;
    Ok(bytes)
}

/// Default implementation for downloading multiple resident byte ranges into output vectors.
pub(crate) fn download_resident_ranges_into_default<B>(
    backend: &B,
    ranges: &[(&Resource, usize, usize)],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    if ranges.len() != outputs.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: resident ranged batch download expected matching range/output counts but got {} range(s) and {} output(s).",
                ranges.len(),
                outputs.len()
            ),
        });
    }
    for ((resource, byte_offset, byte_len), output) in ranges.iter().zip(outputs.iter_mut()) {
        backend.download_resident_range_into(resource, *byte_offset, *byte_len, output)?;
    }
    Ok(())
}

/// Default implementation for borrowed-into dispatch with runtime observability.
pub(crate) fn dispatch_borrowed_into_default<B>(
    backend: &B,
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
    outputs: &mut crate::backend::OutputBuffers,
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    let result = backend.dispatch_borrowed(program, inputs, config)?;
    crate::observability::record_dispatch_io(inputs, &result);
    let stats =
        crate::backend::dispatch_result::replace_output_buffers_preserving_slots_with_memory_stats(
            result, outputs,
        );
    crate::observability::record_output_replacement_stats(stats);
    Ok(())
}

impl vyre_foundation::GeometryStrategy for dyn VyreBackend {
    fn rank_geometries(
        &self,
        requirements: &vyre_foundation::GeometryRequirements,
        problem_elements: u32,
    ) -> Vec<vyre_foundation::LaunchGeometry> {
        self.device_profile()
            .rank_geometries(requirements, problem_elements)
    }
}
/// The launch configuration one resident step is dispatched with.
///
/// A step carries one whole launch or none; nothing else from the caller's
/// configuration applies, because a step's shape is decided by the planner that
/// built the step, not by the dispatch that runs the sequence.
pub(crate) fn resident_step_config(step: &ResidentDispatchStep<'_>) -> DispatchConfig {
    match &step.launch {
        Some(launch) => launch.dispatch_config(),
        None => DispatchConfig::default(),
    }
}

/// Dispatch every step in order against the already-bound resident resources.
///
/// # Errors
///
/// Returns the first step's [`BackendError`], leaving later steps undispatched.
pub(crate) fn dispatch_resident_steps<B>(
    backend: &B,
    steps: &[ResidentDispatchStep<'_>],
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    for step in steps {
        backend.dispatch_resident_timed(
            step.program,
            step.resources,
            &resident_step_config(step),
        )?;
    }
    Ok(())
}

/// Read every requested resident range into `outputs`, in range order.
///
/// # Errors
///
/// Returns [`BackendError`] when a range is invalid or the backend cannot read
/// back resident storage.
pub(crate) fn read_resident_ranges_into<B>(
    backend: &B,
    read_ranges: &[ResidentReadRange<'_>],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    let ranges = read_ranges
        .iter()
        .map(|range| (range.resource, range.byte_offset, range.byte_len))
        .collect::<SmallVec<[_; 8]>>();
    backend.download_resident_ranges_into(&ranges, outputs)
}

/// Default implementation for timed resident sequence dispatch and read ranges.
pub(crate) fn dispatch_resident_sequence_read_ranges_timed_into_default<B>(
    backend: &B,
    steps: &[ResidentDispatchStep<'_>],
    read_ranges: &[ResidentReadRange<'_>],
    outputs: &mut [&mut Vec<u8>],
) -> Result<ResidentSequenceTiming, BackendError>
where
    B: VyreBackend + ?Sized,
{
    let started = std::time::Instant::now();
    let mut device_ns = Some(0_u64);
    let mut enqueue_ns = Some(0_u64);
    let mut wait_ns = Some(0_u64);
    for step in steps {
        let timed = backend.dispatch_resident_timed(
            step.program,
            step.resources,
            &resident_step_config(step),
        )?;
        device_ns = crate::accounting::sum_optional_timing(
            device_ns,
            timed.device_ns,
            "device timing",
            "resident sequence",
            "per-step",
        )?;
        enqueue_ns = crate::accounting::sum_optional_timing(
            enqueue_ns,
            timed.enqueue_ns,
            "enqueue timing",
            "resident sequence",
            "per-step",
        )?;
        wait_ns = crate::accounting::sum_optional_timing(
            wait_ns,
            timed.wait_ns,
            "wait timing",
            "resident sequence",
            "per-step",
        )?;
    }
    read_resident_ranges_into(backend, read_ranges, outputs)?;
    Ok(ResidentSequenceTiming {
        wall_ns: elapsed_resident_sequence_wall_ns(started)?,
        device_ns,
        enqueue_ns,
        wait_ns,
    })
}

/// Default implementation for repeated resident sequence dispatch and read ranges.
pub(crate) fn dispatch_resident_repeated_sequence_read_ranges_into_default<B>(
    backend: &B,
    prefix_steps: &[ResidentDispatchStep<'_>],
    repeated_steps: &[ResidentDispatchStep<'_>],
    repeat_count: u32,
    read_ranges: &[ResidentReadRange<'_>],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    dispatch_resident_steps(backend, prefix_steps)?;
    for _ in 0..repeat_count {
        dispatch_resident_steps(backend, repeated_steps)?;
    }
    read_resident_ranges_into(backend, read_ranges, outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::sealed;
    use crate::TimedDispatchResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SequenceTimingBackend {
        dispatches: AtomicUsize,
    }

    impl sealed::Sealed for SequenceTimingBackend {}

    impl VyreBackend for SequenceTimingBackend {
        fn id(&self) -> &'static str {
            "sequence-timing-test"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            Ok(Vec::new())
        }

        fn dispatch_resident_timed(
            &self,
            program: &Program,
            _resources: &[Resource],
            config: &DispatchConfig,
        ) -> Result<TimedDispatchResult, BackendError> {
            let index = self.dispatches.fetch_add(1, Ordering::SeqCst) as u64;
            let launch = config
                .launch
                .expect("Fix: a resident step must submit the launch it states.");
            assert_eq!(
                launch.grid(),
                [index as u32 + 1, 1, 1],
                "Fix: default resident sequence timing must preserve each step's grid."
            );
            assert_eq!(
                launch.workgroup(),
                program.workgroup_size(),
                "Fix: default resident sequence timing must preserve each step's workgroup."
            );
            Ok(TimedDispatchResult::split_timed(
                Vec::new(),
                10 + index,
                Some(7 + index),
                3 + index,
                4 + index,
            ))
        }

        fn download_resident_ranges_into(
            &self,
            ranges: &[(&Resource, usize, usize)],
            outputs: &mut [&mut Vec<u8>],
        ) -> Result<(), BackendError> {
            assert_eq!(ranges.len(), outputs.len());
            for ((resource, offset, len), output) in ranges.iter().zip(outputs.iter_mut()) {
                let Resource::Resident(handle) = resource else {
                    panic!("Fix: default timed resident sequence test expects resident resources.");
                };
                output.clear();
                output.extend_from_slice(&handle.id().to_le_bytes());
                output.extend_from_slice(&(*offset as u64).to_le_bytes());
                output.extend_from_slice(&(*len as u64).to_le_bytes());
            }
            Ok(())
        }
    }

    #[test]
    fn default_resident_sequence_timing_sums_step_device_times_and_reads_ranges() {
        let backend = SequenceTimingBackend {
            dispatches: AtomicUsize::new(0),
        };
        let program = Program::empty();
        let owner = crate::ResidentOwner::new().expect("Fix: owner ids must be available");
        let first_resources = [Resource::Resident(owner.handle(11))];
        let second_resources = [Resource::Resident(owner.handle(22))];
        let steps = [
            ResidentDispatchStep {
                program: &program,
                resources: &first_resources,
                launch: Some(
                    crate::launch_directive::LaunchDirective::stated_for(&program, [1, 1, 1])
                        .expect("the fixture launch is positive"),
                ),
            },
            ResidentDispatchStep {
                program: &program,
                resources: &second_resources,
                launch: Some(
                    crate::launch_directive::LaunchDirective::stated_for(&program, [2, 1, 1])
                        .expect("the fixture launch is positive"),
                ),
            },
        ];
        let read_resource = Resource::Resident(owner.handle(33));
        let reads = [ResidentReadRange {
            resource: &read_resource,
            byte_offset: 4,
            byte_len: 8,
        }];
        let mut output = Vec::new();

        let timing = backend
            .dispatch_resident_sequence_read_ranges_timed_into(&steps, &reads, &mut [&mut output])
            .expect("Fix: default timed resident sequence must execute and read ranges.");

        assert_eq!(backend.dispatches.load(Ordering::SeqCst), 2);
        assert_eq!(timing.device_ns, Some(15));
        assert_eq!(timing.enqueue_ns, Some(7));
        assert_eq!(timing.wait_ns, Some(9));
        assert!(timing.wall_ns > 0);
        assert_eq!(output.len(), 24);
        assert_eq!(u64::from_le_bytes(output[0..8].try_into().unwrap()), 33);
        assert_eq!(u64::from_le_bytes(output[8..16].try_into().unwrap()), 4);
        assert_eq!(u64::from_le_bytes(output[16..24].try_into().unwrap()), 8);
    }

    /// Recording backend that keeps every submitted launch shape in order.
    struct RecordingBackend {
        submitted: std::sync::Mutex<Vec<([u32; 3], [u32; 3])>>,
    }

    impl sealed::Sealed for RecordingBackend {}

    impl VyreBackend for RecordingBackend {
        fn id(&self) -> &'static str {
            "resident-launch-recording-test"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            Ok(Vec::new())
        }

        fn dispatch_resident_timed(
            &self,
            _program: &Program,
            _resources: &[Resource],
            config: &DispatchConfig,
        ) -> Result<TimedDispatchResult, BackendError> {
            let launch = config
                .launch
                .expect("Fix: a resident step must submit the launch it states.");
            self.submitted
                .lock()
                .expect("Fix: the recording lock must stay usable.")
                .push((launch.workgroup(), launch.grid()));
            Ok(TimedDispatchResult::host_timed(Vec::new(), 1))
        }
    }

    /// WHY: a resident step used to carry its workgroup and its grid in two
    /// independent options, and the sequence loop forwarded only the grid. A grid
    /// sized for a 64-lane workgroup then ran with the program's declared shape,
    /// covering a fraction of the work with no error anywhere. This closes that
    /// class: whatever a step states, both axes of the launch reach the backend.
    #[test]
    fn a_resident_step_submits_the_whole_launch_it_states() {
        let backend = RecordingBackend {
            submitted: std::sync::Mutex::new(Vec::new()),
        };
        // The program declares [1, 1, 1], so a dropped workgroup is observable.
        let program = Program::empty();
        assert_eq!(program.workgroup_size(), [1, 1, 1]);
        let owner = crate::ResidentOwner::new().expect("Fix: owner ids must be available");
        let resources = [Resource::Resident(owner.handle(11))];
        let stated = [
            ([64, 1, 1], [3, 1, 1]),
            ([32, 2, 1], [5, 7, 1]),
            ([8, 1, 4], [1, 1, 9]),
        ];
        let launches = stated
            .iter()
            .map(|(workgroup, grid)| {
                crate::launch_directive::LaunchDirective::stated(*workgroup, *grid, 0)
                    .expect("the stated fixture launches are positive")
            })
            .collect::<Vec<_>>();
        let steps = launches
            .iter()
            .map(|launch| ResidentDispatchStep {
                program: &program,
                resources: &resources,
                launch: Some(*launch),
            })
            .collect::<Vec<_>>();

        dispatch_resident_steps(&backend, &steps).expect("Fix: the fixture sequence must run.");

        let submitted = backend
            .submitted
            .lock()
            .expect("Fix: the recording lock must stay usable.")
            .clone();
        assert_eq!(submitted, stated.to_vec());
    }
}
