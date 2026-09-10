//! Timing contracts for the host grid-sync split.
//!
//! WHY: a program that needs a whole-grid fence on a backend without
//! cooperative launch runs as several kernel launches, and the split is one
//! implementation of one launch. A caller that asks it for device time must get
//! the same kind of answer a native cooperative launch gives: the total across
//! every launch the split performed, or nothing at all. The timed split used to
//! return `TimedDispatchResult::host_timed`, which reports wall time and no
//! device time whatever the backend measured, so any case comparing a split
//! route against a non-split route saw one route with device timing and one
//! without and could not compare them at all.
//!
//! The class closed here is a total that misrepresents what was measured, in
//! either direction: a device sum that drops launches, and a sum assembled out
//! of segments that reported nothing. Both are asserted against the segment
//! count the split actually performed, read from the backend, so the assertions
//! hold whatever fixpoint iteration count the split arrives at.
//!
//! Not covered here: whether a real device timer is wired to a real backend.
//! That is a device fact and belongs to the CUDA and wgpu suites.

use std::sync::atomic::{AtomicUsize, Ordering};

use vyre_driver::grid_sync::dispatch_with_grid_sync_split_timed;
use vyre_driver::{BackendError, DispatchConfig, TimedDispatchResult, VyreBackend};
use vyre_foundation::ir::Program;
use vyre_test_support::grid_sync_programs::cross_segment_store_program;

/// Per-segment nanoseconds the fake device reports, distinct per field so a sum
/// that pairs the wrong field with the wrong total is visible in the failure.
const DEVICE_NS: u64 = 700;
const ENQUEUE_NS: u64 = 30;
const WAIT_NS: u64 = 11;

/// Which segment dispatches report no device time.
#[derive(Clone, Copy)]
enum DeviceTimer {
    /// Every segment reports `DEVICE_NS`.
    Always,
    /// No segment reports device time; the backend has no timer at all.
    Never,
    /// Only the first segment reports nothing, every later one reports
    /// `DEVICE_NS`. A sum that skips the absent segment reads as a whole
    /// launch's device time when it is missing a launch.
    NotOnFirstSegment,
}

/// A backend that reports scripted timing and counts the segments it ran.
struct ScriptedTimingBackend {
    timer: DeviceTimer,
    calls: AtomicUsize,
}

impl ScriptedTimingBackend {
    fn new(timer: DeviceTimer) -> Self {
        Self {
            timer,
            calls: AtomicUsize::new(0),
        }
    }

    fn segments_dispatched(&self) -> u64 {
        self.calls.load(Ordering::SeqCst) as u64
    }
}

impl vyre_driver::sealed::Sealed for ScriptedTimingBackend {}

impl VyreBackend for ScriptedTimingBackend {
    fn id(&self) -> &'static str {
        "grid-sync-scripted-timing"
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![vec![0u8; 16]; program.output_buffer_indices().len()])
    }

    fn dispatch_borrowed_timed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let device_ns = match self.timer {
            DeviceTimer::Always => Some(DEVICE_NS),
            DeviceTimer::Never => None,
            DeviceTimer::NotOnFirstSegment if call == 0 => None,
            DeviceTimer::NotOnFirstSegment => Some(DEVICE_NS),
        };
        Ok(TimedDispatchResult {
            outputs: self.dispatch_borrowed(program, inputs, config)?,
            wall_ns: 1,
            device_ns,
            enqueue_ns: Some(ENQUEUE_NS),
            wait_ns: Some(WAIT_NS),
        })
    }
}

fn run(timer: DeviceTimer) -> (TimedDispatchResult, u64) {
    let backend = ScriptedTimingBackend::new(timer);
    let program = cross_segment_store_program();
    let timed =
        dispatch_with_grid_sync_split_timed(&backend, &program, &[], &DispatchConfig::default())
            .expect("split dispatch of a two-segment grid-sync program");
    let segments = backend.segments_dispatched();
    assert!(
        segments >= 2,
        "a two-segment grid-sync program must dispatch at least one launch per segment, ran {segments}"
    );
    (timed, segments)
}

#[test]
fn split_device_time_totals_every_segment_launch() {
    let (timed, segments) = run(DeviceTimer::Always);
    assert_eq!(
        timed.device_ns,
        Some(DEVICE_NS * segments),
        "the split ran {segments} launches at {DEVICE_NS} ns each and must report their total"
    );
}

#[test]
fn split_enqueue_and_wait_time_total_every_segment_launch() {
    let (timed, segments) = run(DeviceTimer::Always);
    assert_eq!(
        timed.enqueue_ns,
        Some(ENQUEUE_NS * segments),
        "enqueue time totals across the {segments} launches the split performed"
    );
    assert_eq!(
        timed.wait_ns,
        Some(WAIT_NS * segments),
        "wait time totals across the {segments} launches the split performed"
    );
}

#[test]
fn split_reports_no_device_time_when_no_segment_measures_it() {
    let (timed, segments) = run(DeviceTimer::Never);
    assert_eq!(
        timed.device_ns, None,
        "a backend with no device timer must yield no device total, not a zero one"
    );
    assert_eq!(
        timed.enqueue_ns,
        Some(ENQUEUE_NS * segments),
        "an absent device timer must not suppress the host timings the segments did report"
    );
}

#[test]
fn split_reports_no_device_time_when_one_segment_does_not_measure_it() {
    let (timed, segments) = run(DeviceTimer::NotOnFirstSegment);
    assert_eq!(
        timed.device_ns, None,
        "{} of {segments} launches reported device time, and their partial sum is not this dispatch's device time",
        segments - 1
    );
}

#[test]
fn split_reports_host_wall_time_alongside_the_device_total() {
    let (timed, _) = run(DeviceTimer::Always);
    assert!(
        timed.wall_ns > 0,
        "the split is host-driven and always observes its own wall duration"
    );
}
