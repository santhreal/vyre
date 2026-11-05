//! Contracts for the per-thread allocation probe.
//!
//! The probe exists so an allocation budget measures the code under test and
//! not the test schedule. Two properties carry that: a region charges only the
//! thread that opened it, and gross traffic is reported separately from net, so
//! churn is distinguishable from retention. Both are asserted here against the
//! probe installed as this binary's global allocator, which is the only
//! configuration it is ever used in.
//!
//! `#![allow(unsafe_code)]`: two cases drive `std::alloc` directly, because a
//! `Vec` growth is one realloc plus whatever the allocator decides, and the
//! realloc accounting contract needs exactly one call.
#![allow(unsafe_code)]

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use vyre_alloc_probe::{AllocStats, Region, RegionChange, ThreadAlloc};

#[global_allocator]
static GLOBAL: ThreadAlloc = ThreadAlloc;

/// One megabyte, large enough that no incidental harness allocation reaches it.
const LARGE: usize = 1 << 20;

/// Spin until `phase` reaches `at`. Spinning rather than a `Barrier` or a
/// channel keeps the waiting thread free of heap traffic, so a region open across
/// the wait measures exactly what the thread itself did.
fn wait_for(phase: &AtomicU8, at: u8) {
    while phase.load(Ordering::Acquire) < at {
        std::hint::spin_loop();
        std::thread::yield_now();
    }
}

/// WHY: a process-wide counter charges a measuring thread for every allocation
/// another thread makes inside its region, so an allocation budget measures the
/// test schedule. Each thread here holds a region open while the other one
/// allocates a megabyte and does nothing itself, and each region must report
/// exactly zero traffic: a count leaking from the background thread into the
/// measuring thread fails the first assertion, and one leaking back fails the
/// second. The megabyte each thread does allocate is asserted on its own
/// region, so a probe that counted nothing at all fails too. It does not cover
/// allocations made during thread-local destruction, which go uncounted by
/// design.
#[test]
fn a_region_sees_exactly_zero_while_another_thread_allocates() {
    let phase = Arc::new(AtomicU8::new(0));
    let background_phase = Arc::clone(&phase);
    let background = std::thread::spawn(move || {
        wait_for(&background_phase, 1);
        let own = Region::new();
        let buffer = vec![1u8; LARGE];
        let own_change = own.change();
        background_phase.store(2, Ordering::Release);
        let measured = Region::new();
        background_phase.store(3, Ordering::Release);
        wait_for(&background_phase, 4);
        let measured_change = measured.change();
        drop(buffer);
        (own_change, measured_change)
    });

    let measured = Region::new();
    phase.store(1, Ordering::Release);
    wait_for(&phase, 2);
    let measured_change = measured.change();

    wait_for(&phase, 3);
    let own = Region::new();
    let held = vec![2u8; LARGE];
    let own_change = own.change();
    phase.store(4, Ordering::Release);
    let (background_own, background_measured) = background
        .join()
        .expect("Fix: the background thread must not panic");
    drop(held);

    assert_eq!(
        measured_change,
        RegionChange::default(),
        "Fix: a region must not be charged the megabyte another thread allocated while it was open"
    );
    assert_eq!(
        background_measured,
        RegionChange::default(),
        "Fix: a background thread's region must not be charged the measuring thread's allocation"
    );
    assert!(
        own_change.bytes_allocated >= LARGE && background_own.bytes_allocated >= LARGE,
        "Fix: each thread must see its own megabyte, saw {} and {}",
        own_change.bytes_allocated,
        background_own.bytes_allocated
    );
}

#[test]
fn churn_reports_gross_traffic_and_a_net_of_zero() {
    let rounds = 64;
    let region = Region::new();
    for _ in 0..rounds {
        drop(vec![0u8; 4096]);
    }
    let change = region.change();

    assert!(
        change.allocations >= rounds,
        "each round allocates once, saw {} for {rounds} rounds",
        change.allocations
    );
    assert_eq!(
        change.allocations, change.deallocations,
        "every buffer was freed, so gross allocations and deallocations must match"
    );
    assert_eq!(change.net_allocations(), 0, "churn retains no block");
    assert_eq!(change.net_bytes(), 0, "churn retains no byte");
}

#[test]
fn a_retained_buffer_reports_a_positive_net() {
    let region = Region::new();
    let buffer = vec![0u8; LARGE];
    let change = region.change();

    assert_eq!(change.net_allocations(), 1, "one block is still live");
    assert_eq!(
        change.net_bytes(),
        LARGE as isize,
        "the retained block's bytes are the region's net"
    );
    drop(buffer);
}

#[test]
fn a_region_that_frees_more_than_it_takes_reports_a_negative_net() {
    let buffer = vec![0u8; LARGE];
    let region = Region::new();
    drop(buffer);
    let change = region.change();

    assert_eq!(
        change.net_allocations(),
        -1,
        "the region freed a block it never took"
    );
    assert_eq!(
        change.net_bytes(),
        -(LARGE as isize),
        "a freed-but-not-taken block makes net bytes negative"
    );
}

#[test]
fn one_realloc_counts_once_in_each_gross_half() {
    let layout = Layout::from_size_align(1024, 8).expect("Fix: fixture layout must be valid");
    let ptr = unsafe { alloc(layout) };
    assert!(!ptr.is_null(), "Fix: fixture allocation must succeed");

    let region = Region::new();
    let grown = unsafe { realloc(ptr, layout, 4096) };
    let change = region.change();
    assert!(!grown.is_null(), "Fix: fixture reallocation must succeed");

    assert_eq!(change.reallocations, 1, "exactly one realloc was issued");
    assert_eq!(
        change.allocations, 1,
        "a realloc counts once as an allocation"
    );
    assert_eq!(
        change.deallocations, 1,
        "a realloc counts once as a deallocation"
    );
    assert_eq!(change.bytes_allocated, 4096, "the new size is allocated");
    assert_eq!(change.bytes_deallocated, 1024, "the old size is released");
    assert_eq!(change.net_bytes(), 3072, "the region retains the growth");
    assert_eq!(
        change.net_allocations(),
        0,
        "a realloc replaces a block rather than adding one"
    );

    let grown_layout = Layout::from_size_align(4096, 8).expect("Fix: fixture layout must be valid");
    unsafe { dealloc(grown, grown_layout) };
}

#[test]
fn zeroed_allocation_is_counted_like_any_other() {
    let layout = Layout::from_size_align(2048, 8).expect("Fix: fixture layout must be valid");
    let region = Region::new();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    let change = region.change();
    assert!(!ptr.is_null(), "Fix: fixture allocation must succeed");

    assert_eq!(change.allocations, 1, "alloc_zeroed is an allocation");
    assert_eq!(change.bytes_allocated, 2048, "its size is charged in full");
    assert_eq!(change.deallocations, 0, "it frees nothing");

    unsafe { dealloc(ptr, layout) };
}

#[test]
fn regions_nest_and_the_inner_one_reports_a_subset() {
    let outer = Region::new();
    let first = vec![0u8; LARGE];
    let inner = Region::new();
    let second = vec![0u8; LARGE];
    let inner_change = inner.change();
    let outer_change = outer.change();
    drop((first, second));

    assert_eq!(
        inner_change.bytes_allocated, LARGE,
        "the inner region opened after the first buffer, so it sees one"
    );
    assert_eq!(
        outer_change.bytes_allocated,
        2 * LARGE,
        "the outer region spans both, and opening the inner one resets nothing"
    );
}

#[test]
fn a_region_over_no_work_reports_nothing() {
    let region = Region::new();
    let change = region.change();

    assert_eq!(
        change,
        RegionChange::default(),
        "an empty region must not manufacture traffic"
    );
}

#[test]
fn a_default_stats_snapshot_is_empty() {
    assert_eq!(
        AllocStats::default(),
        AllocStats {
            allocations: 0,
            deallocations: 0,
            reallocations: 0,
            bytes_allocated: 0,
            bytes_deallocated: 0,
        },
        "the derived default must stay the all-zero snapshot a fresh thread starts from"
    );
}
