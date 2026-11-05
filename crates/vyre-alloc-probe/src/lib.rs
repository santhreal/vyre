//! Counts heap traffic on the measuring thread and nowhere else.
//!
//! A contract that budgets allocations per dispatch is measured while the rest
//! of the test binary runs. `cargo test` runs its cases on a thread pool, so a
//! process-wide counter charges one case for every allocation every other case
//! made between the two snapshots, and the budget then reports the schedule
//! rather than the code under test. The counters here are thread-local, so a
//! snapshot pair on one thread spans only that thread's work.
//!
//! Deallocations and reallocations are counted alongside allocations. A region
//! that allocates and frees the same buffer a thousand times has a net of zero
//! and a gross of a thousand, and a budget that cannot see the difference
//! cannot tell a leak from churn.
//!
//! `#![allow(unsafe_code)]`: `GlobalAlloc` is an unsafe trait and every method
//! on it is unsafe. Each one forwards to `System` and adds a thread-local
//! counter update, so the safety obligation is the one `System` already
//! discharges. This is the whole unsafe budget of the module.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Gross heap traffic on one thread since that thread started.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// Successful allocations, counting the allocating half of a realloc.
    pub allocations: usize,
    /// Deallocations, counting the freeing half of a realloc.
    pub deallocations: usize,
    /// Reallocations, also counted once in each of the two fields above.
    pub reallocations: usize,
    /// Bytes requested by the allocations above.
    pub bytes_allocated: usize,
    /// Bytes released by the deallocations above.
    pub bytes_deallocated: usize,
}

thread_local! {
    static THREAD_STATS: Cell<AllocStats> = const { Cell::new(AllocStats {
        allocations: 0,
        deallocations: 0,
        reallocations: 0,
        bytes_allocated: 0,
        bytes_deallocated: 0,
    }) };
}

/// A global allocator that charges each allocation to the thread that made it.
///
/// Install it in the binary that measures a budget:
///
/// ```ignore
/// #[global_allocator]
/// static GLOBAL: vyre_alloc_probe::ThreadAlloc = vyre_alloc_probe::ThreadAlloc;
/// ```
pub struct ThreadAlloc;

/// Record `f` against the calling thread's counters.
///
/// `try_with` rather than `with`: an allocation during thread-local destruction
/// happens after this key is torn down, and a panic in the allocator aborts the
/// process. Such an allocation goes uncounted, which is correct, because no
/// measurement region is open at that point.
fn record(f: impl FnOnce(&mut AllocStats)) {
    let _ = THREAD_STATS.try_with(|cell| {
        let mut stats = cell.get();
        f(&mut stats);
        cell.set(stats);
    });
}

unsafe impl GlobalAlloc for ThreadAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `GlobalAlloc::alloc` requires a non-zero-size layout, which
        // the caller has already promised. That is the whole obligation, and it
        // is forwarded unchanged to `System`.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(|stats| {
                stats.allocations += 1;
                stats.bytes_allocated += layout.size();
            });
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `GlobalAlloc::alloc_zeroed` carries the same non-zero-size
        // obligation as `alloc`, discharged by the caller and forwarded
        // unchanged to `System`.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(|stats| {
                stats.allocations += 1;
                stats.bytes_allocated += layout.size();
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by this allocator under `layout`, which
        // `GlobalAlloc::dealloc` requires of its caller. Every allocation this
        // allocator returns came from `System` under the same layout, so
        // `System` is the correct owner to free it.
        unsafe { System.dealloc(ptr, layout) };
        record(|stats| {
            stats.deallocations += 1;
            stats.bytes_deallocated += layout.size();
        });
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` was allocated by this allocator under `layout` and
        // `new_size` is a valid size for it, both required of the caller by
        // `GlobalAlloc::realloc`. Every allocation this allocator returns came
        // from `System`, so `System` owns the block being resized.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record(|stats| {
                stats.reallocations += 1;
                stats.allocations += 1;
                stats.deallocations += 1;
                stats.bytes_allocated += new_size;
                stats.bytes_deallocated += layout.size();
            });
        }
        new_ptr
    }
}

/// Heap traffic between two snapshots on one thread.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionChange {
    /// Allocations in the region.
    pub allocations: usize,
    /// Deallocations in the region.
    pub deallocations: usize,
    /// Reallocations in the region.
    pub reallocations: usize,
    /// Bytes allocated in the region.
    pub bytes_allocated: usize,
    /// Bytes deallocated in the region.
    pub bytes_deallocated: usize,
}

impl RegionChange {
    /// Bytes the region retains: allocated minus deallocated, negative when the
    /// region frees more than it takes.
    #[must_use]
    pub fn net_bytes(&self) -> isize {
        self.bytes_allocated as isize - self.bytes_deallocated as isize
    }

    /// Live allocations the region retains, negative when it frees more blocks
    /// than it takes.
    #[must_use]
    pub fn net_allocations(&self) -> isize {
        self.allocations as isize - self.deallocations as isize
    }
}

/// An open measurement region: the counters at construction, diffed on demand.
///
/// Construct one, run the code under budget, then read [`Region::change`].
/// Nothing is reset, so regions nest.
pub struct Region {
    start: AllocStats,
}

impl Region {
    /// Snapshot the calling thread's counters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: THREAD_STATS.try_with(Cell::get).unwrap_or_default(),
        }
    }

    /// Traffic since this region was constructed.
    #[must_use]
    pub fn change(&self) -> RegionChange {
        let current = THREAD_STATS.try_with(Cell::get).unwrap_or_default();
        RegionChange {
            allocations: current.allocations.saturating_sub(self.start.allocations),
            deallocations: current
                .deallocations
                .saturating_sub(self.start.deallocations),
            reallocations: current
                .reallocations
                .saturating_sub(self.start.reallocations),
            bytes_allocated: current
                .bytes_allocated
                .saturating_sub(self.start.bytes_allocated),
            bytes_deallocated: current
                .bytes_deallocated
                .saturating_sub(self.start.bytes_deallocated),
        }
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}
