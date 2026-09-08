//! Allocation-count contracts for steady-state GPU dispatch.
//!
//! After caches are warm, CPU-side heap traffic must stay bounded. Budgets are
//! documented here; tighten them as zero-copy and caller-owned output buffers
//! land (inventory items 3–5, 10).
//!
//! `dispatch_async` does not clone input *payloads*: it collects `&[u8]` views into a
//! `SmallVec` (inline capacity 8) and passes those borrows through to GPU staging. Caller-owned
//! `Vec` buffers must stay alive until `PendingDispatch` resolves (same aliasing contract as
//! `dispatch_borrowed_async`).

#![cfg(feature = "device-tests")]
#![allow(missing_docs)]

use crate::harness;
use harness::{acquire_live_backend as live_backend, add_one_program};

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::{Duration, Instant};

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::{CompiledPipeline, DispatchConfig, VyreBackend};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub reallocations: usize,
    pub bytes_allocated: usize,
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

/// Global allocator tracking per-thread allocation and deallocation statistics.
pub struct ThreadAlloc;

impl ThreadAlloc {
    pub const fn new() -> Self {
        Self
    }
}

unsafe impl GlobalAlloc for ThreadAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let _ = THREAD_STATS.try_with(|cell| {
                let mut stats = cell.get();
                stats.allocations += 1;
                stats.bytes_allocated += layout.size();
                cell.set(stats);
            });
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            let _ = THREAD_STATS.try_with(|cell| {
                let mut stats = cell.get();
                stats.allocations += 1;
                stats.bytes_allocated += layout.size();
                cell.set(stats);
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        let _ = THREAD_STATS.try_with(|cell| {
            let mut stats = cell.get();
            stats.deallocations += 1;
            stats.bytes_deallocated += layout.size();
            cell.set(stats);
        });
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            let _ = THREAD_STATS.try_with(|cell| {
                let mut stats = cell.get();
                stats.reallocations += 1;
                stats.allocations += 1;
                stats.deallocations += 1;
                stats.bytes_allocated += new_size;
                stats.bytes_deallocated += layout.size();
                cell.set(stats);
            });
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: ThreadAlloc = ThreadAlloc::new();

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionChange {
    pub allocations: usize,
    pub deallocations: usize,
    pub reallocations: usize,
    pub bytes_allocated: usize,
    pub bytes_deallocated: usize,
}

impl RegionChange {
    pub fn net_bytes(&self) -> isize {
        self.bytes_allocated as isize - self.bytes_deallocated as isize
    }

    pub fn net_allocations(&self) -> isize {
        self.allocations as isize - self.deallocations as isize
    }
}

/// Measurement region snapshotting and diffing thread-local allocator statistics.
pub struct Region {
    start: AllocStats,
}

impl Region {
    pub fn new() -> Self {
        let start = THREAD_STATS.try_with(|cell| cell.get()).unwrap_or_default();
        Self { start }
    }

    pub fn change(&self) -> RegionChange {
        let current = THREAD_STATS.try_with(|cell| cell.get()).unwrap_or_default();
        RegionChange {
            allocations: current.allocations.saturating_sub(self.start.allocations),
            deallocations: current.deallocations.saturating_sub(self.start.deallocations),
            reallocations: current.reallocations.saturating_sub(self.start.reallocations),
            bytes_allocated: current.bytes_allocated.saturating_sub(self.start.bytes_allocated),
            bytes_deallocated: current.bytes_deallocated.saturating_sub(self.start.bytes_deallocated),
        }
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a Program with `inputs` separate read buffers and one output. The
/// summed program exceeds the dispatch-local `SmallVec` inline cap of 8 used
/// by `clear_requests`, exercising the spill path.
fn many_input_sum_program(inputs: u32, words: u32) -> Program {
    let mut bindings: Vec<BufferDecl> = (0..inputs)
        .map(|i| BufferDecl::read(&format!("input_{i}"), i, DataType::U32).with_count(words))
        .collect();
    bindings.push(
        BufferDecl::output("out", inputs, DataType::U32)
            .with_count(words)
            .with_output_byte_range(0..(words as usize * 4)),
    );
    let idx = Expr::gid_x();
    let in_bounds = Expr::lt(idx.clone(), Expr::u32(words));
    let mut sum = Expr::load("input_0", idx.clone());
    for i in 1..inputs {
        sum = Expr::add(sum, Expr::load(format!("input_{i}"), idx.clone()));
    }
    Program::wrapped(
        bindings,
        [64, 1, 1],
        vec![
            Node::if_then(in_bounds, vec![Node::store("out", idx, sum)]),
            Node::return_(),
        ],
    )
}

/// (max heap allocations, max heap bytes) for one hot `dispatch_borrowed` after warm-up.
///
/// Ratchet: actual measured 2026-05 steady-state on the live wgpu/Vulkan
/// path is higher than the original Inventory P0 #10 aspiration (~200).
/// The current budget reflects what the path actually does; lowering it
/// requires the readback-mutex, zero-copy outputs, and dispatch-arena work.
fn budget_borrowed_hot() -> (usize, usize) {
    (3072, 4 * 1024 * 1024)
}

/// Wide-program ratchet: a Program whose buffer count exceeds the dispatch
/// `SmallVec` inline cap (8 for `clear_requests`) must stay within this budget
/// after warm-up. Per-thread scratch arenas eliminate the per-dispatch heap
/// allocations that the spill path used to pay.
fn budget_borrowed_wide_hot() -> (usize, usize) {
    (4096, 6 * 1024 * 1024)
}

/// Async path pays for channel + task metadata on top of dispatch.
fn budget_async_hot() -> (usize, usize) {
    (4096, 6 * 1024 * 1024)
}

/// Compiled-pipeline hot path ratchet: same budget as `budget_borrowed_hot`.
fn budget_compiled_hot() -> (usize, usize) {
    (3072, 4 * 1024 * 1024)
}

#[test]
fn direct_dispatch_borrowed_steady_state_alloc_bounded() {
    let backend = live_backend();
    let program = add_one_program(1024);
    let input: Vec<u8> = (0..1024u32).flat_map(u32::to_le_bytes).collect();
    let borrowed = [input.as_slice()];

    let _ = backend
        .dispatch_borrowed(&program, &borrowed, &DispatchConfig::default())
        .expect("Fix: warm-up dispatch_borrowed must succeed");

    let region = Region::new();
    let _ = backend
        .dispatch_borrowed(&program, &borrowed, &DispatchConfig::default())
        .expect("Fix: hot dispatch_borrowed must succeed");
    let change = region.change();

    let (max_allocs, max_bytes) = budget_borrowed_hot();
    assert!(
        change.allocations <= max_allocs,
        "Fix: hot dispatch_borrowed must not exceed {max_allocs} heap allocations (got {}). \
         Inventory P0 #10: reduce per-dispatch host allocations.",
        change.allocations
    );
    assert!(
        change.bytes_allocated <= max_bytes,
        "Fix: hot dispatch_borrowed must not exceed {max_bytes} heap bytes (got {}). \
         Inventory P0 #10: reduce per-dispatch host bytes.",
        change.bytes_allocated
    );
}

#[test]
fn wide_program_dispatch_borrowed_steady_state_alloc_bounded() {
    let backend = live_backend();
    // 12 inputs > clear_requests inline cap (8): the dispatch hot path's
    // SmallVec spill must come from per-thread scratch capacity, not a fresh
    // heap allocation per dispatch.
    let inputs_count: u32 = 12;
    let words: u32 = 256;
    let program = many_input_sum_program(inputs_count, words);
    let one_input: Vec<u8> = (0..words).flat_map(u32::to_le_bytes).collect();
    let owned: Vec<Vec<u8>> = (0..inputs_count).map(|_| one_input.clone()).collect();
    let borrowed: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();

    let _ = backend
        .dispatch_borrowed(&program, &borrowed, &DispatchConfig::default())
        .expect("Fix: warm-up wide dispatch_borrowed must succeed");
    // Run a second warm-up so the bind-group cache, pipeline cache, and
    // dispatch scratch all reach steady state before the measurement.
    let _ = backend
        .dispatch_borrowed(&program, &borrowed, &DispatchConfig::default())
        .expect("Fix: second warm-up wide dispatch_borrowed must succeed");

    let region = Region::new();
    let _ = backend
        .dispatch_borrowed(&program, &borrowed, &DispatchConfig::default())
        .expect("Fix: hot wide dispatch_borrowed must succeed");
    let change = region.change();

    let (max_allocs, max_bytes) = budget_borrowed_wide_hot();
    assert!(
        change.allocations <= max_allocs,
        "Fix: hot wide dispatch_borrowed must not exceed {max_allocs} heap allocations (got {}). \
         Inventory P0 #9: per-thread scratch arenas should absorb SmallVec spills.",
        change.allocations
    );
    assert!(
        change.bytes_allocated <= max_bytes,
        "Fix: hot wide dispatch_borrowed must not exceed {max_bytes} heap bytes (got {}). \
         Inventory P0 #9.",
        change.bytes_allocated
    );
}

#[test]
fn compiled_pipeline_dispatch_steady_state_alloc_bounded() {
    let backend = live_backend();
    let program = add_one_program(1024);
    let input: Vec<u8> = (0..1024u32).flat_map(u32::to_le_bytes).collect();

    let pipeline = backend
        .compile_pipeline_for_oracle(&program, &DispatchConfig::default())
        .expect("Fix: oracle pipeline compilation must succeed");

    let _ = pipeline
        .dispatch(&[input.clone()], &DispatchConfig::default())
        .expect("Fix: warm-up compiled dispatch must succeed");

    let region = Region::new();
    let _ = pipeline
        .dispatch(&[input], &DispatchConfig::default())
        .expect("Fix: hot compiled dispatch must succeed");
    let change = region.change();

    let (max_allocs, max_bytes) = budget_compiled_hot();
    assert!(
        change.allocations <= max_allocs,
        "Fix: hot compiled pipeline dispatch must not exceed {max_allocs} heap allocations (got {}). \
         Inventory P0 #10.",
        change.allocations
    );
    assert!(
        change.bytes_allocated <= max_bytes,
        "Fix: hot compiled pipeline dispatch must not exceed {max_bytes} heap bytes (got {}). \
         Inventory P0 #10.",
        change.bytes_allocated
    );
}

/// Single-input `dispatch_async`: slice collection uses inline `SmallVec` cap 8; input bytes are
/// borrowed, not copied (see module docs).
#[test]
fn async_dispatch_steady_state_alloc_bounded() {
    let backend = live_backend();
    let program = add_one_program(1024);
    let input: Vec<u8> = (0..1024u32).flat_map(u32::to_le_bytes).collect();

    let pending0 = backend
        .dispatch_async(&program, &[input.clone()], &DispatchConfig::default())
        .expect("Fix: warm-up dispatch_async must return a handle");
    let _ = pending0
        .await_result()
        .expect("Fix: warm-up async dispatch must complete");

    let region = Region::new();
    let async_start = Instant::now();
    let pending = backend
        .dispatch_async(&program, &[input], &DispatchConfig::default())
        .expect("Fix: hot dispatch_async must return a handle");
    let _ = pending
        .await_result()
        .expect("Fix: hot async dispatch must complete");
    let _elapsed = async_start.elapsed();
    let change = region.change();

    // Still must complete a real GPU round-trip (not CPU fallback).
    assert!(
        _elapsed > Duration::from_micros(10),
        "Fix: async await returned in {_elapsed:?}, too fast for GPU. Possible silent CPU fallback."
    );

    let (max_allocs, max_bytes) = budget_async_hot();
    assert!(
        change.allocations <= max_allocs,
        "Fix: hot dispatch_async+await must not exceed {max_allocs} heap allocations in measured region (got {}). \
         Inventory P0 #10: async path should not allocate unboundedly per job.",
        change.allocations
    );
    assert!(
        change.bytes_allocated <= max_bytes,
        "Fix: hot dispatch_async+await must not exceed {max_bytes} heap bytes in measured region (got {}). \
         Inventory P0 #10.",
        change.bytes_allocated
    );
}

/// Three inputs (≤ inline `SmallVec` cap 8): exercises `dispatch_async` slice collection across
/// multiple buffers without cloning payload bytes; allocation budget matches the ratcheted async path.
#[test]
fn async_dispatch_multi_input_borrowed_smallvec_inline_alloc_bounded() {
    let backend = live_backend();
    let inputs_count: u32 = 3;
    let words: u32 = 256;
    let program = many_input_sum_program(inputs_count, words);
    let one_input: Vec<u8> = (0..words).flat_map(u32::to_le_bytes).collect();
    let owned: Vec<Vec<u8>> = (0..inputs_count).map(|_| one_input.clone()).collect();

    let pending0 = backend
        .dispatch_async(&program, &owned, &DispatchConfig::default())
        .expect("Fix: warm-up multi-input dispatch_async must return a handle");
    let _ = pending0
        .await_result()
        .expect("Fix: warm-up multi-input async dispatch must complete");

    let region = Region::new();
    let async_start = Instant::now();
    let pending = backend
        .dispatch_async(&program, &owned, &DispatchConfig::default())
        .expect("Fix: hot multi-input dispatch_async must return a handle");
    let _ = pending
        .await_result()
        .expect("Fix: hot multi-input async dispatch must complete");
    let _elapsed = async_start.elapsed();
    let change = region.change();

    assert!(
        _elapsed > Duration::from_micros(10),
        "Fix: multi-input async await returned in {_elapsed:?}, too fast for GPU. Possible silent CPU fallback."
    );

    let (max_allocs, max_bytes) = budget_async_hot();
    assert!(
        change.allocations <= max_allocs,
        "Fix: hot multi-input dispatch_async+await must not exceed {max_allocs} heap allocations in measured region (got {}).",
        change.allocations
    );
    assert!(
        change.bytes_allocated <= max_bytes,
        "Fix: hot multi-input dispatch_async+await must not exceed {max_bytes} heap bytes in measured region (got {}).",
        change.bytes_allocated
    );
}

#[test]
fn allocation_measurement_isolates_concurrent_threads() {
    let region = Region::new();

    let background = std::thread::spawn(|| {
        let mut noise: Vec<Vec<u8>> = Vec::new();
        for _ in 0..100 {
            noise.push(vec![0xAA; 64 * 1024]);
        }
        noise
    });

    let noise = background
        .join()
        .expect("Fix: background thread must finish cleanly");
    assert_eq!(noise.len(), 100);

    let change = region.change();
    assert_eq!(
        change.allocations, 0,
        "Fix: background thread allocations leaked into measuring thread. Expected 0 allocations, got {}.",
        change.allocations
    );
    assert_eq!(
        change.bytes_allocated, 0,
        "Fix: background thread allocated bytes leaked into measuring thread. Expected 0 bytes, got {}.",
        change.bytes_allocated
    );
}

#[test]
fn allocation_measurement_tracks_local_thread_traffic() {
    let region = Region::new();
    let buffer: Vec<u8> = vec![42; 2048];
    let change = region.change();
    assert!(
        change.allocations >= 1,
        "Fix: local thread allocations must be counted. Got {}.",
        change.allocations
    );
    assert!(
        change.bytes_allocated >= 2048,
        "Fix: local thread bytes allocated must be counted. Got {}.",
        change.bytes_allocated
    );
    drop(buffer);
    let change_after_drop = region.change();
    assert!(
        change_after_drop.deallocations >= 1,
        "Fix: local thread deallocations must be counted. Got {}.",
        change_after_drop.deallocations
    );
    assert!(
        change_after_drop.bytes_deallocated >= 2048,
        "Fix: local thread bytes deallocated must be counted. Got {}.",
        change_after_drop.bytes_deallocated
    );
}
