//! Driver planning contracts and PersistentEngine integration stress.
//!
//! Binding, launch geometry, and ring-buffer behavior are host-side planning
//! decisions and are proved here against declared device limits. Execution is
//! a device concern: a program's result is judged by conformance against the
//! oracle, never by running it on the host from this crate.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use vyre_driver::persistent::{PersistentEngine, PersistentWorkItem};
use vyre_driver::validation::LaunchGeometryLimits;
use vyre_driver::{BindingPlan, DispatchConfig, LaunchPlan};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

const PERSISTENT_PRODUCERS: usize = 16;
const PERSISTENT_CONSUMERS: usize = 16;
const PERSISTENT_TOTAL_ITEMS: u32 = 100_000;
const PERSISTENT_RING_SIZE: u32 = 1024;

/// Minimal multi-op Program: `out = (a + b) * (a - b)` on u32 inputs.
fn multi_op_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(1),
            BufferDecl::read("b", 1, DataType::U32).with_count(1),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind(
                "sum",
                Expr::add(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
            ),
            Node::let_bind(
                "diff",
                Expr::sub(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
            ),
            Node::let_bind("product", Expr::mul(Expr::var("sum"), Expr::var("diff"))),
            Node::store("out", Expr::u32(0), Expr::var("product")),
        ],
    )
}

/// Limits of a device that accepts a 256-thread workgroup and a 65535-wide grid.
///
/// Written out rather than read from a backend: the planner's contract is with
/// the numbers, and a fixed vector keeps the expected geometry exact.
fn launch_limits() -> LaunchGeometryLimits {
    LaunchGeometryLimits {
        backend: "lifecycle-planning-limits",
        max_threads_per_block: 256,
        max_block_dim: [256, 256, 64],
        max_grid_dim: [65_535, 65_535, 65_535],
        // The backend trait exposes no per-compute-unit thread budget, so this
        // lifecycle harness reports none and residency-aware launch decisions
        // stay inert.
        max_threads_per_sm: 0,
    }
}

#[test]
fn driver_low_level_plan_geometry_and_params() {
    let program = multi_op_program();

    let binding_plan = BindingPlan::build(&program).expect("Fix: lifecycle program must bind");
    assert_eq!(
        binding_plan.bindings.len(),
        3,
        "Fix: every declared buffer must receive one binding"
    );

    let launch = LaunchPlan::from_bindings(
        &program,
        &binding_plan.bindings,
        &DispatchConfig::default(),
        launch_limits(),
    )
    .expect("Fix: lifecycle launch plan must prepare geometry");

    assert_eq!(launch.element_count, 1);
    assert!(!launch.param_words.is_empty());
    assert!(
        launch.grid[0] <= 65_535,
        "Fix: planned grid must stay inside the declared limit"
    );
}

#[test]
fn a_grid_wider_than_the_device_admits_is_refused_before_launch() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(1 << 24),
            BufferDecl::output("out", 1, DataType::U32).with_count(1 << 24),
        ],
        [1 << 24, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("a", Expr::u32(0)),
        )],
    );
    let binding_plan = BindingPlan::build(&program).expect("Fix: wide program must bind");
    let mut limits = launch_limits();
    limits.max_grid_dim = [4, 1, 1];

    let error = LaunchPlan::from_bindings(
        &program,
        &binding_plan.bindings,
        &DispatchConfig::default(),
        limits,
    )
    .expect_err("Fix: a grid past the declared limit must be refused, not clamped");
    let text = error.to_string();
    assert!(
        text.contains("Fix:"),
        "Fix: launch refusal must state the corrective action, got `{text}`"
    );
}

struct PersistentWaitGroup {
    producers_done: AtomicBool,
}

impl PersistentWaitGroup {
    fn new() -> Self {
        Self {
            producers_done: AtomicBool::new(false),
        }
    }

    fn mark_producers_done(&self) {
        self.producers_done.store(true, Ordering::Release);
    }
}

fn producer_correlation_range(producer_id: usize) -> (u32, u32) {
    let total = usize::try_from(PERSISTENT_TOTAL_ITEMS)
        .expect("Fix: persistent stress item count must fit usize");
    let base = total / PERSISTENT_PRODUCERS;
    let extra = total % PERSISTENT_PRODUCERS;
    let start = producer_id * base + producer_id.min(extra);
    let count = base + usize::from(producer_id < extra);
    let end = start + count;
    (
        u32::try_from(start).expect("Fix: persistent stress start index must fit u32"),
        u32::try_from(end).expect("Fix: persistent stress end index must fit u32"),
    )
}

fn persistent_work_item(correlation: u32) -> PersistentWorkItem {
    PersistentWorkItem {
        input_offset: correlation.wrapping_mul(64),
        input_len: 64,
        rule_set_id: correlation % 8,
        correlation,
    }
}

#[test]
fn persistent_engine_stress_16_prod_16_cons_100k_items() {
    let engine = Arc::new(PersistentEngine::new(PERSISTENT_RING_SIZE));
    assert_eq!(engine.ring_size(), PERSISTENT_RING_SIZE);

    let wait = Arc::new(PersistentWaitGroup::new());
    let shared_consumed = Arc::new(Mutex::new(Vec::new()));
    let enqueued = Arc::new(AtomicU32::new(0));
    let total_items = usize::try_from(PERSISTENT_TOTAL_ITEMS)
        .expect("Fix: persistent stress item count must fit usize");

    let mut producer_handles = Vec::with_capacity(PERSISTENT_PRODUCERS);
    for producer_id in 0..PERSISTENT_PRODUCERS {
        let engine = Arc::clone(&engine);
        let enqueued = Arc::clone(&enqueued);
        producer_handles.push(thread::spawn(move || {
            let (start, end) = producer_correlation_range(producer_id);
            for correlation in start..end {
                let item = persistent_work_item(correlation);
                loop {
                    if engine.enqueue(item).is_ok() {
                        enqueued.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    thread::yield_now();
                }
            }
        }));
    }

    let mut consumer_handles = Vec::with_capacity(PERSISTENT_CONSUMERS);
    for _ in 0..PERSISTENT_CONSUMERS {
        let engine = Arc::clone(&engine);
        let wait = Arc::clone(&wait);
        let shared_consumed = Arc::clone(&shared_consumed);
        consumer_handles.push(thread::spawn(move || {
            let mut local = Vec::with_capacity(total_items / PERSISTENT_CONSUMERS);
            loop {
                if let Some(item) = engine.claim() {
                    local.push(item.correlation);
                    continue;
                }
                if wait.producers_done.load(Ordering::Acquire) && engine.in_flight() == 0 {
                    break;
                }
                thread::yield_now();
            }
            shared_consumed.lock().unwrap().extend(local);
        }));
    }

    for handle in producer_handles {
        handle
            .join()
            .expect("Fix: persistent stress producer must not panic");
    }
    assert_eq!(
        enqueued.load(Ordering::Relaxed),
        PERSISTENT_TOTAL_ITEMS,
        "Fix: persistent stress producers must enqueue the full workload"
    );
    wait.mark_producers_done();

    for handle in consumer_handles {
        handle
            .join()
            .expect("Fix: persistent stress consumer must not panic");
    }

    let mut consumed = Arc::try_unwrap(shared_consumed)
        .expect("Fix: persistent stress consumers must join before merge")
        .into_inner()
        .expect("Fix: persistent stress consumed mutex must not be poisoned");
    while let Some(item) = engine.claim() {
        consumed.push(item.correlation);
    }
    let raw_count = consumed.len();
    consumed.sort_unstable();
    consumed.dedup();
    assert_eq!(
        raw_count, total_items,
        "Fix: persistent stress claim count must match enqueued workload before dedup"
    );
    assert_eq!(
        consumed.len(),
        total_items,
        "Fix: persistent stress must consume every enqueued item exactly once"
    );
    assert!(
        consumed
            .iter()
            .enumerate()
            .all(|(index, correlation)| *correlation == index as u32),
        "Fix: persistent stress must observe correlation ids 0..={} without gaps or duplicates",
        PERSISTENT_TOTAL_ITEMS - 1
    );
    assert_eq!(engine.head_counter(), u64::from(PERSISTENT_TOTAL_ITEMS));
    assert_eq!(engine.tail_counter(), u64::from(PERSISTENT_TOTAL_ITEMS));
    assert_eq!(engine.in_flight(), 0);
}
