//! Driver lifecycle E2E and PersistentEngine integration stress.
//!
//! Exercises the full driver path (build Program → binding plan → launch plan
//! → pipeline compile → dispatch → readback) through the cpu-ref backend, and
//! a 16×16 producer/consumer PersistentEngine stress run at 100k items.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use vyre_driver::pipeline::compile;
use vyre_driver::validation::LaunchGeometryLimits;
use vyre_driver::persistent::{PersistentEngine, PersistentWorkItem, QueueFull};
use vyre_driver::{BindingPlan, DispatchConfig, LaunchPlan, VyreBackend};
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;

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
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("sum", Expr::add(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0)))),
            Node::let_bind(
                "diff",
                Expr::sub(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
            ),
            Node::let_bind("product", Expr::mul(Expr::var("sum"), Expr::var("diff"))),
            Node::store("out", Expr::u32(0), Expr::var("product")),
        ],
    )
}

fn launch_limits(backend: &dyn VyreBackend) -> LaunchGeometryLimits {
    LaunchGeometryLimits {
        backend: backend.id(),
        max_threads_per_block: backend.max_compute_invocations_per_workgroup(),
        max_block_dim: backend.max_workgroup_size(),
        max_grid_dim: [
            backend.max_compute_workgroups_per_dimension(),
            backend.max_compute_workgroups_per_dimension(),
            backend.max_compute_workgroups_per_dimension(),
        ],
    }
}

fn reference_bytes(program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let values: Vec<Value> = inputs.iter().cloned().map(Value::from).collect();
    vyre_reference::reference_eval(program, &values)
        .expect("Fix: lifecycle oracle program must execute on reference interpreter")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}

#[test]
fn driver_lifecycle_e2e_parse_plan_emit_dispatch_readback() {
    let program = multi_op_program();
    let a_bytes = 13_u32.to_le_bytes().to_vec();
    let b_bytes = 7_u32.to_le_bytes().to_vec();
    let inputs = vec![a_bytes.clone(), b_bytes.clone()];

    let binding_plan = BindingPlan::build(&program).expect("Fix: lifecycle program must bind");
    vyre_driver::validate_program_for_backend(
        &CpuRefBackend,
        &program,
        &DispatchConfig::default(),
    )
    .expect("Fix: lifecycle program must pass backend validation");

    let launch = LaunchPlan::from_bindings(
        &program,
        &binding_plan.bindings,
        &DispatchConfig::default(),
        launch_limits(&CpuRefBackend),
    )
    .expect("Fix: lifecycle launch plan must prepare geometry");

    assert_eq!(launch.element_count, 1);
    assert!(!launch.param_words.is_empty());

    let backend: Arc<dyn VyreBackend> = Arc::new(CpuRefBackend);
    let pipeline = compile(
        Arc::clone(&backend),
        &program,
        &DispatchConfig::default(),
    )
    .expect("Fix: lifecycle pipeline compile must succeed");

    let outputs = pipeline
        .dispatch(&inputs, &DispatchConfig::default())
        .expect("Fix: lifecycle pipeline dispatch must succeed");

    let expected = reference_bytes(&program, &inputs);
    assert_eq!(
        outputs, expected,
        "Fix: driver lifecycle readback must match reference_eval bytes"
    );

    let sum = 13_u32.wrapping_add(7);
    let diff = 13_u32.wrapping_sub(7);
    assert_eq!(outputs[0], (sum.wrapping_mul(diff)).to_le_bytes().to_vec());
}

struct PersistentWaitGroup {
    lock: Mutex<()>,
    work_ready: Condvar,
    space_ready: Condvar,
}

impl PersistentWaitGroup {
    fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            work_ready: Condvar::new(),
            space_ready: Condvar::new(),
        }
    }

    fn blocking_enqueue(
        &self,
        engine: &PersistentEngine,
        item: PersistentWorkItem,
    ) -> u32 {
        loop {
            match engine.enqueue(item) {
                Ok(slot) => {
                    self.work_ready.notify_all();
                    return slot;
                }
                Err(QueueFull) => {
                    let guard = self.lock.lock().unwrap();
                    let _guard = self.space_ready.wait(guard).unwrap();
                }
            }
        }
    }

    fn blocking_claim(&self, engine: &PersistentEngine) -> PersistentWorkItem {
        loop {
            if let Some(item) = engine.claim() {
                self.space_ready.notify_all();
                return item;
            }
            let guard = self.lock.lock().unwrap();
            let _guard = self.work_ready.wait(guard).unwrap();
        }
    }
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
    let consumed = Arc::new(Mutex::new(vec![false; PERSISTENT_TOTAL_ITEMS as usize]));
    let consumed_count = Arc::new(AtomicUsize::new(0));
    let duplicate_claims = Arc::new(AtomicUsize::new(0));

    let items_per_producer = usize::try_from(PERSISTENT_TOTAL_ITEMS)
        .expect("Fix: persistent stress item count must fit usize")
        / PERSISTENT_PRODUCERS;

    let mut producer_handles = Vec::with_capacity(PERSISTENT_PRODUCERS);
    for producer_id in 0..PERSISTENT_PRODUCERS {
        let engine = Arc::clone(&engine);
        let wait = Arc::clone(&wait);
        producer_handles.push(thread::spawn(move || {
            let base = producer_id * items_per_producer;
            for offset in 0..items_per_producer {
                let correlation = u32::try_from(base + offset)
                    .expect("Fix: persistent stress correlation ids must fit u32");
                let _slot = wait.blocking_enqueue(&engine, persistent_work_item(correlation));
            }
        }));
    }

    let mut consumer_handles = Vec::with_capacity(PERSISTENT_CONSUMERS);
    for _ in 0..PERSISTENT_CONSUMERS {
        let engine = Arc::clone(&engine);
        let wait = Arc::clone(&wait);
        let consumed = Arc::clone(&consumed);
        let consumed_count = Arc::clone(&consumed_count);
        let duplicate_claims = Arc::clone(&duplicate_claims);
        consumer_handles.push(thread::spawn(move || loop {
            if consumed_count.load(Ordering::Acquire) >= PERSISTENT_TOTAL_ITEMS as usize {
                break;
            }
            let item = wait.blocking_claim(&engine);
            let correlation = item.correlation as usize;
            let mut seen = consumed.lock().unwrap();
            if correlation >= seen.len() || seen[correlation] {
                duplicate_claims.fetch_add(1, Ordering::Relaxed);
            } else {
                seen[correlation] = true;
                consumed_count.fetch_add(1, Ordering::Release);
            }
            drop(seen);
            if consumed_count.load(Ordering::Acquire) >= PERSISTENT_TOTAL_ITEMS as usize {
                break;
            }
        }));
    }

    for handle in producer_handles {
        handle.join().expect("Fix: persistent stress producer must not panic");
    }
    wait.work_ready.notify_all();

    for handle in consumer_handles {
        handle.join().expect("Fix: persistent stress consumer must not panic");
    }

    assert_eq!(
        duplicate_claims.load(Ordering::Relaxed),
        0,
        "Fix: persistent stress must not double-claim items"
    );
    assert_eq!(
        consumed_count.load(Ordering::Acquire),
        PERSISTENT_TOTAL_ITEMS as usize,
        "Fix: persistent stress must consume every enqueued item exactly once"
    );
    assert_eq!(engine.head_counter(), u64::from(PERSISTENT_TOTAL_ITEMS));
    assert_eq!(engine.tail_counter(), u64::from(PERSISTENT_TOTAL_ITEMS));
    assert_eq!(engine.in_flight(), 0);
}
