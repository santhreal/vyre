//! The single-workgroup prefix scan: launch geometry, executed shared-memory
//! traffic, and numeric agreement with the host oracle.
//!
//! The executed-store measurement is the reason this file exists. A scan that
//! stages `n` values into workgroup scratch and sweeps them can be correct and
//! still be the wrong algorithm: Hillis-Steele writes every lane on every round,
//! so its shared-memory traffic grows as `N log2 N`, while a work-efficient
//! sweep writes `O(N)` slots in total. Node counts cannot tell the two apart
//! because both unroll to `log2 N` rounds; only the number of LANES that pass
//! each round's guard can. `executed_stores` evaluates every enclosing guard for
//! every lane and counts the store executions the dispatch actually performs.

#![cfg(feature = "math-kernels")]

use std::collections::HashMap;

use vyre_foundation::ir::{BinOp, BufferAccess, Expr, Node, Program};
use vyre_foundation::visit::child_bodies;
use vyre_libs::math::prefix_scan::{prefix_scan, ScanKind};
fn cpu_ref(input: &[u32], kind: ScanKind) -> Vec<u32> {
    match kind {
        ScanKind::InclusiveSum => {
            vyre_reference::composition_witness::inclusive_prefix_sum_witness(input)
        }
        ScanKind::ExclusiveSum => {
            vyre_reference::composition_witness::exclusive_prefix_sum_witness(input)
        }
    }
}
use vyre_reference::value::Value;

/// Every `n` the contract has a distinct shape for: the degenerate sizes, both
/// sides of the workgroup cap, both sides of the single-block limit, and two
/// non-power-of-two lengths.
const SIZES: [u32; 11] = [1, 2, 3, 5, 255, 256, 257, 511, 1023, 1024, 1000];

/// Largest workgroup a single-block scan may request. A scan that pads the
/// workgroup to one lane per element reaches 1024 here and fails.
const MAX_SCAN_WORKGROUP: u32 = 256;

#[test]
fn scan_matches_the_host_oracle_at_every_contract_size() {
    for n in SIZES {
        for kind in [ScanKind::InclusiveSum, ScanKind::ExclusiveSum] {
            let input: Vec<u32> = (0..n)
                .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(n))
                .collect();
            let actual = run_scan(n, kind, &input);
            let expected = cpu_ref(&input, kind);
            assert_eq!(actual, expected, "n={n} kind={kind:?}");
        }
    }
}

#[test]
fn scan_wraps_modulo_two_to_the_thirty_second() {
    let n = 300_u32;
    let mut input = vec![1_u32; n as usize];
    input[7] = u32::MAX;
    input[280] = u32::MAX;
    for kind in [ScanKind::InclusiveSum, ScanKind::ExclusiveSum] {
        assert_eq!(
            run_scan(n, kind, &input),
            cpu_ref(&input, kind),
            "kind={kind:?}"
        );
    }
}

#[test]
fn scan_never_inflates_the_workgroup_past_the_cap() {
    for n in SIZES {
        let program = prefix_scan("in", "out", n, ScanKind::InclusiveSum);
        let [x, y, z] = program.workgroup_size();
        assert!(
            x <= MAX_SCAN_WORKGROUP,
            "n={n}: workgroup {x} exceeds the {MAX_SCAN_WORKGROUP}-lane cap"
        );
        assert_eq!([y, z], [1, 1], "n={n}: scan is one-dimensional");
        for buffer in program.buffers() {
            if buffer.access() == BufferAccess::Workgroup {
                assert!(
                    buffer.count() <= MAX_SCAN_WORKGROUP,
                    "n={n}: scratch {} holds {} slots, past the {MAX_SCAN_WORKGROUP}-lane cap",
                    buffer.name(),
                    buffer.count()
                );
            }
        }
    }
}

#[test]
fn scan_shared_memory_traffic_is_a_constant_per_lane() {
    // A work-efficient sweep touches each scratch slot a bounded number of
    // times. Hillis-Steele writes every lane on each of `log2(lanes)` rounds
    // and then copies the round back, so its traffic carries a `log2(lanes)`
    // factor this budget refuses. Measured on the Hillis-Steele form this
    // replaced: 6,400 stores at n=255 and 31,745 at n=1024, against the 1,544
    // and 1,544 the sweep below executes.
    const STORES_PER_LANE: u32 = 6;
    const STAGING_SLACK: u32 = 8;
    for n in SIZES {
        let program = prefix_scan("in", "out", n, ScanKind::InclusiveSum);
        let lanes = program.workgroup_size()[0];
        let budget = STORES_PER_LANE * lanes + STAGING_SLACK;
        let executed = executed_stores(&program, scratch_names(&program));
        assert!(
            executed <= budget,
            "n={n}: the scan executes {executed} scratch stores over {lanes} lanes, \
             past the budget of {budget}; the sweep is not work-efficient"
        );
    }
}

/// Run the emitted Program on the reference interpreter and return `out`.
fn run_scan(n: u32, kind: ScanKind, input: &[u32]) -> Vec<u32> {
    let program = prefix_scan("in", "out", n, kind);
    let mut values = Vec::new();
    let mut out_slot = None;
    let mut writable = 0_usize;
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        let bytes = if buffer.name() == "in" {
            input.iter().flat_map(|word| word.to_le_bytes()).collect()
        } else {
            vec![0_u8; (buffer.count() as usize) * 4]
        };
        values.push(Value::from(bytes));
        if buffer.access() == BufferAccess::ReadWrite {
            if buffer.name() == "out" {
                out_slot = Some(writable);
            }
            writable += 1;
        }
    }
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("Fix: the scan Program must execute on the reference interpreter");
    let slot = out_slot.expect("Fix: the scan Program must declare a writable out buffer");
    outputs[slot]
        .to_bytes()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// The workgroup-scratch buffers the Program declares.
fn scratch_names(program: &Program) -> Vec<String> {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() == BufferAccess::Workgroup)
        .map(|buffer| buffer.name().to_string())
        .collect()
}

/// Total store executions into `targets` across every lane of the workgroup.
///
/// Walks the Program once per lane, evaluating each enclosing `If` condition
/// with that lane's invocation id. Every guard the scans emit is arithmetic over
/// the lane index and literals, so this is exact rather than an estimate; a
/// guard the evaluator cannot fold is treated as taken, which can only
/// overcount.
fn executed_stores(program: &Program, targets: Vec<String>) -> u32 {
    let lanes = program.workgroup_size()[0];
    let mut total = 0_u32;
    for lane in 0..lanes {
        let mut env = HashMap::new();
        env.insert("__lane__".to_string(), lane);
        for node in program.entry() {
            total += count_node(node, lane, &targets, &mut env);
        }
    }
    total
}

fn count_node(node: &Node, lane: u32, targets: &[String], env: &mut HashMap<String, u32>) -> u32 {
    match node {
        Node::Store { buffer, .. } => u32::from(targets.iter().any(|name| name == buffer.as_str())),
        Node::Let { name, value } => {
            if let Some(folded) = fold(value, lane, env) {
                env.insert(name.as_str().to_string(), folded);
            }
            0
        }
        Node::Assign { name, value } => {
            let folded = fold(value, lane, env);
            match folded {
                Some(word) => {
                    env.insert(name.as_str().to_string(), word);
                }
                None => {
                    env.remove(name.as_str());
                }
            }
            0
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let taken = fold(cond, lane, env).is_none_or(|word| word != 0);
            let arm = if taken { then } else { otherwise };
            arm.iter()
                .map(|inner| count_node(inner, lane, targets, env))
                .sum()
        }
        other => child_bodies(other)
            .into_iter()
            .flatten()
            .map(|inner| count_node(inner, lane, targets, env))
            .sum(),
    }
}

/// Fold an expression over lane-index arithmetic, or `None` when it reads
/// memory or an unbound name.
fn fold(expr: &Expr, lane: u32, env: &HashMap<String, u32>) -> Option<u32> {
    match expr {
        Expr::LitU32(word) => Some(*word),
        Expr::InvocationId { axis: 0 } | Expr::LocalId { axis: 0 } => Some(lane),
        Expr::Var(name) => env.get(name.as_str()).copied(),
        Expr::BinOp { op, left, right } => {
            let left = fold(left, lane, env)?;
            let right = fold(right, lane, env)?;
            Some(match op {
                BinOp::Add => left.wrapping_add(right),
                BinOp::Sub => left.wrapping_sub(right),
                BinOp::Mul => left.wrapping_mul(right),
                BinOp::WrappingAdd => left.wrapping_add(right),
                BinOp::WrappingSub => left.wrapping_sub(right),
                BinOp::Lt => u32::from(left < right),
                BinOp::Le => u32::from(left <= right),
                BinOp::Gt => u32::from(left > right),
                BinOp::Ge => u32::from(left >= right),
                BinOp::Eq => u32::from(left == right),
                BinOp::Ne => u32::from(left != right),
                BinOp::And => u32::from(left != 0 && right != 0),
                BinOp::Or => u32::from(left != 0 || right != 0),
                BinOp::Min => left.min(right),
                BinOp::Max => left.max(right),
                _ => return None,
            })
        }
        _ => None,
    }
}
