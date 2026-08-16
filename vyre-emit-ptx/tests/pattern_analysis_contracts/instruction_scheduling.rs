//! `instruction_scheduling` pattern analysis contracts.

use vyre_emit_ptx::patterns::instruction_scheduling::*;
use vyre_foundation::ir::BinOp;
use vyre_lower::descriptor_builder::{body, descriptor, lit, op};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

fn linear_chain(length: usize) -> KernelDescriptor {
    // x0 = literal; x1 = x0 + lit; x2 = x1 + lit; ...
    let mut ops = vec![lit(0, 0)];
    for i in 1..length {
        ops.push(lit(1, 100 + i as u32));
        ops.push(op(
            KernelOpKind::BinOpKind(BinOp::Add),
            [(i - 1) as u32, 100 + i as u32],
            i as u32,
        ));
    }
    descriptor("chain")
        .body(
            body()
                .ops(ops)
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build()
}

#[test]
fn empty_kernel_no_chains() {
    let desc = descriptor("empty").build();
    let h = analyze(&desc);
    assert!(h.long_chains.is_empty());
    assert_eq!(h.total_op_count, 0);
}

#[test]
fn short_independent_ops_no_long_chain() {
    let desc = descriptor("indep")
        .body(
            body()
                .ops([lit(0, 0), lit(0, 1), lit(0, 2)])
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let h = analyze(&desc);
    assert!(h.long_chains.is_empty());
    assert_eq!(h.total_op_count, 3);
}

#[test]
fn long_dep_chain_detected() {
    // Build a chain of length 8 (1 literal + 7 add hops).
    let desc = linear_chain(8);
    let h = analyze(&desc);
    // Note: the linear_chain helper interleaves literals so the
    // chain detection sees: lit(r0); lit(r101); add(r0, r101)→r1;
    // lit(r102); add(r1, r102)→r2; ...  -  chain reads previous
    // hop's result through the Add op. The dependency-chain
    // detector should find at least one long chain.
    assert!(!h.long_chains.is_empty());
    assert!(h.longest_chain() >= LONG_CHAIN_THRESHOLD);
}

#[test]
fn longest_chain_aggregates_correctly() {
    let h = SchedulingHints {
        kernel_id: "k".into(),
        long_chains: vec![
            DependencyChain {
                start_op_index: 0,
                length: 5,
            },
            DependencyChain {
                start_op_index: 10,
                length: 12,
            },
            DependencyChain {
                start_op_index: 25,
                length: 8,
            },
        ],
        total_op_count: 50,
    };
    assert_eq!(h.long_chain_count(), 3);
    assert_eq!(h.longest_chain(), 12);
    assert_eq!(h.schedule_latency_pressure(), 36);
}

#[test]
fn longest_chain_zero_when_empty() {
    let h = SchedulingHints {
        kernel_id: "k".into(),
        long_chains: vec![],
        total_op_count: 0,
    };
    assert_eq!(h.longest_chain(), 0);
}

#[test]
fn threshold_constant_is_documented() {
    assert_eq!(LONG_CHAIN_THRESHOLD, 4);
}
