use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, Correctness, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::suite::SuiteKind;
use crate::cases::harness::{CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::reference_sample::timed_reference;
use rand::{RngExt, SeedableRng};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Lanes, and so the element count of both buffers.
const LANES: u32 = 1024;
/// Independent live variables the register allocator has to keep alive at once.
const LIVE_VARIABLES: usize = 100;

const REGISTER_EXHAUSTION_SUITES: &[SuiteKind] =
    &[SuiteKind::Adversarial, SuiteKind::Deep, SuiteKind::Release];

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "adversarial.register_exhaustion.u32_1024",
    name: "Register Exhaustion",
    summary: "Generates a highly-nested set of independent live variables to stress-test register allocators",
    tags: &["adversarial", "compiler"],
    layer: BenchLayer::Backend,
    workload: WorkloadClass::Adversarial,
    suites: REGISTER_EXHAUSTION_SUITES,
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<RegisterExhaustionPrepared> = CaseOps {
    build: prepare,
    measure,
    verify,
    program: |prepared| Some(&prepared.program),
    fingerprint: None,
    bytes_touched: |prepared| crate::api::case::static_program_bytes_touched(&prepared.program),
};

pub(crate) static CASE: HarnessCase<RegisterExhaustionPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

pub(crate) struct RegisterExhaustionPrepared {
    program: Program,
    inputs: [Vec<u8>; 1],
}

fn prepare(_ctx: &mut BenchContext) -> Result<RegisterExhaustionPrepared, BenchError> {
    // Seeded, so the bytes are the same every sample. Generated once here rather
    // than re-filled per sample inside the measured loop.
    let mut input = vec![0u8; LANES as usize * 4];
    rand::rngs::StdRng::seed_from_u64(1337).fill(input.as_mut_slice());

    Ok(RegisterExhaustionPrepared {
        program: register_exhaustion_program(),
        inputs: [input],
    })
}

/// The program under measurement: `LIVE_VARIABLES` independent values, a
/// mixing loop that keeps every one of them live, and a reduction that
/// consumes them all.
fn register_exhaustion_program() -> Program {
    let mut body = Vec::with_capacity(LIVE_VARIABLES + 3);
    body.push(Node::let_bind("tid", Expr::gid_x()));

    // The independent live variables.
    for index in 0..LIVE_VARIABLES {
        body.push(Node::let_bind(
            format!("v{index}"),
            Expr::add(Expr::var("tid"), Expr::u32(index as u32)),
        ));
    }

    // A mixing loop so none of them is dead before the reduce below.
    let mut loop_body = Vec::new();
    for index in 0..LIVE_VARIABLES {
        let next = (index + 1) % LIVE_VARIABLES;
        loop_body.push(Node::assign(
            format!("v{index}"),
            Expr::add(
                Expr::var(format!("v{index}")),
                Expr::var(format!("v{next}")),
            ),
        ));
    }

    body.push(Node::Loop {
        var: "iter".into(),
        from: Expr::u32(0),
        to: Expr::u32(10),
        body: loop_body,
    });

    // A reduce over all of them, so none is eliminated as dead.
    let reduced = balanced_sum(
        (0..LIVE_VARIABLES)
            .map(|index| Expr::var(format!("v{index}")))
            .collect(),
    );
    body.push(Node::store("out", Expr::var("tid"), reduced));

    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(LANES),
            BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
        ],
        [256, 1, 1],
        body,
    )
}

/// Sum `terms` through a balanced tree.
///
/// A chain that adds each term to the running total in turn nests one
/// expression per term, which for a hundred of them is deeper than the IR wire
/// format decodes, and artifact preparation refuses the program before it
/// reaches a device. Pairing the terms instead makes the depth logarithmic in
/// their count. `u32` addition wraps and is associative, so the value is the
/// one the reference computes by adding them in order.
fn balanced_sum(mut terms: Vec<Expr>) -> Expr {
    while terms.len() > 1 {
        let mut folded = Vec::with_capacity(terms.len().div_ceil(2));
        let mut pending = terms.into_iter();
        while let Some(left) = pending.next() {
            folded.push(match pending.next() {
                Some(right) => Expr::add(left, right),
                None => left,
            });
        }
        terms = folded;
    }
    terms.pop().unwrap_or_else(|| Expr::u32(0))
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut RegisterExhaustionPrepared,
) -> Result<BenchRun, BenchError> {
    let program = &prepared.program;
    let input = &prepared.inputs[0];

    let timed = ctx
        .dispatch_timed(
            program,
            &prepared.inputs,
            &vyre_driver::DispatchConfig::default(),
        )
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;

    let (baseline, elapsed_ref) =
        timed_reference(|| cpu_register_exhaustion_outputs(LANES as usize));

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(timed.wall_ns),
            dispatch_ns: timed.device_ns,
            input_bytes: Some(input.len() as u64),
            output_bytes: Some(timed.outputs.iter().map(Vec::len).sum::<usize>() as u64),
            bytes_read: Some(input.len() as u64),
            bytes_written: Some(timed.outputs.iter().map(Vec::len).sum::<usize>() as u64),
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(elapsed_ref),
            ..Default::default()
        }),
        outputs: timed.outputs,
        baseline_outputs: Some(vec![baseline]),
    })
}

fn verify(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

fn cpu_register_exhaustion_outputs(lanes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(lanes * 4);
    for tid in 0..lanes {
        let tid = tid as u32;
        let mut values = [0u32; 100];
        for (i, value) in values.iter_mut().enumerate() {
            *value = tid.wrapping_add(i as u32);
        }
        for _ in 0..10 {
            for i in 0..100 {
                let next = (i + 1) % 100;
                values[i] = values[i].wrapping_add(values[next]);
            }
        }
        let reduced = values
            .iter()
            .copied()
            .fold(0u32, |acc, value| acc.wrapping_add(value));
        out.extend_from_slice(&reduced.to_le_bytes());
    }
    out
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: artifact preparation decodes the program from its wire form, and
    /// the decoder refuses a nesting depth a hostile blob could use to overflow
    /// the stack. A producer that builds its reduction as a chain crosses that
    /// limit at a term count it never states, and the case fails before any
    /// kernel runs. The contract is the round trip itself, so the limit stays
    /// owned by the wire format rather than restated here.
    #[test]
    fn the_measured_program_survives_the_wire_format() {
        let program = register_exhaustion_program();
        let bytes = program
            .to_wire()
            .expect("the measured program must encode to the IR wire format");
        let decoded = Program::from_wire(&bytes)
            .expect("the measured program must decode from the IR wire format");
        assert_eq!(
            decoded.fingerprint(),
            program.fingerprint(),
            "the round trip must preserve the program"
        );
    }

    /// WHY: the balanced tree exists to keep the depth bounded, and it is only
    /// correct because it sums the same terms. An odd count is the case a
    /// pairing fold drops a term in, so both parities are read, and the value
    /// is checked against the order the reference adds them in.
    #[test]
    fn a_balanced_sum_adds_every_term_at_any_count() {
        for count in 0u32..=9 {
            let terms = (0..count).map(Expr::u32).collect::<Vec<_>>();
            let folded = balanced_sum(terms);
            let expected = (0..count).fold(0u32, u32::wrapping_add);
            assert_eq!(
                constant_fold_u32(&folded),
                Some(expected),
                "{count} terms must sum to {expected}"
            );
        }
    }

    /// Value of a tree of `u32` literals and additions.
    fn constant_fold_u32(expr: &Expr) -> Option<u32> {
        match expr {
            Expr::LitU32(value) => Some(*value),
            Expr::BinOp {
                op: vyre_foundation::ir::BinOp::Add,
                left,
                right,
            } => Some(constant_fold_u32(left)?.wrapping_add(constant_fold_u32(right)?)),
            _ => None,
        }
    }
}
