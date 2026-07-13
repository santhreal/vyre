//! Oracle-differential hunt over LOOP value-transforms (directive PRIMARY lane).
//!
//! The reference interpreter is the GOLD oracle: every loop transform, whether
//! run inside the full `optimize::optimize` pipeline (Release profile schedules
//! `loop_strip_mine`, `loop_unroll`, `loop_licm`, `loop_software_pipeline`, …)
//! or invoked directly, must preserve the byte-exact result of every loop
//! program for every input. This hunts the classic loop-miscompile shapes:
//! off-by-one bounds, dropped/duplicated iterations, loop-variable leakage into
//! prologue/epilogue, hoisting a non-invariant subexpression, and reordering a
//! loop-carried read-after-write (the reduction case). Unlike a single-pass
//! structural assertion, this drives real data through both the base and the
//! rewritten program and compares observable memory.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_fission::LoopFission;
use vyre_foundation::optimizer::passes::loops::loop_fusion::LoopFusion;
use vyre_foundation::optimizer::passes::loops::loop_licm::LoopLicm;
use vyre_foundation::optimizer::passes::loops::loop_software_pipeline::LoopSoftwarePipeline;
use vyre_foundation::optimizer::passes::loops::loop_strip_mine::LoopStripMine;
use vyre_foundation::optimizer::passes::loops::loop_unroll::LoopUnroll;
use vyre_foundation::optimizer::pre_lowering as optimize;
use vyre_reference::value::Value;

const N: u32 = 8;

fn input_value(xs: &[u32]) -> Value {
    Value::from(xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

/// Adversarial length-8 `buf_in` payloads: ascending, descending, all-equal,
/// the bit-pattern boundaries, and an alternating mix.
fn input_vectors() -> Vec<Vec<u32>> {
    vec![
        vec![10, 11, 12, 13, 14, 15, 16, 17],
        vec![17, 16, 15, 14, 13, 12, 11, 10],
        vec![7, 7, 7, 7, 7, 7, 7, 7],
        vec![
            0,
            0xFFFF_FFFF,
            0x8000_0000,
            1,
            0x7FFF_FFFF,
            2,
            0xFFFF_0000,
            0x0000_FFFF,
        ],
        vec![
            0x5555_5555,
            0xAAAA_AAAA,
            0x5555_5555,
            0xAAAA_AAAA,
            0,
            0,
            0xFFFF_FFFF,
            0xFFFF_FFFF,
        ],
    ]
}

fn read_only_in(binding: u32) -> BufferDecl {
    BufferDecl::storage("in", binding, BufferAccess::ReadOnly, DataType::U32).with_count(N)
}

fn loop_0_to_n(body: Vec<Node>) -> Node {
    Node::Loop {
        var: Ident::from("i"),
        from: Expr::u32(0),
        to: Expr::u32(N),
        body,
    }
}

fn iv() -> Expr {
    Expr::var("i")
}
fn in_i() -> Expr {
    Expr::load("in", iv())
}

/// Each builder returns a named loop program with one ReadOnly `in` (binding 0)
/// and one or more output buffers. The reference oracle maps the single input
/// payload to `in`; outputs are returned as results.
fn loop_programs() -> Vec<(&'static str, Program)> {
    // out[i] = in[i] * 3 + i   (index-affine value; unroll / strip-mine target)
    let index_affine = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("out", 1, DataType::U32).with_count(N),
        ],
        [1, 1, 1],
        vec![loop_0_to_n(vec![Node::store(
            "out",
            iv(),
            Expr::add(Expr::mul(in_i(), Expr::u32(3)), iv()),
        )])],
    );

    // out[i] = in[i] + (in[7] ^ 0x55)   (in[7]^0x55 is loop-invariant; LICM target)
    let licm_invariant = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("out", 1, DataType::U32).with_count(N),
        ],
        [1, 1, 1],
        vec![loop_0_to_n(vec![Node::store(
            "out",
            iv(),
            Expr::add(
                in_i(),
                Expr::bitxor(Expr::load("in", Expr::u32(N - 1)), Expr::u32(0x55)),
            ),
        )])],
    );

    // { let x = in[i]; out[i] = x + i }   (load-then-store; software-pipeline target)
    let load_store = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("out", 1, DataType::U32).with_count(N),
        ],
        [1, 1, 1],
        vec![loop_0_to_n(vec![
            Node::let_bind("x", in_i()),
            Node::store("out", iv(), Expr::add(Expr::var("x"), iv())),
        ])],
    );

    // out[i] = ((in[i] << 3) ^ (in[i] >> 2)) + (in[i] % 5) - i   (rich body)
    let rich_body = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("out", 1, DataType::U32).with_count(N),
        ],
        [1, 1, 1],
        vec![loop_0_to_n(vec![Node::store(
            "out",
            iv(),
            Expr::sub(
                Expr::add(
                    Expr::bitxor(
                        Expr::shl(in_i(), Expr::u32(3)),
                        Expr::shr(in_i(), Expr::u32(2)),
                    ),
                    Expr::rem(in_i(), Expr::u32(5)),
                ),
                iv(),
            ),
        )])],
    );

    // { out[i] = in[i] << 1; out[i+N] = in[i] >> 1 }   (two independent stores
    // into the two halves of one output buffer; fission target. vyre IR permits
    // at most one output buffer. V022, so disjoint sinks are index-disjoint
    // regions of a single buffer, which is the real fission shape anyway.)
    let fission = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("out", 1, DataType::U32).with_count(2 * N),
        ],
        [1, 1, 1],
        vec![loop_0_to_n(vec![
            Node::store("out", iv(), Expr::shl(in_i(), Expr::u32(1))),
            Node::store(
                "out",
                Expr::add(iv(), Expr::u32(N)),
                Expr::shr(in_i(), Expr::u32(1)),
            ),
        ])],
    );

    // loop i { out[i] = in[i] + 1 }  loop i { out[i+N] = in[i] * 2 }
    // (two adjacent loops over disjoint halves of one output; fusion target)
    let fusion = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("out", 1, DataType::U32).with_count(2 * N),
        ],
        [1, 1, 1],
        vec![
            loop_0_to_n(vec![Node::store(
                "out",
                iv(),
                Expr::add(in_i(), Expr::u32(1)),
            )]),
            loop_0_to_n(vec![Node::store(
                "out",
                Expr::add(iv(), Expr::u32(N)),
                Expr::mul(in_i(), Expr::u32(2)),
            )]),
        ],
    );

    // loop i { acc[0] = acc[0] + in[i] }   (loop-carried read-after-write; any
    // transform that reorders/parallelizes the dependency corrupts the sum)
    let reduction = Program::wrapped(
        vec![
            read_only_in(0),
            BufferDecl::output("acc", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![loop_0_to_n(vec![Node::store(
            "acc",
            Expr::u32(0),
            Expr::add(Expr::load("acc", Expr::u32(0)), in_i()),
        )])],
    );

    vec![
        ("index_affine", index_affine),
        ("licm_invariant", licm_invariant),
        ("load_store", load_store),
        ("rich_body", rich_body),
        ("fission", fission),
        ("fusion", fusion),
        ("reduction", reduction),
    ]
}

/// Sanity: the oracle actually computes the loop's real values (not a vacuous
/// empty/zero result). `acc = sum(1..=8) == 36`.
#[test]
fn reduction_oracle_computes_the_real_sum() {
    let (_, reduction) = loop_programs()
        .into_iter()
        .find(|(name, _)| *name == "reduction")
        .expect("reduction program present");
    let inputs = [input_value(&[1, 2, 3, 4, 5, 6, 7, 8])];
    let result = vyre_reference::reference_eval(&reduction, &inputs)
        .expect("reduction program must run on the reference interpreter");
    assert_eq!(
        result,
        vec![Value::from(36u32.to_le_bytes().to_vec())],
        "acc must equal sum(1..=8) == 36"
    );
}

#[test]
fn full_optimize_preserves_every_loop_program_value() {
    for (name, program) in loop_programs() {
        let optimized = optimize::optimize(program.clone());
        for vec in input_vectors() {
            let inputs = [input_value(&vec)];
            let base = vyre_reference::reference_eval(&program, &inputs)
                .unwrap_or_else(|e| panic!("base `{name}` must run on the reference oracle: {e}"));
            let opt = vyre_reference::reference_eval(&optimized, &inputs).unwrap_or_else(|e| {
                panic!("optimized `{name}` must run on the reference oracle: {e}")
            });
            assert_eq!(
                base, opt,
                "optimize::optimize changed the observable result of loop program `{name}` for input {vec:?}"
            );
        }
    }
}

#[test]
fn each_loop_pass_preserves_every_loop_program_value() {
    // (label, transform), every loop pass applied to every program must
    // preserve the oracle result, whether or not it fires on that shape.
    type Pass = fn(Program) -> vyre_foundation::optimizer::PassResult;
    let passes: [(&str, Pass); 6] = [
        ("loop_unroll", LoopUnroll::transform),
        ("loop_strip_mine", LoopStripMine::transform),
        ("loop_licm", LoopLicm::transform),
        ("loop_software_pipeline", LoopSoftwarePipeline::transform),
        ("loop_fission", LoopFission::transform),
        ("loop_fusion", LoopFusion::transform),
    ];

    for (name, program) in loop_programs() {
        let bases: Vec<(Vec<u32>, Vec<Value>)> = input_vectors()
            .into_iter()
            .map(|vec| {
                let inputs = [input_value(&vec)];
                let base = vyre_reference::reference_eval(&program, &inputs)
                    .unwrap_or_else(|e| panic!("base `{name}` must run on the oracle: {e}"));
                (vec, base)
            })
            .collect();

        for (pass_name, transform) in passes {
            let result = transform(program.clone());
            for (vec, base) in &bases {
                let inputs = [input_value(vec)];
                let after = vyre_reference::reference_eval(&result.program, &inputs)
                    .unwrap_or_else(|e| {
                        panic!("`{pass_name}`-transformed `{name}` must run on the oracle: {e}")
                    });
                assert_eq!(
                    base, &after,
                    "{pass_name} changed the observable result of loop program `{name}` for input {vec:?} (changed={})",
                    result.changed
                );
            }
        }
    }
}
