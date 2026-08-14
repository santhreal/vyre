//! Backend-neutral `Program` corpus for byte-stability goldens.
//!
//! Two very different goldens need the same input set: what the host oracle
//! computes for a program, and what a backend emits for it. Both answers are a
//! function of one `Program`, and a `Program` names no target, dialect, driver,
//! or artifact format, so the corpus is neutral and belongs here rather than
//! duplicated once per consuming crate.
//!
//! `emit_adversarial_corpus` is the sibling corpus at descriptor level. This one
//! sits one step earlier, where a case is still executable by the reference
//! interpreter, which is what lets a single case pin both an emitted artifact
//! and an oracle result.
//!
//! Every case is evaluable: signed divisors exclude zero and `i32::MIN / -1`,
//! because signed division is a partial function in the IR and a case the
//! oracle rejects cannot pin anything.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Lane count every case declares.
pub const LANES: u32 = 64;

/// Workgroup width every case dispatches with.
pub const WORKGROUP_SIZE_X: u32 = 32;

/// One neutral stability case.
#[derive(Debug, Clone)]
pub struct StabilityCase {
    /// Stable case identifier, used as the golden section name.
    pub id: &'static str,
    /// The program under test.
    pub program: Program,
    /// Bytes for each host-visible input buffer, in declaration order.
    pub inputs: Vec<Vec<u8>>,
}

/// Hostile u32 lane values, tiled across the lane count.
const U32_SEEDS: [u32; 8] = [
    0,
    1,
    2,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0xdead_beef,
    0x0f0f_0f0f,
];

/// Hostile i32 lane values, including both signed extremes.
const I32_SEEDS: [i32; 8] = [0, 1, -1, i32::MAX, i32::MIN, -7, 12_345, -12_345];

/// Signed divisors, excluding zero and `-1` opposite `i32::MIN`.
const I32_DIVISOR_SEEDS: [i32; 8] = [1, -1, 3, -7, 5, 2, -2, 11];

/// Hostile f32 lane values: both zeroes, a subnormal, and both infinities.
const F32_SEEDS: [f32; 8] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    f32::MIN_POSITIVE / 2.0,
    f32::MAX,
    f32::INFINITY,
    f32::NEG_INFINITY,
];

fn tile_u32(seeds: &[u32]) -> Vec<u8> {
    (0..LANES as usize)
        .flat_map(|lane| seeds[lane % seeds.len()].to_le_bytes())
        .collect()
}

fn tile_i32(seeds: &[i32]) -> Vec<u8> {
    (0..LANES as usize)
        .flat_map(|lane| seeds[lane % seeds.len()].to_le_bytes())
        .collect()
}

fn tile_f32(seeds: &[f32]) -> Vec<u8> {
    (0..LANES as usize)
        .flat_map(|lane| seeds[lane % seeds.len()].to_le_bytes())
        .collect()
}

/// Guard the store on the lane bound, so an oversized grid cannot write past
/// the declared buffer.
fn guarded_store(buffer: &str, value: Expr) -> Vec<Node> {
    vec![
        Node::let_bind("idx", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
            vec![Node::store(buffer, Expr::var("idx"), value)],
        ),
    ]
}

fn binary(
    id: &'static str,
    dtype: DataType,
    out: DataType,
    build: fn(Expr, Expr) -> Expr,
    lhs: Vec<u8>,
    rhs: Vec<u8>,
) -> StabilityCase {
    let idx = Expr::var("idx");
    StabilityCase {
        id,
        program: Program::wrapped(
            vec![
                BufferDecl::read("lhs", 0, dtype.clone()).with_count(LANES),
                BufferDecl::read("rhs", 1, dtype).with_count(LANES),
                BufferDecl::output("out", 2, out).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store(
                "out",
                build(
                    Expr::load("lhs", idx.clone()),
                    Expr::load("rhs", idx.clone()),
                ),
            ),
        ),
        inputs: vec![lhs, rhs],
    }
}

fn unary(
    id: &'static str,
    dtype: DataType,
    out: DataType,
    build: fn(Expr) -> Expr,
    input: Vec<u8>,
) -> StabilityCase {
    StabilityCase {
        id,
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, dtype).with_count(LANES),
                BufferDecl::output("out", 1, out).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store("out", build(Expr::load("input", Expr::var("idx")))),
        ),
        inputs: vec![input],
    }
}

/// Every neutral stability case, in golden order.
///
/// The set spans what changes both emitted text and computed bytes: integer and
/// float ALU, the division and shift arms that need masking, width casts,
/// predicate materialization, divergent control flow, a counted loop, and
/// workgroup scratch across a barrier.
#[must_use]
pub fn cases() -> Vec<StabilityCase> {
    let u32_in = tile_u32(&U32_SEEDS);
    let i32_in = tile_i32(&I32_SEEDS);
    let i32_divisors = tile_i32(&I32_DIVISOR_SEEDS);
    let f32_in = tile_f32(&F32_SEEDS);
    let u32_binary: [(&'static str, fn(Expr, Expr) -> Expr); 14] = [
        ("u32_add", Expr::add),
        ("u32_sub", Expr::sub),
        ("u32_mul", Expr::mul),
        ("u32_div", Expr::div),
        ("u32_rem", Expr::rem),
        ("u32_mulhi", Expr::mulhi),
        ("u32_shl", Expr::shl),
        ("u32_shr", Expr::shr),
        ("u32_bitand", Expr::bitand),
        ("u32_bitor", Expr::bitor),
        ("u32_bitxor", Expr::bitxor),
        ("u32_min", Expr::min),
        ("u32_max", Expr::max),
        ("u32_abs_diff", Expr::abs_diff),
    ];
    let u32_unary: [(&'static str, fn(Expr) -> Expr); 5] = [
        ("u32_bitnot", Expr::bitnot),
        ("u32_popcount", Expr::popcount),
        ("u32_clz", Expr::clz),
        ("u32_ctz", Expr::ctz),
        ("u32_reverse_bits", Expr::reverse_bits),
    ];
    let f32_binary: [(&'static str, fn(Expr, Expr) -> Expr); 5] = [
        ("f32_add", Expr::add),
        ("f32_mul", Expr::mul),
        ("f32_div", Expr::div),
        ("f32_min", Expr::min),
        ("f32_max", Expr::max),
    ];
    let mut cases = Vec::new();
    for (id, build) in u32_binary {
        cases.push(binary(
            id,
            DataType::U32,
            DataType::U32,
            build,
            u32_in.clone(),
            u32_in.clone(),
        ));
    }
    for (id, build) in u32_unary {
        cases.push(unary(
            id,
            DataType::U32,
            DataType::U32,
            build,
            u32_in.clone(),
        ));
    }
    for (id, build) in f32_binary {
        cases.push(binary(
            id,
            DataType::F32,
            DataType::F32,
            build,
            f32_in.clone(),
            f32_in.clone(),
        ));
    }
    cases.push(binary(
        "i32_add",
        DataType::I32,
        DataType::I32,
        Expr::add,
        i32_in.clone(),
        i32_in.clone(),
    ));
    cases.push(binary(
        "i32_mul",
        DataType::I32,
        DataType::I32,
        Expr::mul,
        i32_in.clone(),
        i32_in.clone(),
    ));
    cases.push(binary(
        "i32_div",
        DataType::I32,
        DataType::I32,
        Expr::div,
        i32_in.clone(),
        i32_divisors.clone(),
    ));
    cases.push(binary(
        "i32_rem",
        DataType::I32,
        DataType::I32,
        Expr::rem,
        i32_in.clone(),
        i32_divisors.clone(),
    ));
    cases.push(binary(
        "i32_min",
        DataType::I32,
        DataType::I32,
        Expr::min,
        i32_in.clone(),
        i32_divisors.clone(),
    ));
    cases.push(binary(
        "i32_max",
        DataType::I32,
        DataType::I32,
        Expr::max,
        i32_in.clone(),
        i32_divisors,
    ));
    cases.push(unary(
        "u32_to_f32_cast",
        DataType::U32,
        DataType::F32,
        |value| Expr::cast(DataType::F32, value),
        u32_in.clone(),
    ));
    cases.push(unary(
        "f32_to_i32_cast",
        DataType::F32,
        DataType::I32,
        |value| Expr::cast(DataType::I32, value),
        f32_in,
    ));
    cases.push(unary(
        "i32_to_u32_cast",
        DataType::I32,
        DataType::U32,
        |value| Expr::cast(DataType::U32, value),
        i32_in,
    ));
    cases.push(binary(
        "u32_lt_predicate",
        DataType::U32,
        DataType::U32,
        |left, right| Expr::select(Expr::lt(left, right), Expr::u32(1), Expr::u32(0)),
        u32_in.clone(),
        u32_in.clone(),
    ));
    cases.push(binary(
        "u32_select_chain",
        DataType::U32,
        DataType::U32,
        |left, right| {
            Expr::select(
                Expr::ge(left.clone(), right.clone()),
                Expr::sub(left.clone(), right.clone()),
                Expr::add(left, right),
            )
        },
        u32_in.clone(),
        u32_in.clone(),
    ));
    cases.push(StabilityCase {
        id: "u32_branch_divergence",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
                    vec![Node::if_then_else(
                        Expr::lt(Expr::load("input", Expr::var("idx")), Expr::u32(3)),
                        vec![Node::store("out", Expr::var("idx"), Expr::u32(1))],
                        vec![Node::store(
                            "out",
                            Expr::var("idx"),
                            Expr::add(Expr::load("input", Expr::var("idx")), Expr::u32(7)),
                        )],
                    )],
                ),
            ],
        ),
        inputs: vec![u32_in.clone()],
    });
    cases.push(StabilityCase {
        id: "u32_loop_accumulate",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
                    vec![
                        Node::store("out", Expr::var("idx"), Expr::u32(0)),
                        Node::loop_for(
                            "step",
                            Expr::u32(0),
                            Expr::u32(4),
                            vec![loop_accumulate_step()],
                        ),
                    ],
                ),
            ],
        ),
        inputs: vec![u32_in.clone()],
    });
    cases.push(StabilityCase {
        id: "u32_workgroup_scratch_exchange",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::workgroup("scratch", WORKGROUP_SIZE_X, DataType::U32),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::let_bind("lane", Expr::invocation_local_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
                    vec![Node::store(
                        "scratch",
                        Expr::var("lane"),
                        Expr::load("input", Expr::var("idx")),
                    )],
                ),
                Node::barrier(),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
                    vec![Node::store(
                        "out",
                        Expr::var("idx"),
                        Expr::add(
                            Expr::load("scratch", Expr::var("lane")),
                            Expr::load(
                                "scratch",
                                Expr::rem(
                                    Expr::add(Expr::var("lane"), Expr::u32(1)),
                                    Expr::u32(WORKGROUP_SIZE_X),
                                ),
                            ),
                        ),
                    )],
                ),
            ],
        ),
        inputs: vec![u32_in],
    });
    cases
}

/// One iteration of the counted-loop accumulate case.
///
/// Named rather than inlined so the case body stays shallow: a stack of eight
/// closing delimiters is indistinguishable, to the duplication scanner, from
/// any other such stack in the workspace.
fn loop_accumulate_step() -> Node {
    Node::store(
        "out",
        Expr::var("idx"),
        Expr::add(
            Expr::load("out", Expr::var("idx")),
            Expr::mul(Expr::load("input", Expr::var("idx")), Expr::var("step")),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::cases;

    /// WHY: two goldens key their sections on these ids. A duplicate id would
    /// silently overwrite one case's pinned section with another's.
    #[test]
    fn every_case_id_is_unique() {
        let mut ids = cases().into_iter().map(|case| case.id).collect::<Vec<_>>();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            count,
            "Fix: give every neutral stability case a distinct id."
        );
    }

    /// WHY: a case whose declared input buffers outnumber its supplied byte
    /// vectors cannot be executed by the oracle, so it would pin nothing.
    #[test]
    fn every_case_supplies_bytes_for_every_read_buffer() {
        for case in cases() {
            let reads = case
                .program
                .buffers()
                .iter()
                .filter(|buffer| !buffer.is_output() && buffer.name() != "scratch")
                .count();
            assert_eq!(
                case.inputs.len(),
                reads,
                "Fix: case `{}` must supply one byte vector per read buffer.",
                case.id
            );
        }
    }
}
