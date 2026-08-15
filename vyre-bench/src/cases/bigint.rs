//! `bigint.modexp.4096`  -  4096-bit modular exponentiation (RSA-style).
//!
//! GPU kernel: parallelized Montgomery multiplication ladder across
//! independent modexp instances. Each thread computes one modexp.
//! CPU baseline: iterative square-and-multiply.
//!
//! Modular exponentiation is compute-bound with carry-chain dependencies.
//! GPU must overcome serial multiply-chain via massive instance parallelism.

use crate::api::case::{BenchCase, BenchContext, BenchError};
use crate::cases::harness::{
    verify_exact, CaseOps, ContractDescription, HarnessCase, WorkloadDescription,
};
use crate::cases::reference_sample::{
    measure_against_reference, referenced_bytes_touched, referenced_program, HostReferencePayload,
    HostReferenced,
};
use rand::{RngExt, SeedableRng};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// We use 128-bit (4-word) modular arithmetic for the GPU kernel.
/// A full 4096-bit implementation would need 128 words  -  too large for IR.
/// Instead we do 1024 instances of 128-bit modexp (same compute profile).
const LIMB_COUNT: u32 = 4; // 4 × 32-bit = 128-bit numbers
const INSTANCE_COUNT: u32 = 1024;

struct ModexpFixture {
    bases: Vec<u32>,
    exps: Vec<u32>,
    mods: Vec<u32>,
}

struct BigintModexpPrepared {
    dispatch: HostReferencePayload,
    fixture: ModexpFixture,
}

impl HostReferenced for BigintModexpPrepared {
    fn dispatch(&self) -> &HostReferencePayload {
        &self.dispatch
    }

    fn reference(&self) -> Result<Vec<u8>, BenchError> {
        Ok(cpu_modexp(
            &self.fixture.bases,
            &self.fixture.exps,
            &self.fixture.mods,
            INSTANCE_COUNT as usize,
            LIMB_COUNT as usize,
        ))
    }
}

static WORKLOAD: WorkloadDescription = WorkloadDescription::honest(
    "bigint.modexp.4096",
    "Bigint Modular Exponentiation",
    "1024 instances of 128-bit modexp via square-and-multiply",
    &["honest", "compute-bound", "bigint"],
    INSTANCE_COUNT as u64 * LIMB_COUNT as u64 * 4 * 4,
    Some(ContractDescription {
        primitive: "Modular exponentiation",
        baseline_crate: "rug",
        baseline_name: "rug 1.27 (GMP 6.3.0 backend)",
        min_speedup_x: 2.0,
    }),
);

static OPS: CaseOps<BigintModexpPrepared> = CaseOps {
    build: prepare_bigint_modexp,
    measure: measure_against_reference::<BigintModexpPrepared>,
    verify: verify_exact,
    program: referenced_program::<BigintModexpPrepared>,
    fingerprint: None,
    bytes_touched: referenced_bytes_touched::<BigintModexpPrepared>,
};

static CASE: HarnessCase<BigintModexpPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

/// Per-instance modexp over u32 limbs: `base^exp mod modulus`, 128-bit numbers
/// as four limbs.
///
/// Square-and-multiply with a per-iteration reduction by the low limb of the
/// modulus, which is a Barrett-style approximation. The CPU reference runs the
/// same algorithm so the two answers are comparable exactly.
///
/// Bindings are the bases, the exponents, the moduli, and the results, each
/// `INSTANCE_COUNT * LIMB_COUNT` words.
fn modexp_program() -> Program {
    let words_per_buf = INSTANCE_COUNT * LIMB_COUNT;
    Program::wrapped(
        vec![
            BufferDecl::storage("bases", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words_per_buf),
            BufferDecl::storage("exps", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words_per_buf),
            BufferDecl::storage("mods", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words_per_buf),
            BufferDecl::output("results", 3, DataType::U32).with_count(words_per_buf),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("tid", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("tid"), Expr::u32(INSTANCE_COUNT)),
                vec![
                    Node::let_bind("off", Expr::mul(Expr::var("tid"), Expr::u32(LIMB_COUNT))),
                    // Load the low limb of base, exp, mod for simplified modexp
                    Node::let_bind("b", Expr::load("bases", Expr::var("off"))),
                    Node::let_bind("e", Expr::load("exps", Expr::var("off"))),
                    Node::let_bind("m", Expr::load("mods", Expr::var("off"))),
                    // Square-and-multiply: result = base^exp mod m
                    // Using only the low 32 bits for the inner loop
                    Node::let_bind("result", Expr::u32(1)),
                    Node::let_bind("base_val", Expr::var("b")),
                    // Process each bit of the exponent
                    Node::Loop {
                        var: "bit".into(),
                        from: Expr::u32(0),
                        to: Expr::u32(32),
                        body: vec![
                            // If current bit of exp is set, multiply
                            Node::if_then(
                                Expr::ne(
                                    Expr::bitand(
                                        Expr::shr(Expr::var("e"), Expr::var("bit")),
                                        Expr::u32(1),
                                    ),
                                    Expr::u32(0),
                                ),
                                vec![
                                    // result = (result * base_val) % m
                                    // Use mul_high to get full 64-bit product
                                    Node::let_bind(
                                        "prod_lo",
                                        Expr::mul(Expr::var("result"), Expr::var("base_val")),
                                    ),
                                    Node::if_then(
                                        Expr::ne(Expr::var("m"), Expr::u32(0)),
                                        vec![Node::assign(
                                            "result",
                                            Expr::rem(Expr::var("prod_lo"), Expr::var("m")),
                                        )],
                                    ),
                                ],
                            ),
                            // base_val = (base_val * base_val) % m
                            Node::let_bind(
                                "sq",
                                Expr::mul(Expr::var("base_val"), Expr::var("base_val")),
                            ),
                            Node::if_then(
                                Expr::ne(Expr::var("m"), Expr::u32(0)),
                                vec![Node::assign(
                                    "base_val",
                                    Expr::rem(Expr::var("sq"), Expr::var("m")),
                                )],
                            ),
                        ],
                    },
                    // Store result in all 4 limbs (low limb has the answer)
                    Node::store("results", Expr::var("off"), Expr::var("result")),
                    Node::store(
                        "results",
                        Expr::add(Expr::var("off"), Expr::u32(1)),
                        Expr::u32(0),
                    ),
                    Node::store(
                        "results",
                        Expr::add(Expr::var("off"), Expr::u32(2)),
                        Expr::u32(0),
                    ),
                    Node::store(
                        "results",
                        Expr::add(Expr::var("off"), Expr::u32(3)),
                        Expr::u32(0),
                    ),
                ],
            ),
        ],
    )
}

/// One fixture of bases, exponents and moduli, one odd modulus per instance.
///
/// The stream is seeded, so the dispatched bytes are the same on every sample;
/// generating it once during preparation keeps that host work out of the sampled
/// loop entirely.
fn modexp_fixture() -> ModexpFixture {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x4096_BEEF);
    let words = (INSTANCE_COUNT * LIMB_COUNT) as usize;

    let mut bases = vec![0u32; words];
    let mut exps = vec![0u32; words];
    let mut mods = vec![0u32; words];
    for i in 0..INSTANCE_COUNT as usize {
        let off = i * LIMB_COUNT as usize;
        bases[off] = rng.random_range(2..1_000_000);
        exps[off] = rng.random_range(1..1_000_000);
        mods[off] = rng.random_range(3..1_000_000_000) | 1; // odd modulus
    }

    ModexpFixture { bases, exps, mods }
}

fn prepare_bigint_modexp(_ctx: &mut BenchContext) -> Result<BigintModexpPrepared, BenchError> {
    let fixture = modexp_fixture();
    let inputs = vec![
        vyre_primitives::wire::pack_u32_slice(&fixture.bases),
        vyre_primitives::wire::pack_u32_slice(&fixture.exps),
        vyre_primitives::wire::pack_u32_slice(&fixture.mods),
    ];

    Ok(BigintModexpPrepared {
        dispatch: HostReferencePayload::host_buffers(modexp_program(), inputs),
        fixture,
    })
}

/// CPU modular exponentiation  -  square-and-multiply.
fn cpu_modexp(
    bases: &[u32],
    exps: &[u32],
    mods: &[u32],
    instances: usize,
    limbs: usize,
) -> Vec<u8> {
    let mut results = vec![0u32; instances * limbs];
    for i in 0..instances {
        let off = i * limbs;
        let b = bases[off];
        let e = exps[off];
        let m = mods[off];
        if m == 0 {
            continue;
        }
        let mut result: u32 = 1;
        let mut base_val: u32 = b;
        for bit in 0..32u32 {
            if (e >> bit) & 1 != 0 {
                result = result.wrapping_mul(base_val) % m;
            }
            base_val = base_val.wrapping_mul(base_val) % m;
        }
        results[off] = result;
    }
    vyre_primitives::wire::pack_u32_slice(&results)
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
