//! Byte-stability golden for the WGSL this backend emits.
//!
//! `vyre-emit-naga` already pins the module it produces from a descriptor. That
//! golden stops short of the driver: the wgpu path adds workgroup selection,
//! dispatch geometry, binding assignment, and the WGSL writer on top, and none
//! of that is covered by a descriptor-level corpus. This file pins the output of
//! [`vyre_driver_wgpu::emit::lower_with_config`], the entry every caller reaches
//! through `WgpuBackend`.
//!
//! What the corpus covers: constant stores; elementwise arithmetic, bitwise,
//! saturating/wrapping, and comparison binary operators; integer, transcendental,
//! numeric, and predicate unary operators; casts across `u32`/`i32`/`f32`;
//! `select` and `fma`; `if`/`else`; counted loops with a carried accumulator;
//! nested loop-in-branch control flow; `arrayLength` through `Expr::buf_len`;
//! invocation, workgroup, and local id reads; the atomic RMW set and
//! compare-exchange; a uniform binding; an explicitly sized workgroup; a trap
//! guard; and one case lowered under a non-default `DispatchConfig` so workgroup
//! override and dispatch geometry are pinned too.
//!
//! What it does not cover: subgroup intrinsics, async load/store and resume,
//! collectives, indirect dispatch, `f64` literals, and anything reached only
//! through an adapter feature that `EnabledFeatures::default()` leaves off.
//! Those paths are exercised by the live-device suites, not here.
//!
//! The corpus, the section format, and the comparison live in
//! `vyre_lower::artifact_golden`. This file supplies only the programs.

use std::path::PathBuf;

use vyre_driver::DispatchConfig;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_lower::artifact_golden;

fn golden_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-driver-wgpu/tests/golden/emitted_wgsl.txt")
}

const LANES: u32 = 8;

fn rw(name: &str, binding: u32, element: DataType) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadWrite, element).with_count(LANES)
}

fn ro(name: &str, binding: u32, element: DataType) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadOnly, element).with_count(LANES)
}

/// Store one expression per lane, built from the lane index.
fn per_lane(buffers: Vec<BufferDecl>, workgroup: [u32; 3], values: Vec<Expr>) -> Program {
    let body = values
        .into_iter()
        .enumerate()
        .map(|(lane, value)| {
            Node::store(
                "out",
                Expr::u32(u32::try_from(lane).expect("corpus lane count fits u32")),
                value,
            )
        })
        .collect();
    Program::wrapped(buffers, workgroup, body)
}

/// Store one `u32` expression per lane against a single read-write buffer.
fn u32_lanes(values: Vec<Expr>) -> Program {
    per_lane(vec![rw("out", 0, DataType::U32)], [1, 1, 1], values)
}

/// Store one `u32` expression per lane with two read-only `u32` inputs.
fn u32_binary_lanes(build: fn(Expr, Expr) -> Expr, count: u32) -> Program {
    let values = (0..count)
        .map(|lane| {
            build(
                Expr::load("a", Expr::u32(lane)),
                Expr::load("b", Expr::u32(lane)),
            )
        })
        .collect();
    per_lane(
        vec![
            rw("out", 0, DataType::U32),
            ro("a", 1, DataType::U32),
            ro("b", 2, DataType::U32),
        ],
        [1, 1, 1],
        values,
    )
}

/// Store one `f32` expression per lane with one read-only `f32` input.
fn f32_unary_lanes(builders: &[fn(Expr) -> Expr]) -> Program {
    let values = builders
        .iter()
        .enumerate()
        .map(|(lane, build)| {
            build(Expr::load(
                "a",
                Expr::u32(u32::try_from(lane).expect("corpus lane count fits u32")),
            ))
        })
        .collect();
    per_lane(
        vec![rw("out", 0, DataType::F32), ro("a", 1, DataType::F32)],
        [1, 1, 1],
        values,
    )
}

fn arith_binops() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    let b = || Expr::load("b", Expr::gid_x());
    per_lane(
        vec![
            rw("out", 0, DataType::U32),
            ro("a", 1, DataType::U32),
            ro("b", 2, DataType::U32),
        ],
        [1, 1, 1],
        vec![
            Expr::add(a(), b()),
            Expr::sub(a(), b()),
            Expr::mul(a(), b()),
            Expr::div(a(), b()),
            Expr::rem(a(), b()),
            Expr::min(a(), b()),
            Expr::max(a(), b()),
            Expr::mulhi(a(), b()),
            Expr::abs_diff(a(), b()),
        ],
    )
}

fn bitwise_binops() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    let b = || Expr::load("b", Expr::gid_x());
    per_lane(
        vec![
            rw("out", 0, DataType::U32),
            ro("a", 1, DataType::U32),
            ro("b", 2, DataType::U32),
        ],
        [1, 1, 1],
        vec![
            Expr::bitand(a(), b()),
            Expr::bitor(a(), b()),
            Expr::bitxor(a(), b()),
            Expr::shl(a(), b()),
            Expr::shr(a(), b()),
            Expr::rotate_left(a(), b()),
            Expr::rotate_right(a(), b()),
        ],
    )
}

fn saturating_binops() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    let b = || Expr::load("b", Expr::gid_x());
    per_lane(
        vec![
            rw("out", 0, DataType::U32),
            ro("a", 1, DataType::U32),
            ro("b", 2, DataType::U32),
        ],
        [1, 1, 1],
        vec![
            Expr::saturating_add(a(), b()),
            Expr::saturating_sub(a(), b()),
            Expr::saturating_mul(a(), b()),
            a().wrapping_add(b()),
            a().wrapping_sub(b()),
        ],
    )
}

fn comparison_binops() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    let b = || Expr::load("b", Expr::gid_x());
    let as_word = |cond: Expr| Expr::select(cond, Expr::u32(1), Expr::u32(0));
    per_lane(
        vec![
            rw("out", 0, DataType::U32),
            ro("a", 1, DataType::U32),
            ro("b", 2, DataType::U32),
        ],
        [1, 1, 1],
        vec![
            as_word(Expr::eq(a(), b())),
            as_word(Expr::ne(a(), b())),
            as_word(Expr::lt(a(), b())),
            as_word(Expr::le(a(), b())),
            as_word(Expr::gt(a(), b())),
            as_word(Expr::ge(a(), b())),
            as_word(Expr::and(Expr::lt(a(), b()), Expr::gt(a(), Expr::u32(0)))),
            as_word(Expr::or(Expr::lt(a(), b()), Expr::gt(a(), Expr::u32(0)))),
        ],
    )
}

fn integer_unops() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    per_lane(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Expr::bitnot(a()),
            Expr::reverse_bits(a()),
            Expr::popcount(a()),
            Expr::clz(a()),
            Expr::ctz(a()),
            Expr::select(Expr::not(Expr::eq(a(), Expr::u32(0))), a(), Expr::u32(1)),
        ],
    )
}

fn transcendental_unops() -> Program {
    f32_unary_lanes(&[
        Expr::sin,
        Expr::cos,
        Expr::tan,
        Expr::asin,
        Expr::acos,
        Expr::atan,
        Expr::sinh,
        Expr::cosh,
    ])
}

fn numeric_unops() -> Program {
    f32_unary_lanes(&[
        Expr::exp,
        Expr::exp2,
        Expr::log,
        Expr::log2,
        Expr::abs,
        Expr::sqrt,
        Expr::inverse_sqrt,
        Expr::reciprocal,
    ])
}

fn rounding_and_predicate_unops() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    let as_word = |cond: Expr| Expr::select(cond, Expr::f32(1.0), Expr::f32(0.0));
    per_lane(
        vec![rw("out", 0, DataType::F32), ro("a", 1, DataType::F32)],
        [1, 1, 1],
        vec![
            Expr::floor(a()),
            Expr::ceil(a()),
            Expr::round(a()),
            Expr::trunc(a()),
            Expr::sign(a()),
            Expr::tanh(a()),
            as_word(Expr::is_nan(a())),
            as_word(Expr::is_inf(a())),
        ],
    )
}

fn casts() -> Program {
    let word = || Expr::load("a", Expr::gid_x());
    per_lane(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Expr::cast(DataType::U32, Expr::cast(DataType::I32, word())),
            Expr::cast(DataType::U32, Expr::cast(DataType::F32, word())),
            Expr::cast(
                DataType::U32,
                Expr::cast(DataType::F32, Expr::cast(DataType::I32, word())),
            ),
            Expr::cast(DataType::U32, Expr::cast(DataType::Bool, word())),
        ],
    )
}

fn select_and_fma() -> Program {
    let a = || Expr::load("a", Expr::gid_x());
    let b = || Expr::load("b", Expr::gid_x());
    per_lane(
        vec![
            rw("out", 0, DataType::F32),
            ro("a", 1, DataType::F32),
            ro("b", 2, DataType::F32),
        ],
        [1, 1, 1],
        vec![
            Expr::select(Expr::lt(a(), b()), a(), b()),
            Expr::fma(a(), b(), Expr::f32(0.5)),
            Expr::fma(Expr::fma(a(), b(), a()), b(), b()),
        ],
    )
}

fn if_else_branch() -> Program {
    Program::wrapped(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![Node::If {
            cond: Expr::lt(Expr::load("a", Expr::gid_x()), Expr::u32(4)),
            then: vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::mul(Expr::load("a", Expr::gid_x()), Expr::u32(2)),
            )],
            otherwise: vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::sub(Expr::load("a", Expr::gid_x()), Expr::u32(4)),
            )],
        }],
    )
}

fn loop_accumulate() -> Program {
    Program::wrapped(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("acc", Expr::u32(0)),
            Node::loop_(
                "i",
                Expr::u32(0),
                Expr::u32(LANES),
                vec![Node::assign(
                    "acc",
                    Expr::add(Expr::var("acc"), Expr::load("a", Expr::var("i"))),
                )],
            ),
            Node::store("out", Expr::u32(0), Expr::var("acc")),
        ],
    )
}

fn nested_loop_in_branch() -> Program {
    Program::wrapped(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("acc", Expr::u32(0)),
            Node::If {
                cond: Expr::lt(Expr::gid_x(), Expr::u32(LANES)),
                then: vec![Node::loop_(
                    "i",
                    Expr::u32(0),
                    Expr::u32(4),
                    vec![Node::loop_(
                        "j",
                        Expr::u32(0),
                        Expr::u32(2),
                        vec![Node::assign(
                            "acc",
                            Expr::add(Expr::var("acc"), Expr::mul(Expr::var("i"), Expr::var("j"))),
                        )],
                    )],
                )],
                otherwise: vec![Node::assign("acc", Expr::u32(1))],
            },
            Node::store("out", Expr::gid_x(), Expr::var("acc")),
        ],
    )
}

fn buffer_length() -> Program {
    u32_lanes(vec![
        Expr::buf_len("out"),
        Expr::sub(Expr::buf_len("out"), Expr::u32(1)),
        Expr::min(Expr::gid_x(), Expr::buf_len("out")),
    ])
}

fn invocation_ids() -> Program {
    u32_lanes(vec![
        Expr::gid_x(),
        Expr::gid_y(),
        Expr::gid_z(),
        Expr::workgroup_x(),
        Expr::workgroup_y(),
        Expr::workgroup_z(),
        Expr::local_x(),
        Expr::local_y(),
    ])
}

fn atomic_rmw() -> Program {
    let value = || Expr::load("a", Expr::gid_x());
    let slot = || Expr::u32(0);
    Program::wrapped(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind("add", Expr::atomic_add("out", slot(), value())),
            Node::let_bind("or", Expr::atomic_or("out", slot(), value())),
            Node::let_bind("and", Expr::atomic_and("out", slot(), value())),
            Node::let_bind("xor", Expr::atomic_xor("out", slot(), value())),
            Node::let_bind("min", Expr::atomic_min("out", slot(), value())),
            Node::let_bind("max", Expr::atomic_max("out", slot(), value())),
            Node::let_bind("swap", Expr::atomic_exchange("out", slot(), value())),
            Node::store("out", Expr::u32(1), Expr::var("swap")),
        ],
    )
}

fn atomic_compare_exchange() -> Program {
    Program::wrapped(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::let_bind(
                "seen",
                Expr::atomic_compare_exchange(
                    "out",
                    Expr::u32(0),
                    Expr::u32(0),
                    Expr::load("a", Expr::gid_x()),
                ),
            ),
            Node::store("out", Expr::u32(1), Expr::var("seen")),
        ],
    )
}

fn uniform_binding() -> Program {
    Program::wrapped(
        vec![
            rw("out", 0, DataType::U32),
            BufferDecl::uniform("scale", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::mul(Expr::gid_x(), Expr::load("scale", Expr::u32(0))),
        )],
    )
}

fn explicit_workgroup() -> Program {
    Program::wrapped(
        vec![rw("out", 0, DataType::U32)],
        [64, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
    )
}

fn trap_guard() -> Program {
    Program::wrapped(
        vec![rw("out", 0, DataType::U32), ro("a", 1, DataType::U32)],
        [1, 1, 1],
        vec![
            Node::if_then(
                Expr::ge(Expr::gid_x(), Expr::buf_len("out")),
                vec![Node::trap(Expr::gid_x(), "out_of_range")],
            ),
            Node::store("out", Expr::gid_x(), Expr::load("a", Expr::gid_x())),
        ],
    )
}

/// The pinned program set, in golden order.
///
/// Each entry is rendered under the default dispatch policy except the last,
/// which pins the override path.
fn corpus() -> Vec<(&'static str, Program, DispatchConfig)> {
    let default = DispatchConfig::default;
    let mut overridden = DispatchConfig::default();
    overridden.workgroup_override = Some([32, 1, 1]);

    vec![
        (
            "constant_store",
            u32_lanes(vec![Expr::u32(7), Expr::u32(0), Expr::u32(u32::MAX)]),
            default(),
        ),
        (
            "elementwise_add",
            u32_binary_lanes(Expr::add, LANES),
            default(),
        ),
        ("arith_binops", arith_binops(), default()),
        ("bitwise_binops", bitwise_binops(), default()),
        ("saturating_binops", saturating_binops(), default()),
        ("comparison_binops", comparison_binops(), default()),
        ("integer_unops", integer_unops(), default()),
        ("transcendental_unops", transcendental_unops(), default()),
        ("numeric_unops", numeric_unops(), default()),
        (
            "rounding_and_predicate_unops",
            rounding_and_predicate_unops(),
            default(),
        ),
        ("casts", casts(), default()),
        ("select_and_fma", select_and_fma(), default()),
        ("if_else_branch", if_else_branch(), default()),
        ("loop_accumulate", loop_accumulate(), default()),
        ("nested_loop_in_branch", nested_loop_in_branch(), default()),
        ("buffer_length", buffer_length(), default()),
        ("invocation_ids", invocation_ids(), default()),
        ("atomic_rmw", atomic_rmw(), default()),
        (
            "atomic_compare_exchange",
            atomic_compare_exchange(),
            default(),
        ),
        ("uniform_binding", uniform_binding(), default()),
        ("explicit_workgroup", explicit_workgroup(), default()),
        ("trap_guard", trap_guard(), default()),
        (
            "workgroup_override_config",
            u32_binary_lanes(Expr::mul, LANES),
            overridden,
        ),
    ]
}

fn render_corpus() -> String {
    let cases = corpus();
    let sections: Vec<(&str, String)> = cases
        .iter()
        .map(|(id, program, config)| {
            let wgsl = vyre_driver_wgpu::emit::lower_with_config(program, config).unwrap_or_else(
                |error| panic!("Fix: corpus program `{id}` must lower to WGSL: {error:?}"),
            );
            (*id, wgsl)
        })
        .collect();
    artifact_golden::render_sections(sections)
}

/// The contract this file exists for: a refactor of this crate must not move a
/// single byte of emitted WGSL.
#[test]
fn emitted_wgsl_matches_the_pinned_corpus() {
    artifact_golden::assert_matches_golden(&golden_path(), &render_corpus());
}

/// A pinned corpus that silently stopped naming a case would keep passing while
/// covering less.
#[test]
fn pinned_corpus_names_every_program() {
    let golden = std::fs::read_to_string(golden_path()).expect("pinned WGSL corpus must exist");
    for (id, _, _) in corpus() {
        assert!(
            golden.contains(&format!("===== {id}\n")),
            "Fix: pinned WGSL corpus is missing case `{id}`; re-bless it."
        );
    }
}

/// Naga interns types and expressions as it walks a descriptor. An iteration
/// order that depended on a hash seed would pass the golden once and fail on the
/// next run rather than on the next refactor.
#[test]
fn emitted_wgsl_is_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

/// Regenerate the golden. Ignored so a normal run compares rather than blesses.
#[test]
#[ignore = "bless: rewrites the pinned WGSL corpus"]
fn bless_pinned_wgsl_corpus() {
    artifact_golden::write_golden(&golden_path(), &render_corpus());
}
