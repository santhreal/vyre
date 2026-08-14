//! The capability contract suite: what `scan` requires of a program, what
//! `check_backend_capabilities` enforces against a backend, and a gate that
//! fails closed when either surface grows a member no case names.
//!
//! `program_caps` used to carry an inline `mod tests` that restated this suite:
//! the same `Program::wrapped` scalar fixture, the same nine-argument
//! all-false backend call. Both copies exercised only the public surface, so
//! one of them was a copy for no reason and the two drifted - the inline copy
//! covered the collective transport shape and the integration copy covered
//! `MissingCapability`'s `std::error::Error` impl, and neither covered both.
//! This file is the one owner.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use proptest::prelude::*;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, Node, Program,
};
use vyre_foundation::program_caps::{
    check_backend_capabilities, scan, MissingCapability, RequiredCapabilities,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A one-workgroup program over a single read-write `u32` output buffer.
///
/// Every scan case that detects a capability from a node rather than from a
/// buffer element type wraps its node here, so the buffer table is never the
/// thing under test.
fn scalar_program(entry: Vec<Node>) -> Program {
    element_program(DataType::U32, entry)
}

/// The same shape as [`scalar_program`] with the output element type chosen, for
/// the capabilities `scan` reads off a buffer declaration.
fn element_program(element: DataType, entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::storage(
            "out",
            0,
            BufferAccess::ReadWrite,
            element,
        )],
        [1, 1, 1],
        entry,
    )
}

/// A 64-lane program over the `input`/`out` pair the collective nodes address.
fn collective_program(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(16),
        ],
        [64, 1, 1],
        entry,
    )
}

/// Advertised backend capabilities, named rather than positional.
///
/// `check_backend_capabilities` takes six `bool`s in a row. Written out at a
/// call site they are unreadable and one transposed pair silently tests the
/// wrong capability, which is why every case in this file builds them here.
#[derive(Clone, Copy)]
struct Advertised {
    subgroup_ops: bool,
    half_precision: bool,
    brain_float: bool,
    indirect_dispatch: bool,
    trap_propagation: bool,
    distributed_collectives: bool,
    max_workgroup_size: [u32; 3],
}

impl Advertised {
    /// A backend that advertises nothing and caps workgroups at 64 lanes.
    const fn none() -> Self {
        Self {
            subgroup_ops: false,
            half_precision: false,
            brain_float: false,
            indirect_dispatch: false,
            trap_propagation: false,
            distributed_collectives: false,
            max_workgroup_size: [64, 1, 1],
        }
    }

    /// A backend that advertises everything and caps workgroups at 128 lanes.
    const fn all() -> Self {
        Self {
            subgroup_ops: true,
            half_precision: true,
            brain_float: true,
            indirect_dispatch: true,
            trap_propagation: true,
            distributed_collectives: true,
            max_workgroup_size: [128, 1, 1],
        }
    }

    const fn with_max_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.max_workgroup_size = size;
        self
    }

    fn check(self, required: &RequiredCapabilities) -> Result<(), MissingCapability> {
        check_backend_capabilities(
            "test_backend",
            self.subgroup_ops,
            self.half_precision,
            self.brain_float,
            self.indirect_dispatch,
            self.trap_propagation,
            self.distributed_collectives,
            self.max_workgroup_size,
            required,
        )
    }

    fn rejects(self, required: &RequiredCapabilities) -> MissingCapability {
        self.check(required)
            .expect_err("Fix: a backend lacking a required capability must fail the check.")
    }
}

/// Assert `error` names `capability` in its flat missing list and in its
/// rendered message, with the `Fix:` hint every diagnostic carries.
fn assert_reports(error: &MissingCapability, capability: &str) {
    assert!(
        error.missing.iter().any(|item| item == capability),
        "missing list must name `{capability}`: {:?}",
        error.missing
    );
    let message = error.to_string();
    assert!(
        message.contains(capability),
        "display must name `{capability}`: {message}"
    );
    assert!(
        message.contains("Fix:"),
        "display must carry a Fix: hint: {message}"
    );
}

// ---------------------------------------------------------------------------
// scan: what a program requires
// ---------------------------------------------------------------------------

#[test]
fn subgroup_reduction_requires_subgroup_ops() {
    let program = scalar_program(vec![Node::let_bind(
        "s",
        Expr::subgroup_add(Expr::u32(1)),
    )]);
    assert!(
        scan(&program).subgroup_ops,
        "subgroup_add must set subgroup_ops"
    );
}

#[test]
fn subgroup_builtin_expressions_require_subgroup_ops() {
    for expr in [Expr::subgroup_local_id(), Expr::subgroup_size()] {
        let program = scalar_program(vec![Node::store("out", Expr::u32(0), expr)]);
        assert!(
            scan(&program).subgroup_ops,
            "subgroup builtin expressions must set subgroup_ops"
        );
    }
}

#[test]
fn call_to_subgroup_intrinsic_requires_subgroup_ops() {
    let program = scalar_program(vec![Node::let_bind(
        "s",
        Expr::call(
            "vyre-primitives::hardware::subgroup_add",
            vec![Expr::u32(1)],
        ),
    )]);
    assert!(scan(&program).subgroup_ops);
}

#[test]
fn indirect_dispatch_node_requires_indirect_dispatch() {
    let program = Program::wrapped(
        vec![BufferDecl::read("counts", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::indirect_dispatch("counts", 0)],
    );
    assert!(scan(&program).indirect_dispatch);
}

#[test]
fn trap_node_requires_trap_propagation() {
    let program = scalar_program(vec![Node::trap(Expr::u32(0), "fault")]);
    assert!(scan(&program).trap);
}

#[test]
fn async_transfer_requires_async_dispatch() {
    let program = scalar_program(vec![Node::async_load("tag")]);
    assert!(scan(&program).async_dispatch);
}

#[test]
fn buffer_element_type_requires_its_own_capability() {
    let cases: [(&str, DataType, fn(&RequiredCapabilities) -> bool); 4] = [
        ("tensor", DataType::Tensor, |caps| caps.tensor_ops),
        ("f64", DataType::F64, |caps| caps.f64),
        ("f16", DataType::F16, |caps| caps.f16),
        ("bf16", DataType::BF16, |caps| caps.bf16),
    ];
    for (element, field, read) in cases {
        let program = element_program(field.clone(), Vec::new());
        assert!(
            read(&scan(&program)),
            "a `{field}` buffer element must set the {element} capability"
        );
        let scalar = element_program(DataType::U32, Vec::new());
        assert!(
            !read(&scan(&scalar)),
            "a u32-only program must not set the {element} capability"
        );
    }
}

#[test]
fn scalar_program_requires_nothing_beyond_its_workgroup() {
    let required = scan(&scalar_program(vec![Node::let_bind("x", Expr::u32(0))]));
    assert!(!required.subgroup_ops);
    assert!(!required.f16);
    assert!(!required.bf16);
    assert!(!required.f64);
    assert!(!required.tensor_ops);
    assert!(!required.async_dispatch);
    assert!(!required.indirect_dispatch);
    assert!(!required.trap);
    assert!(!required.distributed_collectives);
    assert_eq!(required.local_single_rank_collectives, 0);
    assert_eq!(required.transport_collectives, 0);
    assert_eq!(required.max_workgroup_size, [1, 1, 1]);
}

// ---------------------------------------------------------------------------
// scan: collective transport shape
// ---------------------------------------------------------------------------

#[test]
fn world_group_all_gather_lowers_locally_without_transport() {
    let program = collective_program(vec![Node::AllGather {
        input: "input".into(),
        output: "out".into(),
        group: CommGroup::WORLD,
    }]);
    let caps = scan(&program);
    assert!(!caps.distributed_collectives);
    assert_eq!(caps.local_single_rank_collectives, 1);
    assert_eq!(caps.transport_collectives, 0);
}

#[test]
fn broadcast_from_a_nonzero_root_requires_transport() {
    let program = collective_program(vec![Node::Broadcast {
        buffer: "out".into(),
        root: 1,
        group: CommGroup::WORLD,
    }]);
    assert!(scan(&program).distributed_collectives);
}

#[test]
fn mixed_collectives_report_each_side_of_the_transport_split() {
    let program = collective_program(vec![Node::Block(vec![
        Node::AllGather {
            input: "input".into(),
            output: "out".into(),
            group: CommGroup::WORLD,
        },
        Node::Broadcast {
            buffer: "out".into(),
            root: 3,
            group: CommGroup::WORLD,
        },
        Node::ReduceScatter {
            input: "input".into(),
            output: "out".into(),
            op: CollectiveOp::Sum,
            group: CommGroup(9),
        },
    ])]);

    let caps = scan(&program);

    assert!(caps.distributed_collectives);
    assert_eq!(caps.local_single_rank_collectives, 1);
    assert_eq!(caps.transport_collectives, 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn collective_counts_match_the_generated_transport_shape(
        local_count in 0usize..32,
        transport_count in 0usize..32,
    ) {
        let mut nodes = Vec::with_capacity(local_count + transport_count);
        for _ in 0..local_count {
            nodes.push(Node::AllGather {
                input: "input".into(),
                output: "out".into(),
                group: CommGroup::WORLD,
            });
        }
        for root in 1..=transport_count {
            nodes.push(Node::Broadcast {
                buffer: "out".into(),
                root: root as u32,
                group: CommGroup::WORLD,
            });
        }

        let caps = scan(&collective_program(nodes));

        prop_assert_eq!(caps.local_single_rank_collectives, local_count);
        prop_assert_eq!(caps.transport_collectives, transport_count);
        prop_assert_eq!(caps.distributed_collectives, transport_count != 0);
    }
}

// ---------------------------------------------------------------------------
// check_backend_capabilities: what a backend must advertise
// ---------------------------------------------------------------------------

#[test]
fn every_missing_bit_is_reported_in_one_error() {
    let mut required = RequiredCapabilities::none();
    required.subgroup_ops = true;
    required.f16 = true;
    required.trap = true;
    let error = Advertised::none().rejects(&required);
    assert_eq!(error.backend, "test_backend");
    assert_reports(&error, "subgroup_ops");
    assert_reports(&error, "f16");
    assert_reports(&error, "trap_propagation");
}

#[test]
fn missing_collective_transport_reports_the_transport_shape() {
    let mut required = RequiredCapabilities::none();
    required.distributed_collectives = true;
    required.local_single_rank_collectives = 5;
    required.transport_collectives = 8;

    let advertised = Advertised {
        distributed_collectives: false,
        ..Advertised::all()
    };
    let error = advertised.rejects(&required);

    assert_reports(&error, "distributed_collectives");
    assert!(error
        .missing
        .iter()
        .any(|item| item.contains("transport_collectives=8")));
    assert!(error
        .missing
        .iter()
        .any(|item| item.contains("local_single_rank_collectives=5")));
}

#[test]
fn a_workgroup_wider_than_the_backend_allows_is_reported_per_axis() {
    let mut required = RequiredCapabilities::none();
    required.max_workgroup_size = [256, 1, 1];
    let error = Advertised::none()
        .with_max_workgroup_size([128, 1, 1])
        .rejects(&required);
    assert_reports(&error, "workgroup_size");
    assert!(
        error
            .missing
            .iter()
            .any(|item| item == "workgroup_size axis 0 (requested 256, max 128)"),
        "the axis detail must name the axis and both bounds: {:?}",
        error.missing
    );
}

#[test]
fn a_zero_backend_workgroup_size_means_unlimited() {
    let mut required = RequiredCapabilities::none();
    required.max_workgroup_size = [256, 1, 1];
    assert_eq!(
        Advertised::none()
            .with_max_workgroup_size([0, 0, 0])
            .check(&required),
        Ok(()),
        "zero backend workgroup size must mean unlimited"
    );
}

#[test]
fn a_fully_capable_backend_accepts_every_requirement() {
    let mut required = RequiredCapabilities::none();
    required.subgroup_ops = true;
    required.f16 = true;
    required.bf16 = true;
    required.indirect_dispatch = true;
    required.trap = true;
    required.distributed_collectives = true;
    required.max_workgroup_size = [64, 1, 1];
    assert_eq!(
        Advertised::all().check(&required),
        Ok(()),
        "fully supported backend must pass"
    );
}

#[test]
fn missing_capability_is_a_leaf_std_error() {
    let error = MissingCapability {
        backend: "foo".into(),
        missing: vec!["bar".to_string()],
    };
    let dyn_error: &(dyn std::error::Error) = &error;
    assert!(dyn_error.source().is_none());
    let message = dyn_error.to_string();
    assert!(message.contains("foo"));
    assert!(message.contains("bar"));
}

// ---------------------------------------------------------------------------
// The gate: both surfaces enumerated from the checked-in public-API snapshot
// ---------------------------------------------------------------------------

/// One `RequiredCapabilities` boolean field, and the program that makes `scan`
/// set it.
struct ScanCase {
    field: &'static str,
    read: fn(&RequiredCapabilities) -> bool,
    program: fn() -> Program,
}

/// Every boolean field of `RequiredCapabilities`, with a program that requires
/// it. Compared against the snapshot below, so a new boolean capability that
/// nothing here names turns the suite red.
fn scan_cases() -> Vec<ScanCase> {
    vec![
        ScanCase {
            field: "subgroup_ops",
            read: |caps| caps.subgroup_ops,
            program: || scalar_program(vec![Node::let_bind("s", Expr::subgroup_add(Expr::u32(1)))]),
        },
        ScanCase {
            field: "f16",
            read: |caps| caps.f16,
            program: || element_program(DataType::F16, Vec::new()),
        },
        ScanCase {
            field: "bf16",
            read: |caps| caps.bf16,
            program: || element_program(DataType::BF16, Vec::new()),
        },
        ScanCase {
            field: "f64",
            read: |caps| caps.f64,
            program: || element_program(DataType::F64, Vec::new()),
        },
        ScanCase {
            field: "tensor_ops",
            read: |caps| caps.tensor_ops,
            program: || element_program(DataType::Tensor, Vec::new()),
        },
        ScanCase {
            field: "async_dispatch",
            read: |caps| caps.async_dispatch,
            program: || scalar_program(vec![Node::async_load("tag")]),
        },
        ScanCase {
            field: "indirect_dispatch",
            read: |caps| caps.indirect_dispatch,
            program: || {
                Program::wrapped(
                    vec![BufferDecl::read("counts", 0, DataType::U32).with_count(1)],
                    [1, 1, 1],
                    vec![Node::indirect_dispatch("counts", 0)],
                )
            },
        },
        ScanCase {
            field: "trap",
            read: |caps| caps.trap,
            program: || scalar_program(vec![Node::trap(Expr::u32(0), "fault")]),
        },
        ScanCase {
            field: "distributed_collectives",
            read: |caps| caps.distributed_collectives,
            program: || {
                collective_program(vec![Node::Broadcast {
                    buffer: "out".into(),
                    root: 1,
                    group: CommGroup::WORLD,
                }])
            },
        },
    ]
}

/// One advertised-capability parameter of `check_backend_capabilities`, the
/// requirement it gates, and the name it pushes into `MissingCapability`.
struct EnforcementCase {
    parameter: &'static str,
    reported_as: &'static str,
    require: fn(&mut RequiredCapabilities),
    withhold: fn(&mut Advertised),
}

/// Every `supports_*` parameter of `check_backend_capabilities`. Compared
/// against the snapshot below, so a parameter added and then never read - the
/// shape this workspace regresses into - turns the suite red instead of being
/// silently ignored.
fn enforcement_cases() -> Vec<EnforcementCase> {
    vec![
        EnforcementCase {
            parameter: "supports_subgroup_ops",
            reported_as: "subgroup_ops",
            require: |required| required.subgroup_ops = true,
            withhold: |advertised| advertised.subgroup_ops = false,
        },
        EnforcementCase {
            parameter: "supports_half_precision",
            reported_as: "f16",
            require: |required| required.f16 = true,
            withhold: |advertised| advertised.half_precision = false,
        },
        EnforcementCase {
            parameter: "supports_brain_float",
            reported_as: "bf16",
            require: |required| required.bf16 = true,
            withhold: |advertised| advertised.brain_float = false,
        },
        EnforcementCase {
            parameter: "supports_indirect_dispatch",
            reported_as: "indirect_dispatch",
            require: |required| required.indirect_dispatch = true,
            withhold: |advertised| advertised.indirect_dispatch = false,
        },
        EnforcementCase {
            parameter: "supports_trap_propagation",
            reported_as: "trap_propagation",
            require: |required| required.trap = true,
            withhold: |advertised| advertised.trap_propagation = false,
        },
        EnforcementCase {
            parameter: "supports_distributed_collectives",
            reported_as: "distributed_collectives",
            require: |required| required.distributed_collectives = true,
            withhold: |advertised| advertised.distributed_collectives = false,
        },
    ]
}

/// The path of the checked-in public-API snapshot for this crate.
///
/// The snapshot is regenerated from rustdoc by
/// `scripts/check_public_api_snapshot.sh` and a byte-stability gate keeps it
/// equal to the crate's real surface, so it is the one place a new capability
/// field or a new advertised parameter is guaranteed to appear.
fn api_snapshot() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .map(|directory| directory.join("docs/public-api/vyre-foundation.txt"))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| {
            panic!(
                "Fix: no docs/public-api/vyre-foundation.txt above {}. The capability gate enumerates the capability surface from that snapshot.",
                manifest.display()
            )
        })
}

fn snapshot_text() -> String {
    let path = api_snapshot();
    std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: the public-API snapshot at {} must be readable: {error}",
            path.display()
        )
    })
}

/// Boolean field names of `RequiredCapabilities`, from the snapshot.
fn snapshot_boolean_capability_fields() -> BTreeSet<String> {
    let prefix = "pub vyre_foundation::program_caps::RequiredCapabilities::";
    let fields: BTreeSet<String> = snapshot_text()
        .lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .filter_map(|rest| rest.strip_suffix(": bool"))
        .map(str::to_string)
        .collect();
    assert!(
        !fields.is_empty(),
        "Fix: the public-API snapshot lists no boolean `RequiredCapabilities` fields. Refresh it with scripts/check_public_api_snapshot.sh --refresh vyre-foundation."
    );
    fields
}

/// Advertised-capability parameter names of `check_backend_capabilities`, from
/// the snapshot's signature line.
fn snapshot_advertised_parameters() -> BTreeSet<String> {
    let signature = snapshot_text()
        .lines()
        .find(|line| line.starts_with("pub fn vyre_foundation::program_caps::check_backend_capabilities("))
        .map(str::to_string)
        .unwrap_or_else(|| {
            panic!(
                "Fix: the public-API snapshot has no `check_backend_capabilities` signature. Refresh it with scripts/check_public_api_snapshot.sh --refresh vyre-foundation."
            )
        });
    let parameters: BTreeSet<String> = signature
        .split(&[',', '(', ')'][..])
        .map(str::trim)
        .filter_map(|argument| argument.strip_suffix(": bool"))
        .filter(|name| name.starts_with("supports_"))
        .map(str::to_string)
        .collect();
    assert!(
        !parameters.is_empty(),
        "Fix: no `supports_*: bool` parameters parsed out of `{signature}`."
    );
    parameters
}

#[test]
fn every_boolean_capability_field_has_a_scan_case() {
    let named: BTreeSet<String> = scan_cases()
        .iter()
        .map(|case| case.field.to_string())
        .collect();
    assert_eq!(
        snapshot_boolean_capability_fields(),
        named,
        "Fix: `scan_cases` in this file must name exactly the boolean `RequiredCapabilities` fields the public-API snapshot records. Add the new capability with a program that requires it, and decide whether `check_backend_capabilities` must gate it."
    );
}

#[test]
fn every_scan_case_is_required_by_its_program_and_not_by_a_scalar_one() {
    let scalar = scan(&scalar_program(vec![Node::let_bind("x", Expr::u32(0))]));
    for case in scan_cases() {
        let required = scan(&(case.program)());
        assert!(
            (case.read)(&required),
            "Fix: the `{}` program must make `scan` set `{}`.",
            case.field,
            case.field
        );
        assert!(
            !(case.read)(&scalar),
            "Fix: a u32 scalar program must not set `{}`; the case proves nothing otherwise.",
            case.field
        );
    }
}

#[test]
fn every_advertised_parameter_has_an_enforcement_case() {
    let named: BTreeSet<String> = enforcement_cases()
        .iter()
        .map(|case| case.parameter.to_string())
        .collect();
    assert_eq!(
        snapshot_advertised_parameters(),
        named,
        "Fix: `enforcement_cases` in this file must name exactly the `supports_*` parameters of `check_backend_capabilities`. A parameter with no case is a capability the function accepts and may never read."
    );
}

#[test]
fn withholding_an_advertised_capability_rejects_the_program_that_needs_it() {
    for case in enforcement_cases() {
        let mut required = RequiredCapabilities::none();
        (case.require)(&mut required);

        assert_eq!(
            Advertised::all().check(&required),
            Ok(()),
            "Fix: a backend advertising everything must accept a program requiring only `{}`.",
            case.parameter
        );

        let mut advertised = Advertised::all();
        (case.withhold)(&mut advertised);
        let Err(error) = advertised.check(&required) else {
            panic!(
                "Fix: `check_backend_capabilities` accepted a program requiring `{}` from a backend that withholds `{}`. The parameter is not read.",
                case.reported_as, case.parameter
            );
        };
        assert_reports(&error, case.reported_as);
    }
}

// ---------------------------------------------------------------------------
// RequiredCapabilities algebra
// ---------------------------------------------------------------------------

#[test]
fn union_is_fieldwise_or_max_and_sum() {
    let mut left = RequiredCapabilities::none();
    left.subgroup_ops = true;
    left.f16 = true;
    left.max_workgroup_size = [64, 1, 1];
    left.static_storage_bytes = 100;
    left.local_single_rank_collectives = 2;
    left.transport_collectives = 3;

    let mut right = RequiredCapabilities::none();
    right.bf16 = true;
    right.max_workgroup_size = [32, 2, 1];
    right.static_storage_bytes = 50;
    right.local_single_rank_collectives = 4;
    right.transport_collectives = 5;

    let union = left.union(right);
    assert!(union.subgroup_ops);
    assert!(union.f16);
    assert!(union.bf16);
    assert_eq!(union.max_workgroup_size, [64, 2, 1]);
    assert_eq!(union.static_storage_bytes, 150);
    assert_eq!(union.local_single_rank_collectives, 6);
    assert_eq!(union.transport_collectives, 8);
}
