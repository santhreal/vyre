//! Runtime routing contract tests.

use vyre_foundation::execution_plan::plan;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_runtime::routing::standard_policy::StandardPolicy;
use vyre_runtime::routing::{RoutingDecision, RoutingEngine};

fn program_with_nodes(node_count: u32, output_count: u32) -> Program {
    let body = (0..node_count)
        .map(|idx| Node::store("out", Expr::u32(idx), Expr::u32(idx)))
        .collect::<Vec<_>>();
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(output_count)],
        [128, 1, 1],
        body,
    )
}

#[test]
fn standard_runtime_routing_is_megakernel_first() {
    let engine = RoutingEngine::new(StandardPolicy);
    for (node_count, output_count) in [(64, 64), (64, 16_384), (65, 65), (1025, 1025)] {
        let plan = plan(&program_with_nodes(node_count, output_count))
            .expect("routing fixture must be canonical");
        assert_eq!(engine.route(&plan), RoutingDecision::PersistentMegakernel);
    }
}

/// The router declares no host-execution route, and neither does its prose.
///
/// # Why this shape
///
/// The assertion this replaces was `assert_ne!(engine.route(&plan),
/// RoutingDecision::CpuSimd)`: it proved one plan did not pick the CPU route
/// while leaving the route declared, reachable by name, and documented as an
/// opt-in. Vyre executes compute on a device. The only host arithmetic in the
/// workspace is `vyre-reference`, which is a parity oracle and is not a route.
/// So the contract is that no such route exists, and the only place that can be
/// asserted is the declaration.
///
/// The variant set and the module text are read from source at run time, so a
/// route added back under any spelling turns this RED on arrival rather than
/// after somebody remembers to extend a hardcoded list.
///
/// # What it does not catch
///
/// A device route whose implementation runs on the host. That is a property of
/// the executor arm, not of the router, and belongs to the backend that serves
/// the decision.
#[test]
fn the_router_declares_no_host_execution_route() {
    let routing = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/routing");
    let mod_source = std::fs::read_to_string(routing.join("mod.rs"))
        .expect("Fix: vyre-runtime/src/routing/mod.rs must be readable by its own contract test");
    let policy_source = std::fs::read_to_string(routing.join("standard_policy.rs")).expect(
        "Fix: vyre-runtime/src/routing/standard_policy.rs must be readable by its own contract test",
    );

    let body = vyre_test_support::braced_body(&mod_source, "pub enum RoutingDecision {").expect(
        "Fix: no `pub enum RoutingDecision` in vyre-runtime/src/routing/mod.rs; this gate is \
         reading the wrong file",
    );
    let declared: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//") && !line.starts_with('#'))
        .map(|line| line.trim_end_matches(','))
        .collect();

    assert!(
        declared.len() >= 2,
        "Fix: the RoutingDecision scan found {declared:?}; it is reading the wrong region rather \
         than looking at a one-variant enum"
    );
    assert_eq!(
        declared,
        vec!["GpuPipeline", "PersistentMegakernel"],
        "Fix: a route was added or renamed. Record what serves it and why it is not a host \
         execution path, then update this gate; an unserved route tells callers a degradation \
         path exists."
    );

    for (file, source) in [("mod.rs", &mod_source), ("standard_policy.rs", &policy_source)] {
        for forbidden in ["CpuSimd", "cpu_fast_path", "CPU SIMD", "cpu_simd"] {
            assert!(
                !source.contains(forbidden),
                "Fix: vyre-runtime/src/routing/{file} names `{forbidden}`. There is no CPU \
                 execution route; delete the route rather than documenting one."
            );
        }
    }
}
