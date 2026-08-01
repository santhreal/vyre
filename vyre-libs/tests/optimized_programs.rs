//! Shape tests for optimized Cat-A programs.

#[cfg(feature = "nn-attention")]
use vyre::ir::Expr;
use vyre::ir::{MemoryKind, Node};

#[cfg(feature = "nn-linear")]
#[test]
fn linear_tiled_uses_tiled_matmul_kernel_shape() {
    let program = vyre_libs::nn::linear_tiled("x", "w", "b", "out", 37, 65, 16)
        .expect("Fix: optimized linear_tiled must build for positive dimensions.");

    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    assert!(
        program
            .buffers()
            .iter()
            .any(|buffer| buffer.name.as_ref() == "w" && buffer.count == 37 * 65),
        "Fix: linear_tiled must preserve the matmul weight buffer contract."
    );
    assert_region_generator(&program, "vyre-libs::nn::linear_tiled");
}

/// The default attention kernel maps one WORKGROUP, not one invocation, to a
/// query row.
///
/// Two earlier shapes are ruled out here, and the test has been wrong about
/// which one is current before, so both are asserted explicitly:
///
///   - the original kernel bound one invocation per OUTPUT ELEMENT (`idx`),
///     which serializes the dot product across `d` and leaves the lanes idle;
///   - the intermediate kernel bound one invocation per query row (`i`), which
///     is what this test used to assert. It was replaced because a whole row
///     still runs on a single lane.
///
/// The current kernel is workgroup-cooperative: `group` selects a row-block,
/// `lane` is the invocation within it, and `row` is derived from `group`. That
/// is why there is no top-level `InvocationId` binding at all, and asserting
/// for one made this test fail against a kernel that is strictly better than
/// the one it was written for.
#[cfg(feature = "nn-attention")]
#[test]
fn attention_default_maps_one_workgroup_to_a_query_row() {
    // s = 16 clears the `s <= 8 && d <= 16` direct-unroll threshold, which
    // builds a fully unrolled straight-line program with a [1, 1, 1] workgroup
    // and no lane structure at all.
    let program = vyre_libs::nn::attention("q", "k", "v", "out", 16, 4);

    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    let body = root_region_body(&program);

    // The kernel body sits inside `if WorkgroupId.x < total_groups { ... }`,
    // the guard that retires workgroups past the last row-block, so the
    // bindings are one level down rather than at the top of the region.
    assert!(
        matches!(body, [Node::If { .. }]),
        "Fix: attention must guard its whole body on the workgroup bound, got {body:?}"
    );

    let binding = |name: &str| -> Option<&Expr> { find_let(body, name) };

    assert!(
        matches!(binding("group"), Some(Expr::WorkgroupId { axis: 0 })),
        "Fix: attention must bind `group` to the x workgroup id, got {:?}",
        binding("group")
    );
    assert!(
        matches!(binding("lane"), Some(Expr::LocalId { axis: 0 })),
        "Fix: attention must bind `lane` to the x local id, got {:?}",
        binding("lane")
    );
    assert!(
        binding("row").is_some(),
        "Fix: attention must derive a query row from the workgroup id"
    );
    assert_eq!(
        count_invocation_id_lets(body, "idx"),
        0,
        "Fix: attention must not use the old one-invocation-per-output-element idx kernel."
    );
    assert_eq!(
        count_invocation_id_lets(body, "i"),
        0,
        "Fix: attention must not fall back to the one-invocation-per-row kernel."
    );
}

#[cfg(feature = "nn-attention")]
#[test]
fn softmax_default_uses_tiled_workgroup_scratch() {
    let program = vyre_libs::nn::softmax("input", "output", 513);

    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    assert!(
        program.buffers().iter().any(|buffer| {
            buffer.name.as_ref() == "softmax_scratch"
                && buffer.kind == MemoryKind::Shared
                && buffer.count == 256
        }),
        "Fix: optimized softmax must keep its tiled workgroup scratch buffer."
    );
}

#[cfg(feature = "nn-norm")]
#[test]
fn rms_norm_default_uses_tiled_workgroup_scratch() {
    let program = vyre_libs::nn::rms_norm("input", "output", 777, 1.0e-5);

    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    assert!(
        program.buffers().iter().any(|buffer| {
            buffer.name.as_ref() == "rms_scratch"
                && buffer.kind == MemoryKind::Shared
                && buffer.count == 256
        }),
        "Fix: optimized rms_norm must keep its tiled workgroup scratch buffer."
    );
}

fn assert_region_generator(program: &vyre::ir::Program, expected: &str) {
    match &program.entry()[0] {
        Node::Region { generator, .. } => assert_eq!(generator.as_str(), expected),
        other => panic!("Fix: expected optimized root Region, got {other:?}"),
    }
}

#[cfg(feature = "nn-attention")]
fn root_region_body(program: &vyre::ir::Program) -> &[Node] {
    match &program.entry()[0] {
        Node::Region { body, .. } => body.as_ref(),
        other => panic!("Fix: expected optimized root Region, got {other:?}"),
    }
}

/// Find the first `Let` binding of `name` anywhere in a node tree.
///
/// Bindings are not always at the top of a region: the tiled kernels wrap
/// their whole body in a workgroup-bounds guard, so a flat search over the
/// region body finds nothing and makes a correct kernel look broken.
#[cfg(feature = "nn-attention")]
fn find_let<'a>(nodes: &'a [Node], name: &str) -> Option<&'a Expr> {
    for node in nodes {
        let found = match node {
            Node::Let { name: bound, value } if bound.as_str() == name => return Some(value),
            Node::If {
                then, otherwise, ..
            } => find_let(then, name).or_else(|| find_let(otherwise, name)),
            Node::Loop { body, .. } | Node::Block(body) => find_let(body, name),
            Node::Region { body, .. } => find_let(body.as_slice(), name),
            _ => None,
        };
        if found.is_some() {
            return found;
        }
    }
    None
}

#[cfg(feature = "nn-attention")]
fn count_invocation_id_lets(nodes: &[Node], name: &str) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::Let {
                name: let_name,
                value: Expr::InvocationId { .. },
            } if let_name.as_str() == name => 1,
            Node::If {
                then, otherwise, ..
            } => count_invocation_id_lets(then, name) + count_invocation_id_lets(otherwise, name),
            Node::Loop { body, .. } | Node::Block(body) => count_invocation_id_lets(body, name),
            Node::Region { body, .. } => count_invocation_id_lets(body, name),
            _ => 0,
        })
        .sum()
}
