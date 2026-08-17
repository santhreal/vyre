//! Shared helpers used by the per-op Cat-A builders.
//!
//! Each op in `vyre-libs` ships a chainable builder that:
//!
//! 1. Accepts [`TensorRef`]s instead of bare `&str` buffer names, so
//!    dtype + shape mismatches fail at `build()` time.
//! 2. Checks every pair of buffer names is unique.
//! 3. Verifies every [`TensorRef`]'s dtype against the op's expected dtype.
//! 4. Verifies element-count overflow.
//! 5. Allows chained overrides (workgroup size, region generator,
//!    tenant id) without churning the function signature  -  extension
//!    fields live inside a `#[non_exhaustive]` options struct so new
//!    knobs never break existing call sites.
//!
//! `BuildOptions` is intentionally small at launch; fields are added
//! rather than removed (the `#[non_exhaustive]` attribute enforces
//! this). Every Cat-A op exposes its builder as `<Op>Builder::new(...)`
//! and delegates defaults through `BuildOptions::default()`.

/// Mapping an index space onto the lanes of one workgroup.
///
/// Behind `reduce` because the argmax collapses its lane partials with the
/// workgroup reduction children, which that feature owns. Every consumer of a
/// cooperative walk already enables it.
#[cfg(feature = "reduce")]
pub(crate) mod cooperative;
#[cfg(feature = "graph")]
pub mod csr;
pub mod elementwise;
use elementwise::ElementwiseComposer;
/// Canonical matrix multiplication and contraction IR composer.
pub mod gemm;
/// Domain-neutral byte-range ordering predicates over the scanner output
/// contract.
pub mod range_ordering;
pub(crate) mod reduction;
/// Shared table-walking state machine / DFA composer.
pub mod state_machine;
pub(crate) use state_machine::TableStateMachineComposer;
/// The two shared child regions registered as operations in their own right.
///
/// Behind `builder-ops` because `INDEXED_MAP_OP_ID` and
/// `STRIDED_ACCUMULATE_OP_ID` are catalog entries, and a catalog entry is
/// enabled by a feature. The skeletons themselves stay ungated: a dialect
/// composes them without asking for their registrations.
#[cfg(feature = "builder-ops")]
mod registrations;
/// Canonical 2D grid, coordinate decomposition, stencil, and pixel composer.
pub mod stencil;

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};

/// Shared child region for one-output indexed maps.
///
/// This is the kernel skeleton behind embedding lookup, byte shuffles,
/// quant pack/unpack, and similar data-layout transforms:
/// `for i in 0..n { out[dst(i)] = value(i) }`.
pub(crate) const INDEXED_MAP_OP_ID: &str = "vyre-libs::builder::indexed_map";
/// Shared child region for strided per-lane workgroup accumulators.
pub(crate) const STRIDED_ACCUMULATE_OP_ID: &str = "vyre-libs::builder::strided_accumulate";
/// Shared child region for strided writeback after a tiled row reduction.
pub(crate) const STRIDED_WRITEBACK_OP_ID: &str = "anonymous::vyre-libs::builder::strided_writeback";

/// Shared options every Cat-A builder threads through. Lives here so
/// every op agrees on the same surface.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct BuildOptions {
    /// Workgroup size override. `None` = op's canonical default.
    pub workgroup_size: Option<[u32; 3]>,
    /// Region generator override. `None` = op's canonical `"vyre-libs::…"`
    /// identifier. Used when a downstream crate wraps a Cat-A op and
    /// wants its own generator id in conformance certificates.
    pub region_generator: Option<&'static str>,
    /// Tenant id baked into the region metadata for multi-tenant
    /// deployments. Routed through the megakernel's tenant-mask table
    /// when the Program runs inside `vyre-runtime`.
    pub tenant_id: Option<u32>,
}

impl BuildOptions {
    /// Fluent constructor  -  start with defaults and chain overrides.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the workgroup size.
    #[must_use]
    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.workgroup_size = Some(size);
        self
    }

    /// Override the region generator name (must be `&'static str`).
    #[must_use]
    pub fn with_region_generator(mut self, name: &'static str) -> Self {
        self.region_generator = Some(name);
        self
    }

    /// Stamp a tenant id into the Cat-A op's region metadata.
    #[must_use]
    pub fn with_tenant_id(mut self, tenant_id: u32) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }
}

/// Generate the shared Cat-A option surface on a typed builder.
///
/// Gated on `cat-a-builder-options`, which every dialect feature whose module
/// invokes this macro declares in `Cargo.toml`. A build that enables none of
/// them does not compile the macro at all.
#[cfg(any(test, feature = "cat-a-builder-options"))]
macro_rules! impl_cat_a_builder_options {
    ($builder:ident) => {
        impl $builder {
            /// Override the generated Program workgroup size.
            #[must_use]
            pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
                self.options = self.options.with_workgroup_size(size);
                self
            }

            /// Override the Region generator id.
            #[must_use]
            pub fn with_region_generator(mut self, name: &'static str) -> Self {
                self.options = self.options.with_region_generator(name);
                self
            }

            /// Stamp the Region metadata with a tenant id.
            #[must_use]
            pub fn with_tenant_id(mut self, tenant_id: u32) -> Self {
                self.options = self.options.with_tenant_id(tenant_id);
                self
            }
        }
    };
}

#[cfg(any(test, feature = "cat-a-builder-options"))]
pub(crate) use impl_cat_a_builder_options;

/// Validate a slice of `TensorRef`s against an expected `DataType`
/// for each position, plus name-uniqueness across the whole slice.
/// Used by every op's `build()` to consolidate the fanout of checks.
pub fn check_tensors(
    op: &'static str,
    tensors: &[(&TensorRef, DataType)],
) -> Result<(), TensorRefError> {
    // Dtype check per tensor.
    for (r, expected) in tensors {
        crate::plumbing::operand::tensor_ref::check_dtype(r, expected.clone(), op)?;
        if r.element_count().is_none() {
            return Err(TensorRefError::ElementCountOverflow {
                name: r.name.as_str().to_string(),
                shape: r.shape.to_vec(),
            });
        }
    }
    for (idx, (left, _)) in tensors.iter().enumerate() {
        for (right, _) in &tensors[idx + 1..] {
            if left.name_str() == right.name_str() {
                return Err(TensorRefError::NameCollision {
                    name: left.name.as_str().to_string(),
                    op,
                });
            }
        }
    }
    Ok(())
}

/// Reject an output whose shape differs from the input's.
///
/// Every shape-preserving builder in `nn` restated this check verbatim, down to
/// blaming the OUTPUT tensor and reporting the input's shape as `expected`.
/// That attribution is the contract: the input is what the caller asked to
/// transform, so a divergent output is the output's defect.
///
/// It deliberately compares shapes and nothing else. Dtype and name-uniqueness
/// stay with [`check_tensors`], and an op-specific parameter such as an epsilon
/// range stays with the op that owns it, so a caller keeps control of the order
/// its errors surface in.
pub fn check_same_shape(
    op: &'static str,
    input: &TensorRef,
    output: &TensorRef,
) -> Result<(), TensorRefError> {
    if input.shape != output.shape {
        return Err(TensorRefError::ShapeMismatch {
            name: output.name.as_str().to_string(),
            found: output.shape.to_vec(),
            expected: input.shape.to_vec(),
            op,
        });
    }
    Ok(())
}

/// The flattened element count of a tensor a reduction will index, rejecting
/// both an unrepresentable product and an empty tensor.
///
/// The nonzero floor is a contract, not defensive padding: a tiled reduction
/// seeds its accumulator from `load(input, 0)` before the loop bound is known,
/// so an empty tensor is an out-of-bounds read rather than an empty result.
/// Emptiness is reported as a `ShapeMismatch` against `[1]`, which is what the
/// hand-written copies in `nn::softmax` and `nn::layer_norm` both returned.
pub fn checked_element_count(op: &'static str, input: &TensorRef) -> Result<u32, TensorRefError> {
    let n = input
        .element_count()
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: input.name_str().to_string(),
            shape: input.shape.to_vec(),
        })?;
    if n == 0 {
        return Err(TensorRefError::ShapeMismatch {
            name: input.name.as_str().to_string(),
            found: input.shape.to_vec(),
            expected: vec![1],
            op,
        });
    }
    Ok(n)
}

#[cfg(test)]
mod cat_a_builder_option_macro_tests {
    #![allow(unreachable_pub)]

    use super::BuildOptions;

    #[derive(Debug, Clone)]
    struct DemoBuilder {
        options: BuildOptions,
    }

    impl DemoBuilder {
        fn new() -> Self {
            Self {
                options: BuildOptions::default(),
            }
        }
    }

    super::impl_cat_a_builder_options!(DemoBuilder);

    #[test]
    fn generated_option_surface_threads_every_shared_knob() {
        let builder = DemoBuilder::new()
            .with_workgroup_size([8, 4, 2])
            .with_region_generator("custom::generator")
            .with_tenant_id(17);

        assert_eq!(builder.options.workgroup_size, Some([8, 4, 2]));
        assert_eq!(builder.options.region_generator, Some("custom::generator"));
        assert_eq!(builder.options.tenant_id, Some(17));
    }
}

/// Build the canonical one-output indexed-map skeleton.
///
/// Callers provide buffer declarations plus the semantic mapping from logical
/// element `i` to `(dst_index, value)`. The loop, bounds guard, invocation id,
/// workgroup default, and composition region stay centralized.
pub(crate) fn build_indexed_map<F>(
    op_id: &'static str,
    buffers: Vec<BufferDecl>,
    output: &str,
    count: u32,
    workgroup_size: [u32; 3],
    f: F,
) -> Program
where
    F: FnOnce(Expr) -> (Expr, Expr),
{
    let i = Expr::var("i");
    let (dst_index, value) = f(i.clone());
    let child_body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i, Expr::u32(count)),
            vec![Node::store(output, dst_index, value)],
        ),
    ];
    let parent = Ident::from(op_id);

    Program::wrapped(
        buffers,
        workgroup_size,
        vec![wrap_anonymous_region(
            op_id,
            vec![wrap_child_region(INDEXED_MAP_OP_ID, parent, child_body)],
        )],
    )
}

/// Build a shared strided single-accumulator child region.
///
/// The parent must bind `local = LocalId(0)` before this child. The child
/// accumulates `i = chunk * tile + local` for `chunk in 0..chunks`, guards
/// `i < n`, and stores the lane-local accumulator into `scratch[local]`.
pub(crate) fn strided_accumulate_child<F>(
    parent_op_id: &'static str,
    tile: u32,
    chunks: u32,
    n: u32,
    acc_name: &'static str,
    initial: Expr,
    scratch: &'static str,
    step: F,
) -> Node
where
    F: Fn(Expr, Expr) -> Expr,
{
    let local = Expr::var("local");
    let idx = Expr::var("idx");
    let acc = Expr::var(acc_name);
    let child_body = vec![Node::if_then(
        Expr::is_first_workgroup(),
        vec![
            Node::let_bind(acc_name, initial),
            strided_loop(
                tile,
                chunks,
                n,
                vec![Node::assign(acc_name, step(idx, acc))],
            ),
            Node::store(scratch, local, Expr::var(acc_name)),
        ],
    )];

    child_region(parent_op_id, STRIDED_ACCUMULATE_OP_ID, child_body)
}

/// Build a shared strided dual-accumulator child region.
///
/// This keeps paired reductions such as `(sum, sum_sq)` in one memory pass
/// instead of forcing two separate scans over the input.
pub(crate) fn strided_accumulate2_child<F1, F2>(
    parent_op_id: &'static str,
    tile: u32,
    chunks: u32,
    n: u32,
    first: (&'static str, Expr, &'static str, F1),
    second: (&'static str, Expr, &'static str, F2),
) -> Node
where
    F1: Fn(Expr, Expr) -> Expr,
    F2: Fn(Expr, Expr) -> Expr,
{
    let (first_name, first_initial, first_scratch, first_step) = first;
    let (second_name, second_initial, second_scratch, second_step) = second;
    let local = Expr::var("local");
    let idx = Expr::var("idx");
    let child_body = vec![Node::if_then(
        Expr::is_first_workgroup(),
        vec![
            Node::let_bind(first_name, first_initial),
            Node::let_bind(second_name, second_initial),
            strided_loop(
                tile,
                chunks,
                n,
                vec![
                    Node::assign(first_name, first_step(idx.clone(), Expr::var(first_name))),
                    Node::assign(second_name, second_step(idx, Expr::var(second_name))),
                ],
            ),
            Node::store(first_scratch, local.clone(), Expr::var(first_name)),
            Node::store(second_scratch, local, Expr::var(second_name)),
        ],
    )];

    child_region(parent_op_id, STRIDED_ACCUMULATE_OP_ID, child_body)
}

/// Build a shared strided writeback child region.
///
/// The parent must bind `local = LocalId(0)` before this child. Optional
/// `prelude` nodes run once in workgroup zero before the strided write loop,
/// which lets row reductions load reduced scalars exactly once per lane.
pub(crate) fn strided_writeback_child<F>(
    parent_op_id: &'static str,
    tile: u32,
    chunks: u32,
    n: u32,
    output: &str,
    prelude: Vec<Node>,
    value: F,
) -> Node
where
    F: Fn(Expr) -> Expr,
{
    let idx = Expr::var("idx");
    let mut guarded = prelude;
    guarded.push(strided_loop(
        tile,
        chunks,
        n,
        vec![Node::store(output, idx.clone(), value(idx))],
    ));
    child_region(
        parent_op_id,
        STRIDED_WRITEBACK_OP_ID,
        vec![Node::if_then(Expr::is_first_workgroup(), guarded)],
    )
}

fn strided_loop(tile: u32, chunks: u32, n: u32, guarded_body: Vec<Node>) -> Node {
    Node::loop_for(
        "chunk",
        Expr::u32(0),
        Expr::u32(chunks),
        vec![
            Node::let_bind(
                "idx",
                Expr::add(
                    Expr::mul(Expr::var("chunk"), Expr::u32(tile)),
                    Expr::var("local"),
                ),
            ),
            Node::if_then(Expr::lt(Expr::var("idx"), Expr::u32(n)), guarded_body),
        ],
    )
}

fn child_region(parent_op_id: &'static str, child_op_id: &'static str, body: Vec<Node>) -> Node {
    wrap_child_region(child_op_id, Ident::from(parent_op_id), body)
}

/// Tensor-ref elementwise binary builder, used by `math::avg_floor`,
/// `math::algebra`, and other binary-arithmetic primitives.
pub(crate) fn build_elementwise_binary<F>(
    op_id: &'static str,
    a: crate::plumbing::operand::tensor_ref::TensorRef,
    b: crate::plumbing::operand::tensor_ref::TensorRef,
    out: crate::plumbing::operand::tensor_ref::TensorRef,
    options: BuildOptions,
    f: F,
) -> Result<vyre_foundation::ir::Program, crate::plumbing::operand::tensor_ref::TensorRefError>
where
    F: Fn(vyre_foundation::ir::Expr, vyre_foundation::ir::Expr) -> vyre_foundation::ir::Expr,
{
    ElementwiseComposer::try_binary(op_id, a, b, out, options, f)
}

pub(crate) fn build_elementwise_unary<F>(
    op_id: &'static str,
    a: crate::plumbing::operand::tensor_ref::TensorRef,
    out: crate::plumbing::operand::tensor_ref::TensorRef,
    options: BuildOptions,
    f: F,
) -> Result<vyre_foundation::ir::Program, crate::plumbing::operand::tensor_ref::TensorRefError>
where
    F: Fn(vyre_foundation::ir::Expr) -> vyre_foundation::ir::Expr,
{
    ElementwiseComposer::try_unary(op_id, a, out, options, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_options_defaults_are_all_none() {
        let o = BuildOptions::default();
        assert!(o.workgroup_size.is_none());
        assert!(o.region_generator.is_none());
        assert!(o.tenant_id.is_none());
    }

    #[test]
    fn build_options_chain_preserves_earlier_setters() {
        let o = BuildOptions::new()
            .with_workgroup_size([128, 1, 1])
            .with_region_generator("test::op")
            .with_tenant_id(7);
        assert_eq!(o.workgroup_size, Some([128, 1, 1]));
        assert_eq!(o.region_generator, Some("test::op"));
        assert_eq!(o.tenant_id, Some(7));
    }

    #[test]
    fn check_tensors_passes_on_clean_inputs() {
        let a = TensorRef::u32_1d("a", 4);
        let b = TensorRef::u32_1d("b", 4);
        assert!(matches!(
            check_tensors("op", &[(&a, DataType::U32), (&b, DataType::U32)]),
            Ok(())
        ));
    }

    #[test]
    fn check_tensors_catches_dtype_mismatch() {
        let a = TensorRef::u32_1d("a", 4);
        let err = check_tensors("op", &[(&a, DataType::F32)]).unwrap_err();
        assert!(matches!(err, TensorRefError::DtypeMismatch { .. }));
    }

    #[test]
    fn check_tensors_catches_overflow() {
        let a = TensorRef::new("big", DataType::U32, vec![1u32 << 20, 1u32 << 20]);
        let err = check_tensors("op", &[(&a, DataType::U32)]).unwrap_err();
        assert!(matches!(err, TensorRefError::ElementCountOverflow { .. }));
    }

    #[test]
    fn check_tensors_catches_name_collision() {
        let a = TensorRef::u32_1d("x", 4);
        let b = TensorRef::u32_1d("x", 4);
        let err = check_tensors("op", &[(&a, DataType::U32), (&b, DataType::U32)]).unwrap_err();
        assert!(matches!(err, TensorRefError::NameCollision { .. }));
    }

    #[test]
    fn indexed_map_builder_emits_shared_child_region() {
        let program = build_indexed_map(
            "vyre-libs::test::indexed_map_user",
            vec![
                BufferDecl::storage(
                    "input",
                    0,
                    vyre_foundation::ir::BufferAccess::ReadOnly,
                    DataType::U32,
                )
                .with_count(4),
                BufferDecl::output("output", 1, DataType::U32).with_count(4),
            ],
            "output",
            4,
            [64, 1, 1],
            |i| (i.clone(), Expr::load("input", i)),
        );
        let rendered = format!("{:?}", program.entry());
        assert!(
            rendered.contains(INDEXED_MAP_OP_ID),
            "Fix: indexed-map users must share the same child region instead of copying loop skeletons: {rendered}"
        );
    }

    #[test]
    fn strided_writeback_builder_emits_shared_child_region() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::F32).with_count(4)],
            [4, 1, 1],
            vec![wrap_anonymous_region(
                "vyre-libs::test::row_reduction_user",
                vec![
                    Node::let_bind("local", Expr::LocalId { axis: 0 }),
                    strided_writeback_child(
                        "vyre-libs::test::row_reduction_user",
                        4,
                        1,
                        4,
                        "out",
                        vec![Node::let_bind("scale", Expr::f32(0.5))],
                        |_idx| Expr::var("scale"),
                    ),
                ],
            )],
        );
        let rendered = format!("{:?}", program.entry());
        assert!(
            rendered.contains(STRIDED_WRITEBACK_OP_ID),
            "Fix: row-reduction writeback users must share the same child region instead of copying loop skeletons: {rendered}"
        );
    }
}
