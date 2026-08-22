//! Shared elementwise Program builders and canonical composer.
//!
//! Category-A math, logical, quantization and NN wrappers keep domain-specific
//! names and op ids, but the repeated per-lane load/compute/store skeleton lives
//! here. It sits beside the indexed-map child it composes and above every
//! dialect that reaches for it: hosting it inside `math` made the boolean
//! dialect declare a dependency on the math broadcast surface to reach one
//! helper, and left the quantization ops with no way to reach it at all.

use std::ops::Range;
use std::sync::Arc;

use crate::builder::{build_indexed_map, check_tensors, BuildOptions};
use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};
use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

/// Canonical builder and composer for elementwise / pointwise operations.
///
/// Handles arbitrary arity, data types, broadcast policies, and scalar
/// compute closures while enforcing the single-pointwise IR invariant.
#[derive(Debug, Clone)]
pub struct ElementwiseComposer {
    op_id: &'static str,
    count: u32,
    workgroup_size: [u32; 3],
    buffers: Vec<BufferDecl>,
    anonymous: bool,
    child_op_id: Option<&'static str>,
}

impl ElementwiseComposer {
    /// Create a new elementwise composer for `count` elements under `op_id`.
    #[must_use]
    pub fn new(op_id: &'static str, count: u32) -> Self {
        Self {
            op_id,
            count,
            workgroup_size: [64, 1, 1],
            buffers: Vec::new(),
            anonymous: true,
            child_op_id: None,
        }
    }

    /// Set the workgroup geometry.
    #[must_use]
    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Inherit workgroup geometry and tenant options from [`BuildOptions`].
    #[must_use]
    pub fn with_options(mut self, options: &BuildOptions) -> Self {
        if let Some(wg) = options.workgroup_size {
            self.workgroup_size = wg;
        }
        self
    }

    /// Control whether the outer region is anonymous or named.
    #[must_use]
    pub fn with_anonymous(mut self, anonymous: bool) -> Self {
        self.anonymous = anonymous;
        self
    }

    /// Set a child region op_id if wrapping a nested child region.
    #[must_use]
    pub fn with_child_op_id(mut self, child_op_id: &'static str) -> Self {
        self.child_op_id = Some(child_op_id);
        self
    }

    /// Add an explicit buffer declaration.
    #[must_use]
    pub fn add_buffer(mut self, decl: BufferDecl) -> Self {
        self.buffers.push(decl);
        self
    }

    /// Add a standard ReadOnly input buffer with `count` elements.
    #[must_use]
    pub fn add_input(self, name: &str, dtype: DataType, count: u32) -> Self {
        self.add_input_storage(name, BufferAccess::ReadOnly, dtype, count)
    }

    /// Add a storage buffer with custom access permissions.
    #[must_use]
    pub fn add_input_storage(
        self,
        name: &str,
        access: BufferAccess,
        dtype: DataType,
        count: u32,
    ) -> Self {
        let idx = self.buffers.len() as u32;
        let decl = BufferDecl::storage(name, idx, access, dtype);
        let decl = if count == 0 {
            decl
        } else {
            decl.with_count(count)
        };
        self.add_buffer(decl)
    }

    /// Add a standard WriteOnly output buffer with `count` elements.
    pub fn add_output(self, name: &str, dtype: DataType, count: u32) -> Self {
        let idx = self.buffers.len() as u32;
        let elem_size = dtype.size_bytes().unwrap_or(4);
        let range = 0..(count as usize).saturating_mul(elem_size);
        self.add_buffer(
            BufferDecl::output(name, idx, dtype)
                .with_count(count)
                .with_output_byte_range(range),
        )
    }

    /// Add an output buffer with custom output byte range.
    #[must_use]
    pub fn add_output_with_byte_range(
        self,
        name: &str,
        dtype: DataType,
        count: u32,
        range: Range<usize>,
    ) -> Self {
        let idx = self.buffers.len() as u32;
        self.add_buffer(
            BufferDecl::output(name, idx, dtype)
                .with_count(count)
                .with_output_byte_range(range),
        )
    }

    /// Add an output buffer configured as storage with custom access (e.g. WriteOnly, ReadWrite).
    #[must_use]
    pub fn add_output_storage(
        self,
        name: &str,
        access: BufferAccess,
        dtype: DataType,
        count: u32,
    ) -> Self {
        let idx = self.buffers.len() as u32;
        self.add_buffer(
            BufferDecl::written(name, idx, access, dtype)
                .with_count(count)
                .with_full_output_byte_range(),
        )
    }

    /// Build a custom loop kernel given a body generator closure `Fn(Expr) -> Vec<Node>`.
    ///
    /// The closure receives `Expr::var("idx")` or the loop induction variable.
    #[must_use]
    pub fn build_custom<F>(self, body_fn: F) -> Program
    where
        F: FnOnce(Expr) -> Vec<Node>,
    {
        let loop_idx = Expr::InvocationId { axis: 0 };
        let inner_nodes = body_fn(loop_idx.clone());
        let loop_body = vec![Node::if_then(
            Expr::lt(loop_idx, Expr::u32(self.count)),
            inner_nodes,
        )];
        let region = if self.anonymous {
            if let Some(child_op) = self.child_op_id {
                wrap_anonymous_region(
                    self.op_id,
                    vec![wrap_child_region(
                        child_op,
                        Ident::from(self.op_id),
                        loop_body,
                    )],
                )
            } else {
                wrap_anonymous_region(self.op_id, loop_body)
            }
        } else {
            Node::Region {
                generator: Ident::from(self.op_id),
                source_region: None,
                body: Arc::new(loop_body),
            }
        };

        Program::wrapped(self.buffers, self.workgroup_size, vec![region])
    }

    /// Build a standard single-output pointwise loop where `compute(i)` returns the value to store in `output[i]`.
    #[must_use]
    pub fn build_pointwise<F>(self, output: &str, f: F) -> Program
    where
        F: FnOnce(Expr) -> Expr,
    {
        let out_buf = output.to_string();
        self.build_custom(|i| vec![Node::store(&out_buf, i.clone(), f(i))])
    }

    /// Build a multi-output pointwise loop where `compute(i)` returns values stored in `outputs[0][i]`, `outputs[1][i]`, ...
    #[must_use]
    pub fn build_pointwise_multi<F>(self, outputs: &[&str], f: F) -> Program
    where
        F: FnOnce(Expr) -> Vec<Expr>,
    {
        let out_bufs: Vec<String> = outputs.iter().map(|s| s.to_string()).collect();
        self.build_custom(|i| {
            let values = f(i.clone());
            assert_eq!(
                out_bufs.len(),
                values.len(),
                "Output buffer count must match values count"
            );
            out_bufs
                .into_iter()
                .zip(values)
                .map(|(buf, val)| Node::store(&buf, i.clone(), val))
                .collect()
        })
    }

    /// Helper for unary elementwise op `out[i] = op(in[i])`.
    #[must_use]
    pub fn unary<F>(
        op_id: &'static str,
        input: &str,
        in_dtype: DataType,
        output: &str,
        out_dtype: DataType,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr) -> Expr,
    {
        Self::new(op_id, count)
            .add_input(input, in_dtype, count)
            .add_output(output, out_dtype, count)
            .build_pointwise(output, |i| op(Expr::load(input, i)))
    }

    /// Helper for u32 unary elementwise op.
    #[must_use]
    pub fn u32_unary<F>(
        op_id: &'static str,
        input: &str,
        output: &str,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr) -> Expr,
    {
        Self::unary(
            op_id,
            input,
            DataType::U32,
            output,
            DataType::U32,
            count,
            op,
        )
    }

    /// Helper for f32 unary elementwise op.
    #[must_use]
    pub fn f32_unary<F>(
        op_id: &'static str,
        input: &str,
        output: &str,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr) -> Expr,
    {
        Self::unary(
            op_id,
            input,
            DataType::F32,
            output,
            DataType::F32,
            count,
            op,
        )
    }

    /// Helper for binary elementwise op `out[i] = op(lhs[i], rhs[i])`.
    #[must_use]
    pub fn binary<F>(
        op_id: &'static str,
        lhs: &str,
        rhs: &str,
        in_dtype: DataType,
        output: &str,
        out_dtype: DataType,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr, Expr) -> Expr,
    {
        Self::new(op_id, count)
            .add_input(lhs, in_dtype.clone(), count)
            .add_input(rhs, in_dtype, count)
            .add_output(output, out_dtype, count)
            .build_pointwise(output, |i| {
                let l = Expr::load(lhs, i.clone());
                let r = Expr::load(rhs, i);
                op(l, r)
            })
    }

    /// Helper for u32 binary elementwise op.
    #[must_use]
    pub fn u32_binary<F>(
        op_id: &'static str,
        lhs: &str,
        rhs: &str,
        output: &str,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr, Expr) -> Expr,
    {
        Self::binary(
            op_id,
            lhs,
            rhs,
            DataType::U32,
            output,
            DataType::U32,
            count,
            op,
        )
    }

    /// Helper for f32 binary elementwise op.
    #[must_use]
    pub fn f32_binary<F>(
        op_id: &'static str,
        lhs: &str,
        rhs: &str,
        output: &str,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr, Expr) -> Expr,
    {
        Self::binary(
            op_id,
            lhs,
            rhs,
            DataType::F32,
            output,
            DataType::F32,
            count,
            op,
        )
    }

    /// Helper for ternary elementwise op `out[i] = op(a[i], b[i], c[i])`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn ternary<F>(
        op_id: &'static str,
        a: &str,
        b: &str,
        c: &str,
        in_dtype: DataType,
        output: &str,
        out_dtype: DataType,
        count: u32,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr, Expr, Expr) -> Expr,
    {
        Self::new(op_id, count)
            .add_input(a, in_dtype.clone(), count)
            .add_input(b, in_dtype.clone(), count)
            .add_input(c, in_dtype, count)
            .add_output(output, out_dtype, count)
            .build_pointwise(output, |i| {
                let va = Expr::load(a, i.clone());
                let vb = Expr::load(b, i.clone());
                let vc = Expr::load(c, i);
                op(va, vb, vc)
            })
    }

    /// Broadcast a single scalar `src[0]` across all elements `0..n` of `dst`.
    #[must_use]
    pub fn broadcast_scalar(
        op_id: &'static str,
        src: &str,
        dst: &str,
        n: u32,
        dtype: DataType,
    ) -> Program {
        Self::new(op_id, n)
            .add_input(src, dtype.clone(), 1)
            .add_output(dst, dtype, n)
            .build_pointwise(dst, |_i| Expr::load(src, Expr::u32(0)))
    }

    /// Fill all elements `0..n` of `target` with a constant expression.
    #[must_use]
    pub fn fill_constant(
        op_id: &'static str,
        target: &str,
        n: u32,
        dtype: DataType,
        constant: Expr,
    ) -> Program {
        Self::new(op_id, n)
            .add_output(target, dtype, n)
            .build_pointwise(target, |_i| constant.clone())
    }

    /// Helper for elementwise binary map where RHS has custom indexing policy / broadcast.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn binary_broadcast_rhs<F, I>(
        op_id: &'static str,
        lhs: &str,
        rhs: &str,
        out: &str,
        count: u32,
        rhs_count: u32,
        dtype: DataType,
        rhs_index: I,
        op: F,
    ) -> Program
    where
        F: FnOnce(Expr, Expr) -> Expr,
        I: FnOnce(&Expr) -> Expr,
    {
        Self::new(op_id, count)
            .with_workgroup_size([256, 1, 1])
            .add_input_storage(lhs, BufferAccess::ReadOnly, dtype.clone(), count)
            .add_input_storage(rhs, BufferAccess::ReadOnly, dtype.clone(), rhs_count)
            .add_output(out, dtype, count)
            .build_pointwise(out, |i| {
                let r_idx = rhs_index(&i);
                let l = Expr::load(lhs, i);
                let r = Expr::load(rhs, r_idx);
                op(l, r)
            })
    }

    /// Checked builder for unary elementwise Program using `TensorRef`.
    pub fn try_unary<F>(
        op_id: &'static str,
        a: TensorRef,
        out: TensorRef,
        options: BuildOptions,
        op: F,
    ) -> Result<Program, TensorRefError>
    where
        F: FnOnce(Expr) -> Expr,
    {
        check_tensors(op_id, &[(&a, DataType::U32), (&out, DataType::U32)])?;

        if a.shape != out.shape {
            return Err(TensorRefError::ShapeMismatch {
                name: "elementwise_unary".into(),
                found: vec![],
                expected: vec![],
                op: op_id,
            });
        }

        let n = a
            .element_count()
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: a.name_str().to_string(),
                shape: a.shape.to_vec(),
            })?;

        Ok(Self::new(op_id, n)
            .with_options(&options)
            .add_input(a.name_str(), DataType::U32, n)
            .add_output(out.name_str(), DataType::U32, n)
            .build_pointwise(out.name_str(), |i| op(Expr::load(a.name_str(), i))))
    }

    /// Checked builder for binary elementwise Program using `TensorRef`.
    pub fn try_binary<F>(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
        options: BuildOptions,
        op: F,
    ) -> Result<Program, TensorRefError>
    where
        F: FnOnce(Expr, Expr) -> Expr,
    {
        check_tensors(
            op_id,
            &[
                (&a, DataType::U32),
                (&b, DataType::U32),
                (&out, DataType::U32),
            ],
        )?;

        if a.shape != b.shape {
            return Err(TensorRefError::ShapeMismatch {
                name: b.name_str().to_string(),
                found: b.shape.to_vec(),
                expected: a.shape.to_vec(),
                op: op_id,
            });
        }

        if a.shape != out.shape {
            return Err(TensorRefError::ShapeMismatch {
                name: "elementwise_binary".into(),
                found: vec![],
                expected: vec![],
                op: op_id,
            });
        }

        let a_count = a
            .element_count()
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: a.name_str().to_string(),
                shape: a.shape.to_vec(),
            })?;
        let out_count =
            out.element_count()
                .ok_or_else(|| TensorRefError::ElementCountOverflow {
                    name: out.name_str().to_string(),
                    shape: out.shape.to_vec(),
                })?;
        if out_count < a_count {
            return Err(TensorRefError::ShapeMismatch {
                name: out.name_str().to_string(),
                found: out.shape.to_vec(),
                expected: a.shape.to_vec(),
                op: op_id,
            });
        }

        let n = a_count;
        Ok(Self::new(op_id, n)
            .with_options(&options)
            .add_input(a.name_str(), DataType::U32, n)
            .add_input(b.name_str(), DataType::U32, n)
            .add_output(out.name_str(), DataType::U32, n)
            .build_pointwise(out.name_str(), |i| {
                let l = Expr::load(a.name_str(), i.clone());
                let r = Expr::load(b.name_str(), i);
                op(l, r)
            }))
    }
}

/// Right-hand side source for an elementwise F32 multiply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F32MulRhs<'a> {
    /// Reuse the left input as RHS, producing `x * x`.
    SameInput,
    /// Read RHS from a second buffer.
    Buffer(&'a str),
}

/// Build `output[i] = input[i] * rhs[i]` over F32 lanes.
#[must_use]
pub(crate) fn f32_elementwise_mul(
    op_id: &'static str,
    input: &str,
    rhs: F32MulRhs<'_>,
    output: &str,
    n: u32,
) -> Program {
    let mut buffers =
        vec![BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n)];
    if let F32MulRhs::Buffer(buffer) = rhs {
        buffers.push(
            BufferDecl::storage(buffer, 1, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        );
        buffers.push(BufferDecl::output(output, 2, DataType::F32).with_count(n));
    } else {
        buffers.push(BufferDecl::output(output, 1, DataType::F32).with_count(n));
    }

    build_indexed_map(op_id, buffers, output, n, [64, 1, 1], |i| {
        let lhs_value = Expr::load(input, i.clone());
        let rhs_value = match rhs {
            F32MulRhs::SameInput => lhs_value.clone(),
            F32MulRhs::Buffer(buffer) => Expr::load(buffer, i.clone()),
        };
        (i, Expr::mul(lhs_value, rhs_value))
    })
}

/// Build a checked elementwise unary u32 operation.
pub(crate) fn try_u32_elementwise_unary<F>(
    op_id: &'static str,
    input: &str,
    out: &str,
    size: u32,
    op: F,
) -> Result<Program, TensorRefError>
where
    F: Fn(Expr) -> Expr,
{
    crate::builder::build_elementwise_unary(
        op_id,
        TensorRef::u32_1d(input, size),
        TensorRef::u32_1d(out, size),
        BuildOptions::default(),
        op,
    )
}

/// Build an elementwise unary u32 operation with a diagnostic invalid-program fallback.
#[must_use]
pub(crate) fn u32_elementwise_unary<F>(
    op_id: &'static str,
    input: &str,
    out: &str,
    size: u32,
    op: F,
) -> Program
where
    F: Fn(Expr) -> Expr,
{
    try_u32_elementwise_unary(op_id, input, out, size, op).unwrap_or_else(|err| {
        trap_program(op_id, Some((out, DataType::U32)), format!("Fix: {err}"))
    })
}

/// Build a checked elementwise binary u32 operation.
pub(crate) fn try_u32_elementwise_binary<F>(
    op_id: &'static str,
    a: &str,
    b: &str,
    out: &str,
    size: u32,
    op: F,
) -> Result<Program, TensorRefError>
where
    F: Fn(Expr, Expr) -> Expr,
{
    crate::builder::build_elementwise_binary(
        op_id,
        TensorRef::u32_1d(a, size),
        TensorRef::u32_1d(b, size),
        TensorRef::u32_1d(out, size),
        BuildOptions::default(),
        op,
    )
}

/// Build an elementwise binary u32 operation with a diagnostic invalid-program fallback.
#[must_use]
pub(crate) fn u32_elementwise_binary<F>(
    op_id: &'static str,
    a: &str,
    b: &str,
    out: &str,
    size: u32,
    op: F,
) -> Program
where
    F: Fn(Expr, Expr) -> Expr,
{
    try_u32_elementwise_binary(op_id, a, b, out, size, op).unwrap_or_else(|err| {
        trap_program(op_id, Some((out, DataType::U32)), format!("Fix: {err}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_composer_unary_emits_expected_structure() {
        let program =
            ElementwiseComposer::f32_unary("vyre-libs::test::unary", "in", "out", 100, |x| {
                Expr::mul(x, Expr::f32(2.0))
            });
        assert_eq!(program.buffers().len(), 2);
        assert_eq!(program.workgroup_size(), [64, 1, 1]);
        assert_eq!(program.buffers()[0].name(), "in");
        assert_eq!(program.buffers()[1].name(), "out");
    }

    #[test]
    fn elementwise_composer_binary_emits_expected_structure() {
        let program = ElementwiseComposer::u32_binary(
            "vyre-libs::test::binary",
            "a",
            "b",
            "out",
            50,
            Expr::add,
        );
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.buffers()[0].name(), "a");
        assert_eq!(program.buffers()[1].name(), "b");
        assert_eq!(program.buffers()[2].name(), "out");
    }

    #[test]
    fn elementwise_composer_broadcast_scalar_emits_one_input() {
        let program = ElementwiseComposer::broadcast_scalar(
            "vyre-libs::test::bcast",
            "src",
            "dst",
            64,
            DataType::F32,
        );
        assert_eq!(program.buffers().len(), 2);
        assert_eq!(program.buffers()[0].count(), 1);
        assert_eq!(program.buffers()[1].count(), 64);
    }

    #[test]
    fn elementwise_composer_fill_constant_emits_output_target() {
        let program = ElementwiseComposer::fill_constant(
            "vyre-libs::test::zero",
            "target",
            32,
            DataType::U32,
            Expr::u32(0),
        );
        assert_eq!(program.buffers().len(), 1);
        assert!(program.buffers()[0].is_output());
    }

    #[test]
    fn elementwise_composer_try_unary_validates_tensor_mismatches() {
        let a = TensorRef::u32_1d("a", 16);
        let out = TensorRef::u32_1d("out", 32);
        let err = ElementwiseComposer::try_unary(
            "vyre-libs::test::try_unary_mismatch",
            a,
            out,
            BuildOptions::default(),
            |x| x,
        )
        .unwrap_err();
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[test]
    fn elementwise_composer_try_binary_validates_tensor_mismatches() {
        let a = TensorRef::u32_1d("a", 16);
        let b = TensorRef::u32_1d("b", 32);
        let out = TensorRef::u32_1d("out", 16);
        let err = ElementwiseComposer::try_binary(
            "vyre-libs::test::try_binary_mismatch",
            a,
            b,
            out,
            BuildOptions::default(),
            Expr::add,
        )
        .unwrap_err();
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[test]
    fn elementwise_composer_multi_output_emits_multiple_stores() {
        let program = ElementwiseComposer::new("vyre-libs::test::multi_out", 10)
            .add_input("in", DataType::F32, 10)
            .add_output("out1", DataType::F32, 10)
            .add_output("out2", DataType::F32, 10)
            .build_pointwise_multi(&["out1", "out2"], |i| {
                let x = Expr::load("in", i);
                vec![x.clone(), Expr::mul(x, Expr::f32(2.0))]
            });
        assert_eq!(program.buffers().len(), 3);
    }
}
