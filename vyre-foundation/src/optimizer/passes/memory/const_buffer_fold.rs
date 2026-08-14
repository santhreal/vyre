//! Compile-time constant-buffer folding and shader monomorphization support.
//!
//! This pass-level utility replaces loads from known immutable buffers with
//! literal immediates. Lowering then emits immediate expressions instead of
//! storage-buffer reads, which lets a backend monomorphize shaders for static
//! LUTs without carrying a runtime binding.

use std::borrow::Cow;

use crate::ir::{Expr, Ident, Program};
use crate::optimizer::{fingerprint_program, PassResult};
use crate::transform::rewrite_walk::{self, NodeRewrite};

/// Compile-time-known u32 buffer contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstBuffer {
    /// Buffer name referenced by `Expr::Load`.
    pub name: Ident,
    /// Immutable u32 values available at compile time.
    pub values: Vec<u32>,
}

/// Inline literal loads from a compile-time-known u32 buffer.
#[must_use]
pub fn fold_const_buffer(program: &Program, constant: &ConstBuffer) -> PassResult {
    let before_fp = fingerprint_program(program);
    let mut fold = ConstBufferFold { constant };
    let optimized = match rewrite_walk::rewrite_body(program.entry(), &mut fold) {
        Some(entry) => program.with_rewritten_entry(entry),
        None => program.clone(),
    };
    let changed = fingerprint_program(&optimized) != before_fp;
    PassResult {
        program: optimized,
        changed,
    }
}

struct ConstBufferFold<'a> {
    constant: &'a ConstBuffer,
}

impl NodeRewrite for ConstBufferFold<'_> {
    /// Replace a load from the constant buffer at a literal index with that
    /// element.
    ///
    /// The expression rewrite is bottom-up, so the index is already folded when
    /// the load is offered, and it descends into every operand position of
    /// every expression variant. A load nested in a subgroup operand, an atomic
    /// compare value, or a call argument therefore folds without this pass
    /// listing those variants: leaving one out left a residual load behind, and
    /// a caller that then drops the now-immutable binding dangled it.
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        match crate::optimizer::rewrite::rewrite_expr(expr, &mut |candidate| {
            let Expr::Load { buffer, index } = candidate else {
                return None;
            };
            if buffer != &self.constant.name {
                return None;
            }
            let Expr::LitU32(element) = **index else {
                return None;
            };
            self.constant
                .values
                .get(element as usize)
                .copied()
                .map(Expr::u32)
        }) {
            Cow::Borrowed(_) => None,
            Cow::Owned(folded) => Some(folded),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType, Node};

    #[test]
    fn const_buffer_inlined_when_compile_time_known() {
        let program =
            crate::optimizer::passes::cleanup::region_inline_engine::run(Program::wrapped(
                vec![
                    BufferDecl::read("lut", 0, DataType::U32).with_count(256),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::load("lut", Expr::u32(7)),
                )],
            ));
        let result = fold_const_buffer(
            &program,
            &ConstBuffer {
                name: "lut".into(),
                values: (0..256).map(|value| value * 3).collect(),
            },
        );

        assert!(result.changed);
        let body = crate::test_region_body::region_body(&result.program);
        assert!(matches!(
            &body[0],
            Node::Store {
                value: Expr::LitU32(21),
                ..
            }
        ));
    }

    #[test]
    fn const_buffer_folds_inside_subgroup_operand() {
        // Regression: a load from the const buffer nested inside a subgroup
        // operand (`subgroup_add(load(lut, 7))`) must fold to the literal too.
        // Before the fix, `fold_expr`'s `_ => expr.clone()` catch-all covered
        // `SubgroupReduce` and skipped the descent, leaving the load in place --
        // so a caller that then drops the immutable `lut` binding would dangle
        // the residual load. `result.changed` stayed false and the store value
        // remained `subgroup_add(load(...))`.
        let program =
            crate::optimizer::passes::cleanup::region_inline_engine::run(Program::wrapped(
                vec![
                    BufferDecl::read("lut", 0, DataType::U32).with_count(256),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::subgroup_add(Expr::load("lut", Expr::u32(7))),
                )],
            ));
        let result = fold_const_buffer(
            &program,
            &ConstBuffer {
                name: "lut".into(),
                values: (0..256).map(|value| value * 3).collect(),
            },
        );

        assert!(
            result.changed,
            "folding the lut load inside subgroup_add must change the program"
        );
        let body = crate::test_region_body::region_body(&result.program);
        let Node::Store { value, .. } = &body[0] else {
            panic!("expected a store, got {:?}", body[0]);
        };
        let Expr::SubgroupReduce { value: inner, .. } = value else {
            panic!("store value must remain a subgroup reduce, got {value:?}");
        };
        assert_eq!(
            **inner,
            Expr::LitU32(21),
            "lut[7] == 7*3 == 21 must be folded into the subgroup operand, not left as a load"
        );
    }

    #[test]
    fn out_of_range_index_stays_as_load() {
        let program =
            crate::optimizer::passes::cleanup::region_inline_engine::run(Program::wrapped(
                vec![
                    BufferDecl::read("lut", 0, DataType::U32).with_count(4),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::load("lut", Expr::u32(999)),
                )],
            ));
        let result = fold_const_buffer(
            &program,
            &ConstBuffer {
                name: "lut".into(),
                values: vec![10, 20, 30, 40],
            },
        );
        assert!(!result.changed);
    }

    #[test]
    fn different_buffer_not_folded() {
        let program =
            crate::optimizer::passes::cleanup::region_inline_engine::run(Program::wrapped(
                vec![
                    BufferDecl::read("data", 0, DataType::U32).with_count(4),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::load("data", Expr::u32(0)),
                )],
            ));
        let result = fold_const_buffer(
            &program,
            &ConstBuffer {
                name: "lut".into(),
                values: vec![42],
            },
        );
        assert!(!result.changed);
    }

    #[test]
    fn const_buffer_struct_eq() {
        let a = ConstBuffer {
            name: "x".into(),
            values: vec![1, 2, 3],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
