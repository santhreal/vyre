//! `NodeVisitor` implementation for `PreorderValidator`.

use std::convert::Infallible;
use std::ops::ControlFlow;

use hashbrown::hash_map::RawEntryMut;

use super::PreorderValidator;
use crate::composition::self_exclusive_region_key;
use crate::ir_inner::model::expr::{Expr, Ident};
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::op_signature::DataType;
use crate::validate::binding::{check_sibling_duplicate, Binding};
use crate::validate::typecheck::{expr_type, ScopeTypes};
use crate::validate::uniformity::is_uniform;
use crate::validate::{
    barrier, depth, err, node_rules, shadowing, ValidationLocation, ValidationPhase,
};
use crate::visit::NodeVisitor;

macro_rules! async_transfer_visitors {
    () => {
        fn visit_async_load(
            &mut self,
            _node: &Node,
            _source: &Ident,
            destination: &Ident,
            offset: &Expr,
            size: &Expr,
            tag: &Ident,
        ) -> ControlFlow<Self::Break> {
            self.validate_async_transfer(destination, offset, size, tag)
        }

        fn visit_async_store(
            &mut self,
            _node: &Node,
            _source: &Ident,
            destination: &Ident,
            offset: &Expr,
            size: &Expr,
            tag: &Ident,
        ) -> ControlFlow<Self::Break> {
            self.validate_async_transfer(destination, offset, size, tag)
        }
    };
}

impl NodeVisitor for PreorderValidator<'_, '_> {
    type Break = Infallible;

    fn visit_let(&mut self, _node: &Node, name: &Ident, value: &Expr) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(value, 0);

        let Some(frame) = self.scope_stack.last_mut() else {
            self.errors.push(err(
                "V118",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                    "malformed validation frame stream: let binding `{name}` appeared outside any scope"
                ),
                "rebuild the program through the structured IR builder before validation."
                    .to_string(),
            ));
            return ControlFlow::Continue(());
        };
        // Same-region duplicate Lets are always invalid, even when
        // shadowing is allowed for nested scopes  -  the V032 contract
        // covered by `sibling_duplicate_lets_are_rejected_even_when_shadowing_is_allowed`.
        // `allow_shadowing` only opens nested scopes; siblings collide
        // unconditionally.
        let duplicate_sibling = check_sibling_duplicate(
            name,
            &mut frame.region_bindings,
            /*allow_duplicate_siblings=*/ false,
            &mut self.errors,
        );
        if !duplicate_sibling {
            shadowing::check_local(name, &self.scope, self.options, &mut self.errors);
        }
        let ty_opt = expr_type(value, &mut ScopeTypes::new(&self.buffers, &self.scope));
        let ty = ty_opt.clone().unwrap_or(DataType::U32);
        let ty_known = ty_opt.is_some();
        let uniform = is_uniform(value, &self.scope);
        node_rules::insert_binding(
            &mut self.scope,
            name.clone(),
            Binding {
                ty,
                ty_known,
                mutable: true,
                uniform,
            },
            Some(&mut frame.scope_log),
        );

        ControlFlow::Continue(())
    }

    fn visit_assign(
        &mut self,
        _node: &Node,
        name: &Ident,
        value: &Expr,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_assign(name, value, &self.buffers, &self.scope, &mut self.errors);
        self.validate_expr(value, 0);

        // Reassigning with a divergent rhs taints the binding's
        // uniformity for the remainder of its lifetime.
        let new_uniform = is_uniform(value, &self.scope);
        if let Some(binding) = self.scope.get_mut(name.as_str()) {
            binding.uniform = binding.uniform && new_uniform;
        }

        ControlFlow::Continue(())
    }

    fn visit_store(
        &mut self,
        _node: &Node,
        buffer: &Ident,
        index: &Expr,
        value: &Expr,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_store(
            buffer,
            index,
            value,
            &self.buffers,
            &self.scope,
            &mut self.errors,
        );
        self.validate_expr(index, 0);
        self.validate_expr(value, 0);

        ControlFlow::Continue(())
    }

    fn visit_if(
        &mut self,
        _node: &Node,
        cond: &Expr,
        _then: &[Node],
        _otherwise: &[Node],
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(cond, 0);
        node_rules::check_if_condition(cond, &self.buffers, &self.scope, &mut self.errors);

        ControlFlow::Continue(())
    }

    fn visit_loop(
        &mut self,
        _node: &Node,
        var: &Ident,
        from: &Expr,
        to: &Expr,
        body: &[Node],
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(from, 0);
        self.validate_expr(to, 0);
        node_rules::check_loop_bounds(from, to, &self.buffers, &self.scope, &mut self.errors);
        shadowing::check_local(var, &self.scope, self.options, &mut self.errors);
        let bounds_uniform = is_uniform(from, &self.scope) && is_uniform(to, &self.scope);
        let var_uniform = bounds_uniform && !self.current_divergent();
        let mut back_edge_scope = self.scope.clone();
        back_edge_scope.insert(var.clone(), node_rules::loop_var_binding(var_uniform));
        barrier::check_loop_back_edge(body, &back_edge_scope, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_indirect_dispatch(
        &mut self,
        _node: &Node,
        count_buffer: &Ident,
        count_offset: u64,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_indirect_dispatch(
            count_buffer,
            count_offset,
            &self.buffers,
            &mut self.errors,
        );
        ControlFlow::Continue(())
    }

    async_transfer_visitors!();

    fn visit_async_wait(&mut self, _node: &Node, tag: &Ident) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_async_tag(tag, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_trap(
        &mut self,
        _node: &Node,
        address: &Expr,
        _tag: &Ident,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(address, 0);
        ControlFlow::Continue(())
    }

    fn visit_resume(&mut self, _node: &Node, _tag: &Ident) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_return(&mut self, _node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_barrier(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        let divergent = self.current_divergent();
        let Node::Barrier { ordering } = node else {
            self.errors.push(err(
                "V129",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                "malformed barrier visitor dispatch".to_string(),
                "rebuild the program through the structured IR builder before validation."
                    .to_string(),
            ));
            return ControlFlow::Continue(());
        };
        barrier::check_barrier(divergent, *ordering, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_collective(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_collective(node, self.options, &self.buffers, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_tile(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        match node {
            Node::TileLoad {
                tile,
                tile_type,
                buffer,
                origin,
                ..
            } => {
                for expr in origin {
                    self.validate_expr(expr, 0);
                }
                node_rules::check_tile_load(
                    tile,
                    tile_type,
                    buffer,
                    origin,
                    &self.buffers,
                    self.options,
                    &mut self.errors,
                );
            }
            Node::TileStore {
                buffer,
                origin,
                tile,
            } => {
                for expr in origin {
                    self.validate_expr(expr, 0);
                }
                node_rules::check_tile_store(buffer, origin, tile, &self.buffers, &mut self.errors);
            }
            Node::TileMatmul { acc, a, b } => {
                node_rules::check_tile_matmul(acc, a, b, self.options, &mut self.errors);
            }
            Node::TileDecl { name, tile } => {
                node_rules::check_tile_residency(name, tile, self.options, &mut self.errors);
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn visit_block(&mut self, _node: &Node, _body: &[Node]) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_region(
        &mut self,
        _node: &Node,
        generator: &Ident,
        _source_region: &Option<crate::ir_inner::model::expr::Ident>,
        _body: &[Node],
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        if let Some(base) = self_exclusive_region_key(generator.as_str()) {
            match self.self_comp_counts.raw_entry_mut().from_key(base) {
                RawEntryMut::Occupied(mut o) => *o.get_mut() += 1,
                RawEntryMut::Vacant(v) => {
                    v.insert(base.to_string(), 1);
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn visit_opaque_node(
        &mut self,
        _node: &Node,
        extension: &dyn crate::ir_inner::model::node::NodeExtension,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_opaque_node_extension(extension, &mut self.errors);
        ControlFlow::Continue(())
    }
}
