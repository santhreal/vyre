//! Node lowering: statements, control flow, and child-body construction.

use crate::descriptor::{KernelBody, KernelOp, KernelOpKind, OpaqueNodeData};
use crate::error::LowerError;
use rustc_hash::FxHashSet;
use vyre_foundation::ir::model::node::node_op_id;
use vyre_foundation::ir::{Expr, Ident, Node};

use super::body_assembly::{empty_body_for_nodes, push_child};
use super::carrier_names::collect_carrier_names;
use super::{LowerCtx, MAX_NESTING_DEPTH};

impl LowerCtx {
    pub(super) fn lower_nodes(
        &mut self,
        nodes: &[Node],
        body: &mut KernelBody,
        depth: usize,
    ) -> Result<(), LowerError> {
        if depth > MAX_NESTING_DEPTH {
            return Err(LowerError::NestingTooDeep(depth));
        }
        for node in nodes {
            self.lower_node(node, body, depth)?;
        }
        Ok(())
    }

    fn lower_node(
        &mut self,
        node: &Node,
        body: &mut KernelBody,
        depth: usize,
    ) -> Result<(), LowerError> {
        match node {
            Node::Region {
                generator,
                body: region,
                ..
            } => self.lower_child_node(
                body,
                depth,
                region.as_ref(),
                KernelOpKind::Region {
                    generator: generator.shared_text(),
                },
            ),
            Node::Block(region) => {
                self.lower_child_node(body, depth, region, KernelOpKind::StructuredBlock)
            }
            Node::Let { name, value } => {
                let id = self.lower_expr(value, body)?;
                let id = if let Expr::Var(source) = value {
                    if self.is_active_carrier(source) {
                        self.copy_value(body, id)?
                    } else {
                        id
                    }
                } else {
                    id
                };
                self.scope.bind(name.clone(), id);
                Ok(())
            }
            Node::Assign { name, value } => {
                let id = self.lower_expr(value, body)?;
                if self.is_active_carrier(name) {
                    // Assign of an active loop carrier: commit the new
                    // value to the function-local via LoopCarrierEnd,
                    // then re-read so subsequent in-scope references
                    // pick up a fresh SSA id sourced from the local.
                    // This bypasses if-then phi-merge for carrier vars
                    // because the merge's seed-vs-then Select cannot
                    // represent the per-iteration state  -  the
                    // authoritative store is the function-local.
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrierEnd {
                            name: name.shared_text(),
                        },
                        operands: vec![id],
                        result: None,
                    });
                    let read_id = self.alloc_value()?;
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrier {
                            name: name.shared_text(),
                        },
                        operands: Vec::new(),
                        result: Some(read_id),
                    });
                    self.scope.bind(name.clone(), read_id);
                } else {
                    self.scope.bind(name.clone(), id);
                }
                Ok(())
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                let slot = self.buffer_slot(buffer)?;
                let index_id = self.lower_expr(index, body)?;
                let value_id = self.lower_expr(value, body)?;
                body.ops.push(KernelOp {
                    kind: self.store_kind(slot, buffer)?,
                    operands: vec![slot, index_id, value_id],
                    result: None,
                });
                Ok(())
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                let cond_id = self.lower_expr(cond, body)?;
                let incoming_scope = self.scope.snapshot();
                let mut if_carriers = collect_carrier_names(then, &incoming_scope, None);
                for name in collect_carrier_names(otherwise, &incoming_scope, None) {
                    if !if_carriers.contains(&name) {
                        if_carriers.push(name);
                    }
                }
                for name in &if_carriers {
                    let seed_id = incoming_scope
                        .get(name)
                        .copied()
                        .unwrap_or_else(|| unreachable!("if carrier must have incoming binding"));
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrierInit {
                            name: name.shared_text(),
                        },
                        operands: vec![seed_id],
                        result: None,
                    });
                }
                let mut then_body = empty_body_for_nodes(then);
                self.scope.restore(incoming_scope.clone());
                for name in &if_carriers {
                    let read_id = self.alloc_value()?;
                    then_body.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrier {
                            name: name.shared_text(),
                        },
                        operands: Vec::new(),
                        result: Some(read_id),
                    });
                    self.scope.bind(name.clone(), read_id);
                }
                let mut carrier_frame: FxHashSet<Ident> = FxHashSet::default();
                for name in &if_carriers {
                    carrier_frame.insert(name.clone());
                }
                self.active_carriers.push(carrier_frame);
                self.lower_nodes(then, &mut then_body, depth + 1)?;
                self.active_carriers.pop();
                let then_id = push_child(body, then_body)?;
                if otherwise.is_empty() {
                    self.scope.restore(incoming_scope.clone());
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::StructuredIfThen,
                        operands: vec![cond_id, then_id],
                        result: None,
                    });
                } else {
                    let mut else_body = empty_body_for_nodes(otherwise);
                    self.scope.restore(incoming_scope.clone());
                    for name in &if_carriers {
                        let read_id = self.alloc_value()?;
                        else_body.ops.push(KernelOp {
                            kind: KernelOpKind::LoopCarrier {
                                name: name.shared_text(),
                            },
                            operands: Vec::new(),
                            result: Some(read_id),
                        });
                        self.scope.bind(name.clone(), read_id);
                    }
                    let mut carrier_frame: FxHashSet<Ident> = FxHashSet::default();
                    for name in &if_carriers {
                        carrier_frame.insert(name.clone());
                    }
                    self.active_carriers.push(carrier_frame);
                    self.lower_nodes(otherwise, &mut else_body, depth + 1)?;
                    self.active_carriers.pop();
                    let else_id = push_child(body, else_body)?;
                    self.scope.restore(incoming_scope.clone());
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::StructuredIfThenElse,
                        operands: vec![cond_id, then_id, else_id],
                        result: None,
                    });
                }
                for name in &if_carriers {
                    self.emit_loop_carrier_read(body, name)?;
                }
                Ok(())
            }
            Node::Loop {
                var,
                from,
                to,
                body: loop_body,
            } => {
                let from_id = self.lower_expr(from, body)?;
                let to_id = self.lower_expr(to, body)?;
                let incoming_scope = self.scope.snapshot();

                // Identify source-level variables that are reassigned inside
                // the loop body AND were already bound in the incoming scope.
                // These are the loop carriers  -  their per-iteration value
                // must round-trip through a function-local because the SSA
                // operand of in-body reads is baked at lowering time and
                // would otherwise stay anchored to the pre-loop seed,
                // making `Assign` inside a loop have no observable effect
                // across iterations.
                let carrier_names =
                    collect_carrier_names(loop_body, &incoming_scope, Some(var));

                // Pre-loop init: in the parent body, store the seed value
                // into each carrier slot.
                for name in &carrier_names {
                    let seed_id = incoming_scope
                        .get(name)
                        .copied()
                        .unwrap_or_else(|| unreachable!("carrier name has incoming binding"));
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrierInit {
                            name: name.shared_text(),
                        },
                        operands: vec![seed_id],
                        result: None,
                    });
                }

                let mut child = empty_body_for_nodes(loop_body);
                let loop_index_id = self.alloc_value()?;
                child.ops.push(KernelOp {
                    kind: KernelOpKind::LoopIndex {
                        loop_var: var.shared_text(),
                    },
                    operands: Vec::new(),
                    result: Some(loop_index_id),
                });
                self.scope.bind(var.clone(), loop_index_id);

                // First op of each iteration: re-read the carrier slot so
                // every in-body reference to the source-level variable
                // resolves to the latest value committed by the previous
                // iteration (or the pre-loop seed on iteration 0).
                for name in &carrier_names {
                    let read_id = self.alloc_value()?;
                    child.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrier {
                            name: name.shared_text(),
                        },
                        operands: Vec::new(),
                        result: Some(read_id),
                    });
                    self.scope.bind(name.clone(), read_id);
                }

                let mut carrier_frame: FxHashSet<Ident> = FxHashSet::default();
                for name in &carrier_names {
                    carrier_frame.insert(name.clone());
                }
                self.active_carriers.push(carrier_frame);
                self.lower_nodes(loop_body, &mut child, depth + 1)?;
                self.active_carriers.pop();
                let loop_exit_scope = self.scope.snapshot();

                self.scope
                    .restore_loop_exit(incoming_scope, &loop_exit_scope, var);
                let child_id = push_child(body, child)?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::StructuredForLoop {
                        loop_var: var.shared_text(),
                    },
                    operands: vec![from_id, to_id, child_id],
                    result: None,
                });

                // Post-loop: emit a fresh LoopCarrier read in the parent
                // so post-loop references to each carrier name resolve to
                // the loop's final stored value. Rebind in scope so
                // `Var(name)` reads downstream resolve to this id rather
                // than the pre-loop seed.
                for name in &carrier_names {
                    let post_id = self.alloc_value()?;
                    body.ops.push(KernelOp {
                        kind: KernelOpKind::LoopCarrier {
                            name: name.shared_text(),
                        },
                        operands: Vec::new(),
                        result: Some(post_id),
                    });
                    self.scope.bind(name.clone(), post_id);
                }
                Ok(())
            }
            Node::Barrier { ordering } => {
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Barrier {
                        ordering: *ordering,
                    },
                    operands: Vec::new(),
                    result: None,
                });
                Ok(())
            }
            Node::IndirectDispatch {
                count_buffer,
                count_offset,
            } => {
                let slot = self.buffer_slot(count_buffer)?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::IndirectDispatch {
                        count_offset: *count_offset,
                    },
                    operands: vec![slot],
                    result: None,
                });
                Ok(())
            }
            Node::AsyncLoad {
                source,
                destination,
                offset,
                size,
                tag,
            } => self.lower_async_copy(
                body,
                KernelOpKind::AsyncLoad {
                    tag: tag.shared_text(),
                },
                source,
                destination,
                offset,
                size,
            ),
            Node::AsyncStore {
                source,
                destination,
                offset,
                size,
                tag,
            } => self.lower_async_copy(
                body,
                KernelOpKind::AsyncStore {
                    tag: tag.shared_text(),
                },
                source,
                destination,
                offset,
                size,
            ),
            Node::AsyncWait { tag } => {
                body.ops.push(KernelOp {
                    kind: KernelOpKind::AsyncWait {
                        tag: tag.shared_text(),
                    },
                    operands: Vec::new(),
                    result: None,
                });
                Ok(())
            }
            Node::Trap { address, tag } => {
                let address_id = self.lower_expr(address, body)?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Trap {
                        tag: tag.shared_text(),
                    },
                    operands: vec![address_id],
                    result: None,
                });
                Ok(())
            }
            Node::Resume { tag } => {
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Resume {
                        tag: tag.shared_text(),
                    },
                    operands: Vec::new(),
                    result: None,
                });
                Ok(())
            }
            Node::Return => {
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Return,
                    operands: Vec::new(),
                    result: None,
                });
                Ok(())
            }
            Node::Opaque(extension) => {
                body.ops.push(KernelOp {
                    kind: KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
                        extension_kind: extension.extension_kind().to_owned(),
                        payload: extension.wire_payload(),
                    })),
                    operands: Vec::new(),
                    result: None,
                });
                Ok(())
            }
            other => Err(LowerError::UnsupportedConstruct(format!(
                "node `{}` has no KernelDescriptor lowering. Fix: add a KernelOpKind mapping before routing this program through vyre-lower.",
                node_op_id(other)
            ))),
        }
    }

    fn lower_child_node(
        &mut self,
        body: &mut KernelBody,
        depth: usize,
        nodes: &[Node],
        kind: KernelOpKind,
    ) -> Result<(), LowerError> {
        let incoming_scope = self.scope.snapshot();

        // Names reassigned inside the region whose pre-region binding lives
        // in the enclosing scope. The parent body cannot reference an SSA
        // id emitted inside the child KernelBody (Naga's `Statement::Block`
        // closes the inner scope), so reassignments must round-trip through
        // a function-local. Reuses the same `LoopCarrierInit/LoopCarrier/
        // LoopCarrierEnd` machinery loops use  -  the local-allocation and
        // store/load semantics are identical; only the iteration is absent.
        let region_carriers = collect_carrier_names(nodes, &incoming_scope, None);

        // Pre-region: in the parent body, store each carrier's incoming SSA
        // value into the function-local. Idempotent across nested regions
        // sharing a name (the emitter dedupes named-carrier locals).
        for name in &region_carriers {
            let seed_id = incoming_scope
                .get(name)
                .copied()
                .unwrap_or_else(|| unreachable!("region carrier must have incoming binding"));
            body.ops.push(KernelOp {
                kind: KernelOpKind::LoopCarrierInit {
                    name: name.shared_text(),
                },
                operands: vec![seed_id],
                result: None,
            });
        }

        let mut child = empty_body_for_nodes(nodes);

        // Top of region: reload each carrier so reads inside resolve to a
        // fresh in-region SSA id sourced from the local.
        for name in &region_carriers {
            let read_id = self.alloc_value()?;
            child.ops.push(KernelOp {
                kind: KernelOpKind::LoopCarrier {
                    name: name.shared_text(),
                },
                operands: Vec::new(),
                result: Some(read_id),
            });
            self.scope.bind(name.clone(), read_id);
        }

        // Mark the carriers active so `Node::Assign { name, .. }` inside
        // the body emits `LoopCarrierEnd` (commit to local) followed by
        // `LoopCarrier` (re-read) instead of just rebinding the SSA id  -
        // the rebind alone would leak an in-region SSA into the parent.
        let mut carrier_frame: FxHashSet<Ident> = FxHashSet::default();
        for name in &region_carriers {
            carrier_frame.insert(name.clone());
        }
        self.active_carriers.push(carrier_frame);
        self.lower_nodes(nodes, &mut child, depth + 1)?;
        self.active_carriers.pop();

        // Discard in-region `Let`-introduced bindings: they're scoped to
        // the child KernelBody and would otherwise leak into the parent's
        // name table where their SSA ids are out of scope. Carriers will
        // be rebound to fresh post-region read ids below.
        self.scope.restore(incoming_scope);

        let child_id = push_child(body, child)?;
        body.ops.push(KernelOp {
            kind,
            operands: vec![child_id],
            result: None,
        });

        // Post-region: in the parent body, reload each carrier so
        // subsequent reads see the in-region final value. Rebind in scope
        // so `Var(name)` downstream resolves to this id rather than the
        // pre-region seed. Without this read, `n_tokens=0` is the symptom:
        // the in-region final value of `tok_idx` (and every other Assign'd
        // name) was emitted as an SSA id local to the child KernelBody,
        // out of scope from the parent's reads.
        for name in &region_carriers {
            let post_id = self.alloc_value()?;
            body.ops.push(KernelOp {
                kind: KernelOpKind::LoopCarrier {
                    name: name.shared_text(),
                },
                operands: Vec::new(),
                result: Some(post_id),
            });
            self.scope.bind(name.clone(), post_id);
        }
        Ok(())
    }

    fn lower_async_copy(
        &mut self,
        body: &mut KernelBody,
        kind: KernelOpKind,
        source: &Ident,
        destination: &Ident,
        offset: &Expr,
        size: &Expr,
    ) -> Result<(), LowerError> {
        let source_slot = self.buffer_slot(source)?;
        let destination_slot = self.buffer_slot(destination)?;
        let offset_id = self.lower_expr(offset, body)?;
        let size_id = self.lower_expr(size, body)?;
        body.ops.push(KernelOp {
            kind,
            operands: vec![source_slot, destination_slot, offset_id, size_id],
            result: None,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::lower;
    use crate::descriptor::{KernelBody, KernelOp, KernelOpKind};
    use vyre_foundation::ir::{BufferAccess, DataType, Program};

    #[test]
    fn loop_variable_lowers_to_child_loop_index_result() {
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::output("out", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::buf_len("input"),
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::load("input", Expr::var("i")),
                )],
            )],
        );

        let desc = lower(&program).expect("Fix: loop variable must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());
        let (loop_body, loop_op) =
            find_loop(&desc.body).expect("Fix: structured loop op must be present");
        let child = &loop_body.child_bodies[loop_op.operands[2] as usize];
        assert!(
            matches!(
                child.ops.first().map(|op| &op.kind),
                Some(KernelOpKind::LoopIndex { loop_var }) if loop_var.as_ref() == "i"
            ),
            "loop body must materialize the induction value before lowering input[i]"
        );

        fn find_loop(body: &KernelBody) -> Option<(&KernelBody, &KernelOp)> {
            for op in &body.ops {
                if matches!(op.kind, KernelOpKind::StructuredForLoop { .. }) {
                    return Some((body, op));
                }
            }
            body.child_bodies.iter().find_map(find_loop)
        }
    }

    #[test]
    fn loop_variable_does_not_clobber_same_named_outer_binding() {
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("i", Expr::u32(9)),
                Node::loop_for("i", Expr::u32(0), Expr::u32(1), vec![]),
                Node::store("out", Expr::u32(0), Expr::var("i")),
            ],
        );

        let desc = lower(&program).expect("Fix: shadowed loop variable must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());
        let store = find_store(&desc.body).expect("Fix: post-loop store must be present");
        assert_eq!(
            store.operands[2], 0,
            "post-loop read must use the outer i binding, not the loop induction result"
        );

        fn find_store(body: &KernelBody) -> Option<&KernelOp> {
            body.ops
                .iter()
                .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
                .or_else(|| body.child_bodies.iter().find_map(find_store))
        }
    }

    #[test]
    fn if_else_branches_lower_from_the_same_incoming_scope() {
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("x", Expr::u32(1)),
                Node::if_then_else(
                    Expr::bool(true),
                    vec![Node::assign("x", Expr::add(Expr::var("x"), Expr::u32(1)))],
                    vec![Node::store("out", Expr::u32(0), Expr::var("x"))],
                ),
            ],
        );

        let desc = lower(&program).expect("Fix: if/else must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());
        let (_, if_op) = find_if_else(&desc.body).expect("Fix: if/else op must be present");
        let parent = find_parent_body_containing_op(&desc.body, if_op as *const KernelOp)
            .expect("Fix: if op parent body must be found");
        let else_body = &parent.child_bodies[if_op.operands[2] as usize];
        let else_store = else_body
            .ops
            .iter()
            .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
            .expect("Fix: else branch must contain the store");
        let else_carrier = else_body
            .ops
            .iter()
            .find(
                |op| matches!(&op.kind, KernelOpKind::LoopCarrier { name } if name.as_ref() == "x"),
            )
            .expect(
                "Fix: else branch must read x through the if carrier seeded from incoming scope",
            );
        let else_carrier_id = else_carrier
            .result
            .expect("Fix: else carrier read must produce an SSA result");
        assert_eq!(
            else_store.operands[2], else_carrier_id,
            "else branch must read the incoming x through its carrier, not the result assigned only by then"
        );

        fn find_if_else(body: &KernelBody) -> Option<(&KernelBody, &KernelOp)> {
            for op in &body.ops {
                if matches!(op.kind, KernelOpKind::StructuredIfThenElse) {
                    return Some((body, op));
                }
            }
            body.child_bodies.iter().find_map(find_if_else)
        }

        fn find_parent_body_containing_op(
            body: &KernelBody,
            target: *const KernelOp,
        ) -> Option<&KernelBody> {
            if body.ops.iter().any(|op| std::ptr::eq(op, target)) {
                return Some(body);
            }
            body.child_bodies
                .iter()
                .find_map(|child| find_parent_body_containing_op(child, target))
        }
    }
}
