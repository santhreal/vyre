//! Canonical CSR traversal composer and builder.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, Expr, MemoryOrdering, Node, Program};

use super::*;
use crate::bitset::bitset_words;

/// Canonical CSR traversal composer and builder.
#[derive(Clone, Debug)]
pub struct CsrTraversalComposer<'a> {
    /// Operation ID recorded on the wrapping IR Region.
    pub op_id: &'a str,
    /// Diagnostic name used in trap/error messages.
    pub builder_name: &'a str,
    /// CSR buffer names.
    pub buffers: CsrBuffers<'a>,
    /// Traversal direction.
    pub direction: CsrDirection,
    /// Local identifier prefix for hygiene under nested inlining.
    pub prefix: &'a str,
    /// Allowed edge-kind bitmask.
    pub allow_mask: u32,
    /// Logical vertex / node count.
    pub node_count: u32,
    /// Physical / logical edge count.
    pub edge_count: u32,
    /// Execution workgroup size.
    pub workgroup_size: [u32; 3],
}

impl<'a> CsrTraversalComposer<'a> {
    /// Create a new CSR traversal composer with defaults.
    #[must_use]
    pub fn new(op_id: &'a str, builder_name: &'a str, node_count: u32) -> Self {
        Self {
            op_id,
            builder_name,
            buffers: CsrBuffers::default(),
            direction: CsrDirection::Forward,
            prefix: "",
            allow_mask: 0xFFFF_FFFF,
            node_count,
            edge_count: 0,
            workgroup_size: CSR_TRAVERSAL_WORKGROUP_SIZE,
        }
    }

    /// Convenience constructor for forward traversal.
    #[must_use]
    pub fn forward(op_id: &'a str, node_count: u32, edge_count: u32, allow_mask: u32) -> Self {
        Self::new(op_id, op_id, node_count)
            .with_direction(CsrDirection::Forward)
            .with_allow_mask(allow_mask)
            .with_edge_count(edge_count)
    }

    /// Convenience constructor for backward traversal.
    #[must_use]
    pub fn backward(op_id: &'a str, node_count: u32, edge_count: u32, allow_mask: u32) -> Self {
        Self::new(op_id, op_id, node_count)
            .with_direction(CsrDirection::Backward)
            .with_allow_mask(allow_mask)
            .with_edge_count(edge_count)
    }

    /// Set CSR buffer names.
    #[must_use]
    pub const fn with_buffers(mut self, buffers: CsrBuffers<'a>) -> Self {
        self.buffers = buffers;
        self
    }

    /// Set traversal direction.
    #[must_use]
    pub const fn with_direction(mut self, direction: CsrDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set local identifier prefix.
    #[must_use]
    pub const fn with_prefix(mut self, prefix: &'a str) -> Self {
        self.prefix = prefix;
        self
    }

    /// Set allowed edge-kind mask.
    #[must_use]
    pub const fn with_allow_mask(mut self, allow_mask: u32) -> Self {
        self.allow_mask = allow_mask;
        self
    }

    /// Set edge count.
    #[must_use]
    pub const fn with_edge_count(mut self, edge_count: u32) -> Self {
        self.edge_count = edge_count;
        self
    }

    /// Set workgroup size.
    #[must_use]
    pub const fn with_workgroup_size(mut self, workgroup_size: [u32; 3]) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    /// Disambiguate local binding names by prefix.
    #[must_use]
    pub fn local_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}_{name}", self.prefix)
        }
    }

    /// Emit row offset loads `[edge_start = offsets[src], edge_end = offsets[src + 1]]`.
    #[must_use]
    pub fn emit_row_offsets(&self, src: Expr, start_var: &str, end_var: &str) -> [Node; 2] {
        [
            Node::let_bind(start_var, Expr::load(self.buffers.offsets, src.clone())),
            Node::let_bind(
                end_var,
                Expr::load(self.buffers.offsets, Expr::add(src, Expr::u32(1))),
            ),
        ]
    }

    /// Emit a bounded loop over the CSR edges of source node `src`.
    #[must_use]
    pub fn emit_row_bounds_and_loop(
        &self,
        src: Expr,
        edge_var: &str,
        loop_body: Vec<Node>,
    ) -> Vec<Node> {
        let edge_start = self.local_name("edge_start");
        let edge_end = self.local_name("edge_end");
        let [lo, hi] = self.emit_row_offsets(src, edge_start.as_str(), edge_end.as_str());
        vec![
            lo,
            hi,
            Node::loop_for(
                edge_var,
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                loop_body,
            ),
        ]
    }

    /// Emit degree computation for source node `src`: `degree = offsets[src + 1] - offsets[src]`.
    #[must_use]
    pub fn emit_row_degree(
        &self,
        src: Expr,
        lo_var: &str,
        hi_var: &str,
        degree_var: &str,
    ) -> [Node; 3] {
        let [lo, hi] = self.emit_row_offsets(src, lo_var, hi_var);
        [
            lo,
            hi,
            Node::let_bind(degree_var, Expr::sub(Expr::var(hi_var), Expr::var(lo_var))),
        ]
    }

    /// Emit edge walk over source node `src` with edge-kind mask filtering and in-bounds destination check.
    #[must_use]
    pub fn emit_neighbor_walk<F>(
        &self,
        src: Expr,
        edge_var: Option<&str>,
        on_neighbor: F,
    ) -> Vec<Node>
    where
        F: Fn(Expr, Expr) -> Vec<Node>,
    {
        let edge_iter = edge_var
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.local_name("e"));
        let kind_mask_var = self.local_name("kind_mask");
        let dst_var = self.local_name("dst");

        let inner = on_neighbor(Expr::var(dst_var.as_str()), Expr::var(edge_iter.as_str()));
        let dst_guard = if self.node_count > 0 {
            vec![
                Node::let_bind(
                    dst_var.as_str(),
                    Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                ),
                Node::if_then(
                    Expr::lt(Expr::var(dst_var.as_str()), Expr::u32(self.node_count)),
                    inner,
                ),
            ]
        } else {
            vec![
                Node::let_bind(
                    dst_var.as_str(),
                    Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                ),
                Node::block(inner),
            ]
        };

        let loop_body = match self.buffers.edge_kind_mask {
            Some(mask_buf) => {
                vec![
                    Node::let_bind(
                        kind_mask_var.as_str(),
                        Expr::load(mask_buf, Expr::var(edge_iter.as_str())),
                    ),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(
                                Expr::var(kind_mask_var.as_str()),
                                Expr::u32(self.allow_mask),
                            ),
                            Expr::u32(0),
                        ),
                        dst_guard,
                    ),
                ]
            }
            None => dst_guard,
        };

        self.emit_row_bounds_and_loop(src, edge_iter.as_str(), loop_body)
    }

    /// Emit ONLY the CSR edge walk for source node `src`: load edge range, filter by kind mask,
    /// atomic-OR the target bit into `frontier_out`, and invoke `on_new_bit()` when a bit flips 0→1.
    #[must_use]
    pub fn emit_edge_expand(
        &self,
        frontier_out: &str,
        src: Expr,
        frontier_index: impl Fn(Expr) -> Expr,
        on_new_bit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        let name = |n: &str| self.local_name(n);
        let edge_start = name("edge_start");
        let edge_end = name("edge_end");
        let edge_iter = name("e");
        let kind_mask = name("kind_mask");
        let dst = name("dst");
        let dst_word_idx = name("dst_word_idx");
        let dst_bit = name("dst_bit");

        let flip_body = on_new_bit();
        let pre_or_word = name(if flip_body.is_empty() { "_prev" } else { "old" });
        let on_bounded = set_bit(
            frontier_out,
            &Expr::var(dst.as_str()),
            BitAccess {
                word: dst_word_idx.as_str(),
                mask: dst_bit.as_str(),
                value: pre_or_word.as_str(),
            },
            frontier_index,
            flip_body,
        );

        let kind_buf = self.buffers.edge_kind_mask.unwrap_or(NAME_EDGE_KIND_MASK);

        vec![
            Node::let_bind(
                edge_start.as_str(),
                Expr::load(self.buffers.offsets, src.clone()),
            ),
            Node::let_bind(
                edge_end.as_str(),
                Expr::load(self.buffers.offsets, Expr::add(src, Expr::u32(1))),
            ),
            Node::loop_for(
                edge_iter.as_str(),
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                vec![
                    Node::let_bind(
                        kind_mask.as_str(),
                        Expr::load(kind_buf, Expr::var(edge_iter.as_str())),
                    ),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(Expr::var(kind_mask.as_str()), Expr::u32(self.allow_mask)),
                            Expr::u32(0),
                        ),
                        vec![
                            Node::let_bind(
                                dst.as_str(),
                                Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                            ),
                            Node::if_then(
                                Expr::lt(Expr::var(dst.as_str()), Expr::u32(self.node_count)),
                                on_bounded,
                            ),
                        ],
                    ),
                ],
            ),
        ]
    }

    /// Emit the CSR neighbor expansion for one source node `src`, reading its frontier bit INLINE
    /// and expanding out-edges only when set.
    #[must_use]
    pub fn emit_edge_scan(
        &self,
        frontier_out: &str,
        src: Expr,
        frontier_index: impl Fn(Expr) -> Expr,
        on_new_bit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        let name = |n: &str| self.local_name(n);
        let word_idx = name("word_idx");
        let bit_mask = name("bit_mask");
        let src_word = name("src_word");

        let expand = self.emit_edge_expand(frontier_out, src.clone(), &frontier_index, on_new_bit);
        when_bit_set(
            frontier_out,
            &src,
            Some(word_idx.as_str()),
            src_word.as_str(),
            bit_mask.as_str(),
            frontier_index,
            expand,
        )
    }

    /// Emit one backward CSR traversal pass for candidate node `src`.
    #[must_use]
    pub fn emit_backward_scan(
        &self,
        src: Expr,
        frontier_in: &str,
        frontier_out: &str,
        on_hit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        self.emit_backward_scan_full(src, frontier_in, frontier_out, on_hit, Vec::new)
    }

    /// Emit one backward CSR traversal pass for candidate node `src` with custom bit-set actions.
    #[must_use]
    pub fn emit_backward_scan_full(
        &self,
        src: Expr,
        frontier_in: &str,
        frontier_out: &str,
        on_hit: impl Fn() -> Vec<Node>,
        on_new_bit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        let name = |n: &str| self.local_name(n);
        let edge_start = name("edge_start");
        let edge_end = name("edge_end");
        let edge_iter = name("e");
        let kind_mask = name("kind_mask");
        let dst = name("dst");
        let hit = name("hit");

        let kind_buf = self.buffers.edge_kind_mask.unwrap_or(NAME_EDGE_KIND_MASK);

        let hit_actions = on_hit();
        let hit_body = if hit_actions.is_empty() {
            vec![Node::assign(hit.as_str(), Expr::u32(1))]
        } else {
            let mut b = vec![Node::assign(hit.as_str(), Expr::u32(1))];
            b.extend(hit_actions);
            b
        };

        vec![
            Node::let_bind(
                edge_start.as_str(),
                Expr::load(self.buffers.offsets, src.clone()),
            ),
            Node::let_bind(
                edge_end.as_str(),
                Expr::load(self.buffers.offsets, Expr::add(src.clone(), Expr::u32(1))),
            ),
            Node::let_bind(hit.as_str(), Expr::u32(0)),
            Node::loop_for(
                edge_iter.as_str(),
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                vec![Node::if_then(
                    Expr::eq(Expr::var(hit.as_str()), Expr::u32(0)),
                    vec![
                        Node::let_bind(
                            kind_mask.as_str(),
                            Expr::load(kind_buf, Expr::var(edge_iter.as_str())),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(
                                    Expr::var(kind_mask.as_str()),
                                    Expr::u32(self.allow_mask),
                                ),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    dst.as_str(),
                                    Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var(dst.as_str()), Expr::u32(self.node_count)),
                                    when_bit_set(
                                        frontier_in,
                                        &Expr::var(dst.as_str()),
                                        None,
                                        "dst_word",
                                        "dst_bit",
                                        |word| word,
                                        hit_body,
                                    ),
                                ),
                            ],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var(hit.as_str()), Expr::u32(1)),
                set_bit(
                    frontier_out,
                    &src,
                    BitAccess {
                        word: "src_word_idx",
                        mask: "src_bit",
                        value: "_prev",
                    },
                    |word| word,
                    on_new_bit(),
                ),
            ),
        ]
    }

    /// Build a single-step forward CSR frontier traversal program.
    #[must_use]
    pub fn build_forward_step(&self, frontier_in: &str, frontier_out: &str) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        buffers.push(csr_frontier_buffer(
            frontier_in,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            self.node_count,
        ));

        let t = Expr::InvocationId { axis: 0 };
        let active_body =
            self.emit_edge_expand(frontier_out, Expr::var("src"), |word| word, Vec::new);
        let body = vec![active_source_lane(
            self.node_count,
            frontier_in,
            None,
            t,
            active_body,
        )];

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(self.op_id, body)],
        )
    }

    /// Build a single-step forward CSR traversal program excluding specified source nodes.
    #[must_use]
    pub fn build_forward_step_excluding(
        &self,
        frontier_in: &str,
        excluded_sources: &str,
        frontier_out: &str,
    ) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        buffers.push(csr_frontier_buffer(
            frontier_in,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            excluded_sources,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START + 2,
            BufferAccess::ReadWrite,
            self.node_count,
        ));

        let t = Expr::InvocationId { axis: 0 };
        let active_body =
            self.emit_edge_expand(frontier_out, Expr::var("src"), |word| word, Vec::new);
        let body = vec![active_source_lane(
            self.node_count,
            frontier_in,
            Some(excluded_sources),
            t,
            active_body,
        )];

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(self.op_id, body)],
        )
    }

    /// Build a single-step backward CSR frontier traversal program.
    #[must_use]
    pub fn build_backward_step(&self, frontier_in: &str, frontier_out: &str) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        buffers.push(csr_frontier_buffer(
            frontier_in,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            self.node_count,
        ));

        let t = Expr::InvocationId { axis: 0 };
        let mut body = vec![Node::let_bind("src", t.clone())];
        body.extend(self.emit_backward_scan(Expr::var("src"), frontier_in, frontier_out, Vec::new));

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(
                self.op_id,
                vec![Node::if_then(Expr::lt(t, Expr::u32(self.node_count)), body)],
            )],
        )
    }

    /// Build parallel in-place backward expansion program with atomic changed flag.
    #[must_use]
    pub fn build_parallel_backward_or_changed(&self, frontier_out: &str, changed: &str) -> Program {
        let src = Expr::InvocationId { axis: 0 };
        let body =
            self.emit_backward_scan_full(src.clone(), frontier_out, frontier_out, Vec::new, || {
                vec![Node::let_bind(
                    "_changed",
                    Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
                )]
            });

        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        csr_push_frontier_changed_buffers(&mut buffers, frontier_out, changed, self.node_count);
        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(
                self.op_id,
                vec![Node::if_then(
                    Expr::lt(src, Expr::u32(self.node_count)),
                    body,
                )],
            )],
        )
    }

    /// Build parallel in-place forward expansion program with atomic changed flag.
    #[must_use]
    pub fn build_parallel_forward_or_changed(&self, frontier_out: &str, changed: &str) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        csr_push_frontier_changed_buffers(&mut buffers, frontier_out, changed, self.node_count);

        let body =
            self.emit_parallel_forward_or_changed_body(frontier_out, changed, None, None, None);

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(self.op_id, body)],
        )
    }

    /// Build parallel in-place forward expansion body with optional snapshot barrier.
    #[must_use]
    pub fn emit_parallel_forward_or_changed_body(
        &self,
        frontier_out: &str,
        changed: &str,
        snapshot_barrier: Option<MemoryOrdering>,
        active_gate: Option<Expr>,
        extra_changed: Option<(&str, Expr)>,
    ) -> Vec<Node> {
        let local = |name: &str| -> String {
            if self.prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}_{name}", self.prefix)
            }
        };
        let src = Expr::gid_x();
        let in_bounds = local("in_bounds");
        let word_idx = local("word_idx");
        let bit_mask = local("bit_mask");
        let src_word = local("src_word");
        let src_active = local("src_active");
        let changed_old = local("changed_old");
        let extra_changed_old = local("extra_changed_old");

        let mark_changed = || {
            let mut nodes = vec![Node::let_bind(
                changed_old.as_str(),
                Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
            )];
            if let Some((extra_changed_buffer, extra_changed_index)) = &extra_changed {
                nodes.push(Node::let_bind(
                    extra_changed_old.as_str(),
                    Expr::atomic_or(
                        extra_changed_buffer,
                        extra_changed_index.clone(),
                        Expr::u32(1),
                    ),
                ));
            }
            nodes
        };

        let edge_scan =
            || self.emit_edge_expand(frontier_out, src.clone(), |word| word, &mark_changed);

        if let Some(ordering) = snapshot_barrier {
            let ungated_src_active = Expr::select(
                Expr::var(in_bounds.as_str()),
                Expr::bitand(Expr::var(src_word.as_str()), Expr::var(bit_mask.as_str())),
                Expr::u32(0),
            );
            let src_active_expr = if let Some(active_gate) = active_gate {
                Expr::select(
                    Expr::ne(active_gate, Expr::u32(0)),
                    ungated_src_active,
                    Expr::u32(0),
                )
            } else {
                ungated_src_active
            };
            let mut preamble = vec![Node::let_bind(
                in_bounds.as_str(),
                Expr::lt(src.clone(), Expr::u32(self.node_count)),
            )];
            preamble.extend(bind_bit_address(
                &src,
                word_idx.as_str(),
                bit_mask.as_str(),
                |word| Expr::select(Expr::var(in_bounds.as_str()), word, Expr::u32(0)),
            ));
            preamble.extend([
                Node::let_bind(
                    src_word.as_str(),
                    Expr::load(frontier_out, Expr::var(word_idx.as_str())),
                ),
                Node::let_bind(src_active.as_str(), src_active_expr),
                Node::barrier_with_ordering(ordering),
                Node::if_then(
                    Expr::ne(Expr::var(src_active.as_str()), Expr::u32(0)),
                    edge_scan(),
                ),
            ]);
            return preamble;
        }

        let mut body =
            bind_bit_address(&src, word_idx.as_str(), bit_mask.as_str(), |word| word).to_vec();
        body.push(Node::let_bind(
            src_word.as_str(),
            Expr::load(frontier_out, Expr::var(word_idx.as_str())),
        ));
        body.push(Node::if_then(
            bit_is_set(BitAccess {
                word: word_idx.as_str(),
                mask: bit_mask.as_str(),
                value: src_word.as_str(),
            }),
            edge_scan(),
        ));

        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(self.node_count)),
            body,
        )]
    }

    /// Build parallel batched forward expansion program over multiple query frontiers.
    pub fn build_parallel_batch_forward_or_changed(
        &self,
        frontier_out: &str,
        changed: &str,
        query_count: u32,
        changed_index: Expr,
        changed_slots: u32,
        mut prologue: Vec<Node>,
        extra_buffers: Vec<BufferDecl>,
    ) -> Result<Program, String> {
        if query_count == 0 {
            return Err(format!(
                "Fix: {} requires at least one query frontier.",
                self.builder_name
            ));
        }
        let src = Expr::InvocationId { axis: 0 };
        let query = Expr::InvocationId { axis: 1 };
        let words = bitset_words(self.node_count);
        let total_words = checked_batched_frontier_words(words, query_count)?;
        let query_word_base = Expr::mul(query, Expr::u32(words));

        let mut body = vec![Node::let_bind("query_word_base", query_word_base)];
        body.extend(self.emit_edge_scan(
            frontier_out,
            src.clone(),
            |word| Expr::add(Expr::var("query_word_base"), word),
            || {
                vec![Node::let_bind(
                    "_changed",
                    Expr::atomic_or(changed, changed_index.clone(), Expr::u32(1)),
                )]
            },
        ));
        prologue.append(&mut body);

        let mut buffers = try_csr_read_only_buffers(self.node_count, self.edge_count)?;
        buffers.push(csr_word_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadWrite,
            total_words.max(1),
        ));
        buffers.push(csr_word_buffer(
            changed,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            changed_slots,
        ));
        buffers.extend(extra_buffers);

        Ok(Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(
                self.op_id,
                vec![Node::if_then(
                    Expr::lt(src, Expr::u32(self.node_count)),
                    prologue,
                )],
            )],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_OP_ID: &str = "test::csr::traversal";

    #[test]
    fn forward_traversal_program_emits_valid_structure() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_forward", 64)
            .with_edge_count(128)
            .with_allow_mask(0x00FF);
        let program = composer.build_forward_step("fin", "fout");
        assert_eq!(program.workgroup_size, CSR_TRAVERSAL_WORKGROUP_SIZE);
        assert_eq!(program.buffers.len(), 7); // nodes, offsets, targets, kinds, tags, fin, fout
    }

    #[test]
    fn backward_traversal_program_emits_valid_structure() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_backward", 32)
            .with_edge_count(64)
            .with_direction(CsrDirection::Backward);
        let program = composer.build_backward_step("fin", "fout");
        assert_eq!(program.workgroup_size, CSR_TRAVERSAL_WORKGROUP_SIZE);
        assert_eq!(program.buffers.len(), 7);
    }

    #[test]
    fn excluding_forward_traversal_declares_extra_buffer() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_excluding", 100);
        let program = composer.build_forward_step_excluding("fin", "fex", "fout");
        assert_eq!(program.buffers.len(), 8); // nodes, offsets, targets, kinds, tags, fin, fex, fout
    }

    #[test]
    fn parallel_backward_or_changed_structure() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_bwd_changed", 128)
            .with_direction(CsrDirection::Backward);
        let program = composer.build_parallel_backward_or_changed("fout", "changed");
        assert_eq!(program.buffers.len(), 7);
    }

    #[test]
    fn parallel_batch_forward_validation() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_batch", 64);
        let err = composer.build_parallel_batch_forward_or_changed(
            "fout",
            "changed",
            0,
            Expr::u32(0),
            1,
            Vec::new(),
            Vec::new(),
        );
        assert!(err.is_err());

        let ok = composer.build_parallel_batch_forward_or_changed(
            "fout",
            "changed",
            4,
            Expr::InvocationId { axis: 1 },
            4,
            Vec::new(),
            Vec::new(),
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn row_degree_emission_nodes() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_degree", 10);
        let [lo, hi, deg] = composer.emit_row_degree(Expr::var("src"), "lo", "hi", "deg");
        let rendered = format!("{lo:?} {hi:?} {deg:?}");
        assert!(rendered.contains("lo"));
        assert!(rendered.contains("hi"));
        assert!(rendered.contains("deg"));
    }

    #[test]
    fn prefix_hygiene() {
        let composer =
            CsrTraversalComposer::new(TEST_OP_ID, "test_prefix", 10).with_prefix("custom");
        assert_eq!(composer.local_name("e"), "custom_e");
        assert_eq!(composer.local_name("dst"), "custom_dst");
    }
}
