//! Cooperative argmax and lane constants for math kernels.

use vyre_foundation::ir::{Expr, Node};
use vyre_libs_builder::builder::cooperative::for_each_index;
use vyre_libs_reduce::reduce::workgroup_tree::{
    max_f32_child, max_u32_child, min_u32_child, WorkgroupReductionScope,
};

/// Lanes a cooperative walk runs on, and the width of the scratch it reduces
/// through.
pub(crate) const LANES: u32 = 64;

/// The numeric kind of an argmax key.
#[derive(Clone, Copy)]
pub(crate) enum KeyKind {
    /// f32 keys, reduced by `max_f32_child`.
    F32,
    /// u32 keys, reduced by `max_u32_child`.
    U32,
}

impl KeyKind {
    /// The value a lane's key slot starts at.
    fn neutral(self) -> Expr {
        match self {
            Self::F32 => Expr::f32(0.0),
            Self::U32 => Expr::u32(0),
        }
    }

    /// The reduction child that collapses `tile` key partials to slot 0.
    fn max_child(
        self,
        op_id: &str,
        tile: u32,
        scratch: &'static str,
        scope: WorkgroupReductionScope,
    ) -> Node {
        match self {
            Self::F32 => max_f32_child(op_id, tile, scratch, scope),
            Self::U32 => max_u32_child(op_id, tile, scratch, scope),
        }
    }
}

/// A cooperative argmax over an index space, tie-broken by the lowest index.
pub(crate) struct Argmax<'a> {
    /// Op id the emitted reduction children record as their parent.
    pub(crate) op_id: &'static str,
    /// Number of logical indices.
    pub(crate) count: u32,
    /// Lanes the workgroup runs.
    pub(crate) tile: u32,
    /// Workgroup scratch the keys reduce through, `tile` entries of the key kind.
    pub(crate) key_scratch: &'static str,
    /// Numeric kind of the key, which decides the neutral and the reduction.
    pub(crate) key_kind: KeyKind,
    /// Workgroup scratch the indices reduce through, `tile` u32 entries.
    pub(crate) index_scratch: &'static str,
    /// The name the walk binds its index to, per pass.
    pub(crate) var: &'a str,
}

impl Argmax<'_> {
    /// Nodes that leave `key_scratch[0]` holding the maximum key and
    /// `index_scratch[0]` the lowest index attaining it.
    pub(crate) fn nodes<F>(&self, key: F) -> Vec<Node>
    where
        F: Fn(Expr) -> Expr,
    {
        let scope = WorkgroupReductionScope::FirstWorkgroup;
        let mut nodes = vec![
            Node::if_then(
                Expr::is_first_logical_tile(),
                vec![
                    Node::store(
                        self.key_scratch,
                        Expr::var("local"),
                        self.key_kind.neutral(),
                    ),
                    Node::store(self.index_scratch, Expr::var("local"), Expr::u32(u32::MAX)),
                ],
            ),
            Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
            Node::if_then(
                Expr::is_first_logical_tile(),
                vec![for_each_index(
                    self.count,
                    self.tile,
                    self.var,
                    vec![Node::store(
                        self.key_scratch,
                        Expr::var("local"),
                        Expr::max(
                            Expr::load(self.key_scratch, Expr::var("local")),
                            key(Expr::var(self.var)),
                        ),
                    )],
                )],
            ),
            Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
        ];
        nodes.push(
            self.key_kind
                .max_child(self.op_id, self.tile, self.key_scratch, scope),
        );
        nodes.push(Node::if_then(
            Expr::is_first_logical_tile(),
            vec![for_each_index(
                self.count,
                self.tile,
                self.var,
                vec![Node::if_then(
                    Expr::eq(
                        key(Expr::var(self.var)),
                        Expr::load(self.key_scratch, Expr::u32(0)),
                    ),
                    vec![Node::store(
                        self.index_scratch,
                        Expr::var("local"),
                        Expr::min(
                            Expr::load(self.index_scratch, Expr::var("local")),
                            Expr::var(self.var),
                        ),
                    )],
                )],
            )],
        ));
        nodes.push(Node::logical_barrier(
            vyre_foundation::ir::MemoryOrdering::SeqCst,
        ));
        nodes.push(min_u32_child(
            self.op_id,
            self.tile,
            self.index_scratch,
            scope,
        ));
        nodes
    }
}
