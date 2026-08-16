//! Mapping a logical index space onto the lanes of one workgroup.
//!
//! A kernel that walks an index space has two spellings. It can bind the lane
//! id and stride, so `tile` lanes share the work, or it can loop from zero to
//! the count in one lane. The second is what most of this crate does: measured
//! over `vyre-libs/src`, 373 `Node::loop_for` skeletons name 36 uses of a
//! thread index between them, so the dominant shape is a sequential loop that
//! occupies one lane of a workgroup and leaves the rest idle. `[1, 1, 1]` and a
//! guard on invocation zero is the same decision made at the program level.
//!
//! Two things live here, and nothing else: the strided lane walk, and the
//! cooperative argmax built on it. Both are skeletons, so a caller supplies the
//! body or the key and owns the arithmetic; neither knows what is being walked.
//!
//! The walk is emitted rather than written out because it has one bound that is
//! easy to get wrong. The chunk count is a static ceiling over `count / tile`,
//! so the last chunk overshoots and the body has to run behind an
//! `index < count` guard. Written by hand at each site, the ceiling and the
//! guard are two places for an off-by-one to hide, and `vyre-pass-engine`
//! independently grew the same skeleton for its arena passes.

use vyre_foundation::ir::{Expr, Node};

use crate::reduce::workgroup_tree::{
    max_f32_child, max_u32_child, min_u32_child, WorkgroupReductionScope,
};

/// Chunks a workgroup of `tile` lanes needs to cover `count` indices.
///
/// At least one: a body behind the bounds guard costs nothing when the count is
/// zero, and a loop with an upper bound of zero would make an empty space a
/// different shape from a small one.
#[must_use]
pub(crate) const fn chunks(count: u32, tile: u32) -> u32 {
    if tile == 0 {
        return 1;
    }
    let chunks = count.div_ceil(tile);
    if chunks == 0 {
        1
    } else {
        chunks
    }
}

/// A strided walk that offers each index of `0..count` to `body` once.
///
/// `var` is bound to the logical index this lane handles in this chunk, and the
/// body runs only for an index the space contains. `local` must already be
/// bound to the lane id: the reduction children read the same name, so binding
/// it once per program keeps one spelling of "which lane am I".
#[must_use]
pub(crate) fn for_each_index(count: u32, tile: u32, var: &str, body: Vec<Node>) -> Node {
    let chunk = format!("{var}_chunk");
    Node::loop_for(
        chunk.clone(),
        Expr::u32(0),
        Expr::u32(chunks(count, tile)),
        vec![
            Node::let_bind(
                var,
                Expr::add(
                    Expr::var("local"),
                    Expr::mul(Expr::var(chunk), Expr::u32(tile)),
                ),
            ),
            Node::if_then(Expr::lt(Expr::var(var), Expr::u32(count)), body),
        ],
    )
}

/// The numeric kind of an argmax key.
///
/// A key kind decides two things and nothing else: the neutral a lane starts
/// from, and which reduction child collapses the lane partials. Both callers
/// score non-negative keys, so the neutral is zero and an index outside the
/// space never wins.
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
///
/// The result is the pair a sequential scan would have produced. A scan that
/// keeps a strictly greater key keeps the first index attaining the maximum, so
/// the reduction takes the maximum key and then the smallest index whose key
/// equals it, which is the same answer for every input including a space whose
/// keys are all equal.
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
    ///
    /// `key` is called once per pass, so a caller that reads a buffer pays for
    /// the load twice rather than for a second scratch buffer to remember it in.
    /// The keys are non-negative by contract of the callers here, so zero is the
    /// neutral and an index the space does not contain never wins.
    ///
    /// Only the first workgroup writes the scratch, matching the scope the
    /// reduction children reduce under. Every other workgroup falls through to
    /// the same barriers and reads a pivot it does not act on, so the caller
    /// guards its shared-buffer writes the same way.
    pub(crate) fn nodes<F>(&self, key: F) -> Vec<Node>
    where
        F: Fn(Expr) -> Expr,
    {
        let scope = WorkgroupReductionScope::FirstWorkgroup;
        let mut nodes = vec![
            Node::if_then(
                Expr::is_first_workgroup(),
                vec![
                    Node::store(
                        self.key_scratch,
                        Expr::var("local"),
                        self.key_kind.neutral(),
                    ),
                    Node::store(self.index_scratch, Expr::var("local"), Expr::u32(u32::MAX)),
                ],
            ),
            Node::barrier(),
            Node::if_then(
                Expr::is_first_workgroup(),
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
            Node::barrier(),
        ];
        nodes.push(
            self.key_kind
                .max_child(self.op_id, self.tile, self.key_scratch, scope),
        );
        nodes.push(Node::if_then(
            Expr::is_first_workgroup(),
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
        nodes.push(Node::barrier());
        nodes.push(min_u32_child(
            self.op_id,
            self.tile,
            self.index_scratch,
            scope,
        ));
        nodes
    }
}
