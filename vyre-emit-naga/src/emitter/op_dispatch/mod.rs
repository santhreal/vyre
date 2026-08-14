//! `KernelOp` route classification and the per-descriptor route cache.
//!
//! The dispatch itself lives in `op_routing`; every emit helper it reaches
//! lives in a sibling named after the operation family it emits.

mod binary_ops;
mod body_emission;
mod byte_element_access;
mod diagnostics;
mod op_routing;

use std::mem::{self, Discriminant};

use rustc_hash::FxHashMap;
use vyre_lower::KernelOpKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OpDispatchRoute {
    Literal,
    LocalInvocationId,
    GlobalInvocationId,
    WorkgroupId,
    SubgroupLocalId,
    SubgroupSize,
    LoopIndex,
    BufferLength,
    Load,
    Store,
    Copy,
    BinOpKind,
    UnOpKind,
    Cast,
    Select,
    Fma,
    StructuredIfThen,
    StructuredIfThenElse,
    StructuredBlock,
    StructuredForLoop,
    AsyncLoad,
    AsyncStore,
    AsyncWait,
    Trap,
    Resume,
    Barrier,
    Return,
    SubgroupBallot,
    SubgroupReduce,
    SubgroupShuffle,
    SubgroupBroadcast,
    Atomic,
    IndirectDispatch,
    MatrixMma,
    Call,
    OpaqueExpr,
    OpaqueNode,
    LoopCarrierInit,
    LoopCarrier,
    LoopCarrierEnd,
}

pub(super) struct OpDispatchRouteCache {
    routes: FxHashMap<Discriminant<KernelOpKind>, OpDispatchRoute>,
    #[cfg(test)]
    hits: usize,
}

impl Default for OpDispatchRouteCache {
    fn default() -> Self {
        Self {
            routes: FxHashMap::default(),
            #[cfg(test)]
            hits: 0,
        }
    }
}

impl OpDispatchRouteCache {
    pub(super) fn route(&mut self, kind: &KernelOpKind) -> OpDispatchRoute {
        let key = mem::discriminant(kind);
        if let Some(route) = self.routes.get(&key).copied() {
            #[cfg(test)]
            {
                self.hits += 1;
            }
            return route;
        }
        let route = classify_op_dispatch_route(kind);
        self.routes.insert(key, route);
        route
    }
}

#[cfg(test)]
pub(crate) fn op_dispatch_route_cache_probe(kinds: &[KernelOpKind]) -> (bool, usize) {
    let mut cache = OpDispatchRouteCache::default();
    let mut parity = true;
    for kind in kinds {
        let uncached = classify_op_dispatch_route(kind);
        let cached = cache.route(kind);
        parity &= uncached == cached;
    }
    (parity, cache.hits)
}

pub(super) fn classify_op_dispatch_route(kind: &KernelOpKind) -> OpDispatchRoute {
    match kind {
        KernelOpKind::Literal => OpDispatchRoute::Literal,
        KernelOpKind::LocalInvocationId => OpDispatchRoute::LocalInvocationId,
        KernelOpKind::GlobalInvocationId => OpDispatchRoute::GlobalInvocationId,
        KernelOpKind::WorkgroupId => OpDispatchRoute::WorkgroupId,
        KernelOpKind::SubgroupLocalId => OpDispatchRoute::SubgroupLocalId,
        KernelOpKind::SubgroupSize => OpDispatchRoute::SubgroupSize,
        KernelOpKind::LoopIndex { .. } => OpDispatchRoute::LoopIndex,
        KernelOpKind::BufferLength => OpDispatchRoute::BufferLength,
        KernelOpKind::LoadGlobal | KernelOpKind::LoadShared | KernelOpKind::LoadConstant => {
            OpDispatchRoute::Load
        }
        KernelOpKind::StoreGlobal | KernelOpKind::StoreShared => OpDispatchRoute::Store,
        KernelOpKind::Copy => OpDispatchRoute::Copy,
        KernelOpKind::BinOpKind(_) => OpDispatchRoute::BinOpKind,
        KernelOpKind::UnOpKind(_) => OpDispatchRoute::UnOpKind,
        KernelOpKind::Cast { .. } => OpDispatchRoute::Cast,
        KernelOpKind::Select => OpDispatchRoute::Select,
        KernelOpKind::Fma => OpDispatchRoute::Fma,
        KernelOpKind::StructuredIfThen => OpDispatchRoute::StructuredIfThen,
        KernelOpKind::StructuredIfThenElse => OpDispatchRoute::StructuredIfThenElse,
        KernelOpKind::StructuredBlock | KernelOpKind::Region { .. } => {
            OpDispatchRoute::StructuredBlock
        }
        KernelOpKind::StructuredForLoop { .. } => OpDispatchRoute::StructuredForLoop,
        KernelOpKind::AsyncLoad { .. } => OpDispatchRoute::AsyncLoad,
        KernelOpKind::AsyncStore { .. } => OpDispatchRoute::AsyncStore,
        KernelOpKind::AsyncWait { .. } => OpDispatchRoute::AsyncWait,
        KernelOpKind::Trap { .. } => OpDispatchRoute::Trap,
        KernelOpKind::Resume { .. } => OpDispatchRoute::Resume,
        KernelOpKind::Barrier { .. } => OpDispatchRoute::Barrier,
        KernelOpKind::Return => OpDispatchRoute::Return,
        KernelOpKind::SubgroupBallot => OpDispatchRoute::SubgroupBallot,
        KernelOpKind::SubgroupReduce { .. } => OpDispatchRoute::SubgroupReduce,
        KernelOpKind::SubgroupShuffle => OpDispatchRoute::SubgroupShuffle,
        KernelOpKind::SubgroupBroadcast => OpDispatchRoute::SubgroupBroadcast,
        KernelOpKind::Atomic { .. } => OpDispatchRoute::Atomic,
        KernelOpKind::IndirectDispatch { .. } => OpDispatchRoute::IndirectDispatch,
        KernelOpKind::MatrixMma { .. } => OpDispatchRoute::MatrixMma,
        KernelOpKind::Call { .. } => OpDispatchRoute::Call,
        KernelOpKind::OpaqueExpr(_) => OpDispatchRoute::OpaqueExpr,
        KernelOpKind::OpaqueNode(_) => OpDispatchRoute::OpaqueNode,
        KernelOpKind::LoopCarrierInit { .. } => OpDispatchRoute::LoopCarrierInit,
        KernelOpKind::LoopCarrier { .. } => OpDispatchRoute::LoopCarrier,
        KernelOpKind::LoopCarrierEnd { .. } => OpDispatchRoute::LoopCarrierEnd,
    }
}
