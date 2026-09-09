//! Immutable hash-consed arenas and canonical interners for compiler substrate data.
//!
//! Provides thread-safe interners and arenas for strings, types, constants,
//! layouts, expressions, nodes, and logical regions, guaranteeing structural
//! sharing and O(1) identity comparisons.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[inline]
fn arena_read<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
) -> RwLockReadGuard<'a, T> {
    match crate::failure_domain::govern_rwlock_read(
        rwlock,
        owner,
        state,
        crate::failure_domain::RecoveryClass::InvariantViolation,
    ) {
        Ok(guard) => guard,
        Err(_) => crate::failure_domain::process_fatal_poison(owner, state),
    }
}

#[inline]
fn arena_write<'a, T>(
    rwlock: &'a RwLock<T>,
    owner: &'static str,
    state: &'static str,
) -> RwLockWriteGuard<'a, T> {
    match crate::failure_domain::govern_rwlock_write(
        rwlock,
        owner,
        state,
        crate::failure_domain::RecoveryClass::InvariantViolation,
    ) {
        Ok(guard) => guard,
        Err(_) => crate::failure_domain::process_fatal_poison(owner, state),
    }
}
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::ids::{
    ExprId, InternedConstId, InternedLayoutId, InternedStringId, InternedTypeId, NodeId, RegionId,
};
use crate::ir::{DataType, Expr, Node};
use crate::logical::LogicalRegion;

/// Thread-safe canonical string interner.
#[derive(Debug, Default)]
pub struct StringInterner {
    map: RwLock<FxHashMap<Arc<str>, InternedStringId>>,
    strings: RwLock<Vec<Arc<str>>>,
    allocated_bytes: AtomicUsize,
}

impl StringInterner {
    /// Create an empty string interner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string slice and return its stable typed identifier.
    pub fn intern(&self, text: &str) -> InternedStringId {
        {
            let read_guard = arena_read(&self.map, "StringInterner", "map");
            if let Some(&id) = read_guard.get(text) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "StringInterner", "map");
        if let Some(&id) = write_guard.get(text) {
            return id;
        }

        let mut strings_guard = arena_write(&self.strings, "StringInterner", "strings");
        let id = InternedStringId(strings_guard.len() as u32);
        let arc_str: Arc<str> = text.into();
        self.allocated_bytes.fetch_add(
            text.len() + std::mem::size_of::<Arc<str>>(),
            Ordering::Relaxed,
        );
        write_guard.insert(Arc::clone(&arc_str), id);
        strings_guard.push(arc_str);
        id
    }

    /// Resolve an interned string identifier to its shared string reference.
    pub fn lookup(&self, id: InternedStringId) -> Option<Arc<str>> {
        let strings_guard = arena_read(&self.strings, "StringInterner", "strings");
        strings_guard.get(id.0 as usize).cloned()
    }

    /// Total number of unique interned strings.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.strings, "StringInterner", "strings").len()
    }

    /// Whether the interner contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by interned strings.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Canonical representation of an interned type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanonicalType {
    /// Scalar primitive data type.
    Scalar(DataType),
    /// Vector type with primitive element type and lane count.
    Vector {
        /// Element data type.
        elem: DataType,
        /// Vector width (lanes).
        lanes: u32,
    },
    /// Tensor type with element type and dimension ranks.
    Tensor {
        /// Element data type.
        elem: DataType,
        /// Static or bounded rank extents.
        shape: Vec<u64>,
    },
    /// Composite record type.
    Composite {
        /// Type name identifier.
        name: InternedStringId,
        /// Field types.
        fields: Vec<InternedTypeId>,
    },
}

/// Thread-safe canonical type interner.
#[derive(Debug, Default)]
pub struct TypeInterner {
    map: RwLock<FxHashMap<CanonicalType, InternedTypeId>>,
    types: RwLock<Vec<CanonicalType>>,
    allocated_bytes: AtomicUsize,
}

impl TypeInterner {
    /// Create an empty type interner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a canonical type description.
    pub fn intern(&self, ty: CanonicalType) -> InternedTypeId {
        {
            let read_guard = arena_read(&self.map, "TypeInterner", "map");
            if let Some(&id) = read_guard.get(&ty) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "TypeInterner", "map");
        if let Some(&id) = write_guard.get(&ty) {
            return id;
        }

        let mut types_guard = arena_write(&self.types, "TypeInterner", "types");
        let id = InternedTypeId(types_guard.len() as u32);
        self.allocated_bytes
            .fetch_add(std::mem::size_of::<CanonicalType>(), Ordering::Relaxed);
        write_guard.insert(ty.clone(), id);
        types_guard.push(ty);
        id
    }

    /// Intern a scalar primitive data type.
    pub fn intern_scalar(&self, dtype: DataType) -> InternedTypeId {
        self.intern(CanonicalType::Scalar(dtype))
    }

    /// Resolve an interned type identifier.
    pub fn lookup(&self, id: InternedTypeId) -> Option<CanonicalType> {
        let types_guard = arena_read(&self.types, "TypeInterner", "types");
        types_guard.get(id.0 as usize).cloned()
    }

    /// Total number of interned types.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.types, "TypeInterner", "types").len()
    }

    /// Whether the interner contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by interned types.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Canonical representation of an interned constant literal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanonicalConst {
    /// 32-bit unsigned integer literal.
    U32(u32),
    /// 64-bit unsigned integer literal.
    U64(u64),
    /// 32-bit signed integer literal.
    I32(i32),
    /// 64-bit signed integer literal.
    I64(i64),
    /// 32-bit floating point literal (raw bits).
    F32(u32),
    /// 64-bit floating point literal (raw bits).
    F64(u64),
    /// Boolean literal.
    Bool(bool),
    /// Raw byte sequence.
    Blob(Arc<[u8]>),
}

/// Thread-safe canonical constant interner.
#[derive(Debug, Default)]
pub struct ConstInterner {
    map: RwLock<FxHashMap<CanonicalConst, InternedConstId>>,
    constants: RwLock<Vec<CanonicalConst>>,
    allocated_bytes: AtomicUsize,
}

impl ConstInterner {
    /// Create an empty constant interner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a constant value.
    pub fn intern(&self, val: CanonicalConst) -> InternedConstId {
        {
            let read_guard = arena_read(&self.map, "ConstInterner", "map");
            if let Some(&id) = read_guard.get(&val) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "ConstInterner", "map");
        if let Some(&id) = write_guard.get(&val) {
            return id;
        }

        let mut consts_guard = arena_write(&self.constants, "ConstInterner", "constants");
        let id = InternedConstId(consts_guard.len() as u32);
        self.allocated_bytes
            .fetch_add(std::mem::size_of::<CanonicalConst>(), Ordering::Relaxed);
        write_guard.insert(val.clone(), id);
        consts_guard.push(val);
        id
    }

    /// Resolve an interned constant identifier.
    pub fn lookup(&self, id: InternedConstId) -> Option<CanonicalConst> {
        let consts_guard = arena_read(&self.constants, "ConstInterner", "constants");
        consts_guard.get(id.0 as usize).cloned()
    }

    /// Total number of interned constants.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.constants, "ConstInterner", "constants").len()
    }

    /// Whether the interner contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by interned constants.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Canonical representation of an interned tensor or buffer layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalLayout {
    /// Physical storage strides per dimension.
    pub strides: Vec<u64>,
    /// Alignment requirement in bytes.
    pub alignment_bytes: u64,
    /// Row or plane pitch in bytes.
    pub pitch_bytes: Option<u64>,
}

/// Thread-safe canonical layout interner.
#[derive(Debug, Default)]
pub struct LayoutInterner {
    map: RwLock<FxHashMap<CanonicalLayout, InternedLayoutId>>,
    layouts: RwLock<Vec<CanonicalLayout>>,
    allocated_bytes: AtomicUsize,
}

impl LayoutInterner {
    /// Create an empty layout interner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a canonical layout.
    pub fn intern(&self, layout: CanonicalLayout) -> InternedLayoutId {
        {
            let read_guard = arena_read(&self.map, "LayoutInterner", "map");
            if let Some(&id) = read_guard.get(&layout) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "LayoutInterner", "map");
        if let Some(&id) = write_guard.get(&layout) {
            return id;
        }

        let mut layouts_guard = arena_write(&self.layouts, "LayoutInterner", "layouts");
        let id = InternedLayoutId(layouts_guard.len() as u32);
        self.allocated_bytes.fetch_add(
            std::mem::size_of::<CanonicalLayout>() + layout.strides.len() * 8,
            Ordering::Relaxed,
        );
        write_guard.insert(layout.clone(), id);
        layouts_guard.push(layout);
        id
    }

    /// Resolve an interned layout identifier.
    pub fn lookup(&self, id: InternedLayoutId) -> Option<CanonicalLayout> {
        let layouts_guard = arena_read(&self.layouts, "LayoutInterner", "layouts");
        layouts_guard.get(id.0 as usize).cloned()
    }

    /// Total number of interned layouts.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.layouts, "LayoutInterner", "layouts").len()
    }

    /// Whether the interner contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by interned layouts.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

fn compute_expr_digest(expr: &Expr) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre::substrate::expr::v1\0");
    hasher.update(format!("{expr:?}").as_bytes());
    *hasher.finalize().as_bytes()
}

/// Thread-safe hash-consed expression arena for immutable structural sharing.
#[derive(Debug, Default)]
pub struct ExprArena {
    map: RwLock<FxHashMap<[u8; 32], ExprId>>,
    exprs: RwLock<Vec<Arc<Expr>>>,
    allocated_bytes: AtomicUsize,
}

impl ExprArena {
    /// Create an empty expression arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an expression into the hash-consed arena.
    pub fn intern(&self, expr: Expr) -> ExprId {
        let digest = compute_expr_digest(&expr);
        {
            let read_guard = arena_read(&self.map, "ExprArena", "map");
            if let Some(&id) = read_guard.get(&digest) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "ExprArena", "map");
        if let Some(&id) = write_guard.get(&digest) {
            return id;
        }

        let mut exprs_guard = arena_write(&self.exprs, "ExprArena", "exprs");
        let id = ExprId(exprs_guard.len() as u32);
        let arc_expr = Arc::new(expr);
        self.allocated_bytes
            .fetch_add(std::mem::size_of::<Expr>(), Ordering::Relaxed);
        write_guard.insert(digest, id);
        exprs_guard.push(arc_expr);
        id
    }

    /// Resolve an expression identifier to its shared immutable expression.
    pub fn lookup(&self, id: ExprId) -> Option<Arc<Expr>> {
        let exprs_guard = arena_read(&self.exprs, "ExprArena", "exprs");
        exprs_guard.get(id.0 as usize).cloned()
    }

    /// Total number of unique expressions in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.exprs, "ExprArena", "exprs").len()
    }

    /// Whether the arena contains no expressions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by the expression arena.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

fn compute_node_digest(node: &Node) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre::substrate::node::v1\0");
    hasher.update(format!("{node:?}").as_bytes());
    *hasher.finalize().as_bytes()
}

/// Thread-safe hash-consed node arena for immutable AST statements.
#[derive(Debug, Default)]
pub struct NodeArena {
    map: RwLock<FxHashMap<[u8; 32], NodeId>>,
    nodes: RwLock<Vec<Arc<Node>>>,
    allocated_bytes: AtomicUsize,
}

impl NodeArena {
    /// Create an empty node arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a node into the hash-consed arena.
    pub fn intern(&self, node: Node) -> NodeId {
        let digest = compute_node_digest(&node);
        {
            let read_guard = arena_read(&self.map, "NodeArena", "map");
            if let Some(&id) = read_guard.get(&digest) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "NodeArena", "map");
        if let Some(&id) = write_guard.get(&digest) {
            return id;
        }

        let mut nodes_guard = arena_write(&self.nodes, "NodeArena", "nodes");
        let id = NodeId(nodes_guard.len() as u32);
        let arc_node = Arc::new(node);
        self.allocated_bytes
            .fetch_add(std::mem::size_of::<Node>(), Ordering::Relaxed);
        write_guard.insert(digest, id);
        nodes_guard.push(arc_node);
        id
    }

    /// Resolve a node identifier to its shared immutable node.
    pub fn lookup(&self, id: NodeId) -> Option<Arc<Node>> {
        let nodes_guard = arena_read(&self.nodes, "NodeArena", "nodes");
        nodes_guard.get(id.0 as usize).cloned()
    }

    /// Total number of unique nodes in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.nodes, "NodeArena", "nodes").len()
    }

    /// Whether the arena contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by the node arena.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Thread-safe hash-consed region arena for structurally shared logical regions.
#[derive(Debug, Default)]
pub struct RegionArena {
    map: RwLock<FxHashMap<LogicalRegion, RegionId>>,
    regions: RwLock<Vec<Arc<LogicalRegion>>>,
    allocated_bytes: AtomicUsize,
}

impl RegionArena {
    /// Create an empty region arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a logical region into the hash-consed arena.
    pub fn intern(&self, region: LogicalRegion) -> RegionId {
        {
            let read_guard = arena_read(&self.map, "RegionArena", "map");
            if let Some(&id) = read_guard.get(&region) {
                return id;
            }
        }

        let mut write_guard = arena_write(&self.map, "RegionArena", "map");
        if let Some(&id) = write_guard.get(&region) {
            return id;
        }

        let mut regions_guard = arena_write(&self.regions, "RegionArena", "regions");
        let id = RegionId(regions_guard.len() as u32);
        let arc_region = Arc::new(region.clone());
        self.allocated_bytes
            .fetch_add(std::mem::size_of::<LogicalRegion>(), Ordering::Relaxed);
        write_guard.insert(region, id);
        regions_guard.push(arc_region);
        id
    }

    /// Resolve a region identifier to its shared immutable logical region.
    pub fn lookup(&self, id: RegionId) -> Option<Arc<LogicalRegion>> {
        let regions_guard = arena_read(&self.regions, "RegionArena", "regions");
        regions_guard.get(id.0 as usize).cloned()
    }

    /// Total number of unique regions in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        arena_read(&self.regions, "RegionArena", "regions").len()
    }

    /// Whether the arena contains no regions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory allocated by the region arena.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Unified substrate arena consolidating all canonical interners and hash-consed arenas.
#[derive(Debug, Default)]
pub struct SubstrateArena {
    /// String and identifier interner.
    pub strings: StringInterner,
    /// Type system canonical interner.
    pub types: TypeInterner,
    /// Constant literals interner.
    pub constants: ConstInterner,
    /// Memory layout interner.
    pub layouts: LayoutInterner,
    /// Expression hash-consed arena.
    pub exprs: ExprArena,
    /// Node AST hash-consed arena.
    pub nodes: NodeArena,
    /// Logical region hash-consed arena.
    pub regions: RegionArena,
}

impl SubstrateArena {
    /// Create a new shared substrate arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total memory allocated across all interners and arenas in bytes.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> usize {
        self.strings.allocated_bytes()
            + self.types.allocated_bytes()
            + self.constants.allocated_bytes()
            + self.layouts.allocated_bytes()
            + self.exprs.allocated_bytes()
            + self.nodes.allocated_bytes()
            + self.regions.allocated_bytes()
    }
}
