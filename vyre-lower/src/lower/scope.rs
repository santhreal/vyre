//! Variable-scope mechanics for neutral lowering.
//!
//! `vyre_foundation::Node` is name-based while `KernelDescriptor` is
//! result-id based. This module owns the name → result-id transition
//! rules so branch isolation and loop-carried state are explicit.

use vyre_foundation::ir::{DataType, Ident};

/// Tile binding in lowering scope carrying extents, element type, and SSA result IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TileBinding {
    pub(super) extents: Vec<u32>,
    pub(super) element: DataType,
    pub(super) results: Vec<u32>,
}

#[derive(Clone, Default)]
pub(super) struct VarScope {
    bindings: imbl::HashMap<Ident, u32>,
    tile_bindings: imbl::HashMap<Ident, TileBinding>,
}

#[derive(Clone, Default)]
pub(super) struct ScopeSnapshot {
    pub(super) bindings: imbl::HashMap<Ident, u32>,
    pub(super) tile_bindings: imbl::HashMap<Ident, TileBinding>,
}

impl ScopeSnapshot {
    pub(super) fn get(&self, name: &Ident) -> Option<&u32> {
        self.bindings.get(name)
    }

    pub(super) fn contains_key(&self, name: &Ident) -> bool {
        self.bindings.contains_key(name) || self.tile_bindings.contains_key(name)
    }
}

impl VarScope {
    pub(super) fn bind(&mut self, name: Ident, result: u32) -> Option<u32> {
        self.tile_bindings.insert(
            name.clone(),
            TileBinding {
                extents: vec![1],
                element: DataType::F32,
                results: vec![result],
            },
        );
        self.bindings.insert(name, result)
    }

    pub(super) fn bind_tile(
        &mut self,
        name: Ident,
        extents: Vec<u32>,
        element: DataType,
        results: Vec<u32>,
    ) {
        if let Some(&first) = results.first() {
            self.bindings.insert(name.clone(), first);
        }
        self.tile_bindings.insert(
            name,
            TileBinding {
                extents,
                element,
                results,
            },
        );
    }

    pub(super) fn get(&self, name: &Ident) -> Option<u32> {
        self.bindings.get(name).copied()
    }

    pub(super) fn get_tile(&self, name: &Ident) -> Option<TileBinding> {
        self.tile_bindings
            .get(name)
            .cloned()
            .or_else(|| {
                self.bindings.get(name).map(|&id| TileBinding {
                    extents: vec![1],
                    element: DataType::F32,
                    results: vec![id],
                })
            })
    }

    pub(super) fn snapshot(&self) -> ScopeSnapshot {
        ScopeSnapshot {
            bindings: self.bindings.clone(),
            tile_bindings: self.tile_bindings.clone(),
        }
    }

    pub(super) fn restore(&mut self, snapshot: ScopeSnapshot) {
        self.bindings = snapshot.bindings;
        self.tile_bindings = snapshot.tile_bindings;
    }

    pub(super) fn restore_loop_exit(
        &mut self,
        incoming: ScopeSnapshot,
        loop_exit: &ScopeSnapshot,
        loop_var: &Ident,
    ) {
        self.bindings = incoming.bindings.clone();
        self.tile_bindings = incoming.tile_bindings.clone();
        for name in incoming.bindings.keys() {
            if name == loop_var {
                continue;
            }
            if let Some(updated) = loop_exit.bindings.get(name) {
                self.bindings.insert(name.clone(), *updated);
            }
            if let Some(updated_tile) = loop_exit.tile_bindings.get(name) {
                self.tile_bindings.insert(name.clone(), updated_tile.clone());
            }
        }
        for name in incoming.tile_bindings.keys() {
            if name == loop_var {
                continue;
            }
            if let Some(updated_tile) = loop_exit.tile_bindings.get(name) {
                self.tile_bindings.insert(name.clone(), updated_tile.clone());
            }
            if let Some(updated) = loop_exit.bindings.get(name) {
                self.bindings.insert(name.clone(), *updated);
            }
        }
    }
}
