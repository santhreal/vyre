//! Transitive call-graph closure over semantic operation registrations.

use std::collections::{BTreeMap, BTreeSet};

use crate::operation::semantics::OperationEffects;
use crate::program_caps::{scan as scan_capabilities, RequiredCapabilities};
use crate::visit::collect_call_op_ids;

use super::registration::OperationRegistration;

/// Transitive call-graph closure over semantic operation registrations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallGraphClosure {
    /// Direct callees for each registered operation ID (sorted and deduplicated).
    pub direct_callees: BTreeMap<&'static str, Vec<&'static str>>,
    /// Direct (local) effects for each operation ID.
    pub direct_effects: BTreeMap<&'static str, OperationEffects>,
    /// Direct (local) capabilities for each operation ID.
    pub direct_capabilities: BTreeMap<&'static str, RequiredCapabilities>,
    /// Transitive effects solved to a fixed point for each operation ID.
    pub transitive_effects: BTreeMap<&'static str, OperationEffects>,
    /// Transitive capabilities solved to a fixed point for each operation ID.
    pub transitive_capabilities: BTreeMap<&'static str, RequiredCapabilities>,
    /// Operations participating in recursive cycles or unresolved callee chains without closed contracts.
    pub unclosed_or_cyclic: BTreeSet<&'static str>,
    /// Deterministic 64-bit fingerprint of the resolved call-graph closure.
    pub closure_identity: u64,
}

impl CallGraphClosure {
    /// Solve the call graph closure to a fixed point over a collection of registrations.
    ///
    /// Every canonical program is built at most once during this solve pass.
    ///
    /// # Panics
    ///
    /// Panics if internal call graph propagation encounters inconsistent node registration state.
    #[must_use]
    pub fn solve_from_registrations<'a, I>(registrations: I) -> Self
    where
        I: IntoIterator<Item = &'a OperationRegistration>,
    {
        let reg_map: BTreeMap<&'static str, &OperationRegistration> =
            registrations.into_iter().map(|reg| (reg.id, reg)).collect();

        let mut direct_callees: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
        let mut direct_effects: BTreeMap<&'static str, OperationEffects> = BTreeMap::new();
        let mut direct_capabilities: BTreeMap<&'static str, RequiredCapabilities> = BTreeMap::new();
        let mut unclosed_or_cyclic: BTreeSet<&'static str> = BTreeSet::new();

        // 1. Build canonical program once for each registration and extract local facts.
        for (&id, &reg) in &reg_map {
            if let Some(build) = reg.build {
                let program = (build)().with_entry_op_id(id);
                let local_eff = reg
                    .explicit_effects
                    .unwrap_or_else(|| OperationEffects::from_program(&program));
                let local_caps = reg
                    .explicit_capabilities
                    .unwrap_or_else(|| scan_capabilities(&program));
                let raw_callees = collect_call_op_ids(&program);

                let mut callees: Vec<&'static str> = Vec::with_capacity(raw_callees.len());
                for raw_callee in raw_callees {
                    if let Some((&matched_id, _)) = reg_map.get_key_value(raw_callee.as_ref()) {
                        callees.push(matched_id);
                    } else {
                        // Unregistered callee: mark caller unclosed directly.
                        unclosed_or_cyclic.insert(id);
                    }
                }
                callees.sort_unstable();
                callees.dedup();

                direct_callees.insert(id, callees);
                direct_effects.insert(id, local_eff);
                direct_capabilities.insert(id, local_caps);
            } else {
                direct_callees.insert(id, Vec::new());
                if let (Some(eff), Some(caps)) = (reg.explicit_effects, reg.explicit_capabilities) {
                    direct_effects.insert(id, eff);
                    direct_capabilities.insert(id, caps);
                } else {
                    direct_effects
                        .insert(id, reg.explicit_effects.unwrap_or(OperationEffects::ALL));
                    direct_capabilities.insert(
                        id,
                        reg.explicit_capabilities
                            .unwrap_or_else(RequiredCapabilities::all),
                    );
                    unclosed_or_cyclic.insert(id);
                }
            }
        }

        // 2. Identify unresolved callees (nodes calling an unregistered ID).
        for (&id, callees) in &direct_callees {
            for &callee in callees {
                if !reg_map.contains_key(callee) {
                    unclosed_or_cyclic.insert(id);
                }
            }
        }

        // 3. Detect recursive cycles via Tarjan's Strongly Connected Components algorithm.
        let sccs = compute_sccs(&direct_callees);
        for scc in sccs {
            let is_cycle = if scc.len() > 1 {
                true
            } else if scc.len() == 1 {
                let node = scc[0];
                direct_callees
                    .get(node)
                    .is_some_and(|callees| callees.contains(&node))
            } else {
                false
            };

            if is_cycle {
                for &node in &scc {
                    let reg = reg_map.get(node);
                    let has_contract = reg.is_some_and(|r| {
                        r.explicit_effects.is_some() && r.explicit_capabilities.is_some()
                    });
                    if !has_contract {
                        unclosed_or_cyclic.insert(node);
                        direct_effects.insert(node, OperationEffects::ALL);
                        direct_capabilities.insert(node, RequiredCapabilities::all());
                    }
                }
            }
        }

        // Ensure all unclosed/cyclic nodes default to strongest effects and capabilities.
        for &node in &unclosed_or_cyclic {
            let reg = reg_map.get(node);
            let has_contract = reg
                .is_some_and(|r| r.explicit_effects.is_some() && r.explicit_capabilities.is_some());
            if !has_contract {
                direct_effects.insert(node, OperationEffects::ALL);
                direct_capabilities.insert(node, RequiredCapabilities::all());
            }
        }

        // 4. Fixed-Point Propagation over the call graph edges.
        let mut transitive_effects = direct_effects.clone();
        let mut transitive_capabilities = direct_capabilities.clone();

        let mut changed = true;
        while changed {
            changed = false;
            for (&u, callees) in &direct_callees {
                for &v in callees {
                    let (v_eff, v_caps) = if reg_map.contains_key(v) {
                        let eff = transitive_effects
                            .get(v)
                            .copied()
                            .expect("Fix: registered callee must have transitive effect state");
                        let caps = transitive_capabilities
                            .get(v)
                            .copied()
                            .expect("Fix: registered callee must have transitive capability state");
                        (eff, caps)
                    } else {
                        (OperationEffects::ALL, RequiredCapabilities::all())
                    };

                    let u_eff = transitive_effects
                        .get_mut(u)
                        .expect("Fix: caller node in direct_callees must be registered in transitive_effects");
                    let merged_eff = u_eff.union(v_eff);
                    if merged_eff != *u_eff {
                        *u_eff = merged_eff;
                        changed = true;
                    }

                    let u_caps = transitive_capabilities
                        .get_mut(u)
                        .expect("Fix: caller node in direct_callees must be registered in transitive_capabilities");
                    let merged_caps = u_caps.join(v_caps);
                    if merged_caps != *u_caps {
                        *u_caps = merged_caps;
                        changed = true;
                    }
                }
            }
        }

        // 5. Deterministic call-graph closure identity calculation.
        let closure_identity = compute_closure_identity(
            &reg_map,
            &direct_callees,
            &transitive_effects,
            &transitive_capabilities,
            &unclosed_or_cyclic,
        );

        Self {
            direct_callees,
            direct_effects,
            direct_capabilities,
            transitive_effects,
            transitive_capabilities,
            unclosed_or_cyclic,
            closure_identity,
        }
    }

    /// Return the resolved transitive effects for an operation.
    #[must_use]
    pub fn transitive_effects(&self, id: &str) -> Option<OperationEffects> {
        self.transitive_effects.get(id).copied()
    }

    /// Return the resolved transitive required capabilities for an operation.
    #[must_use]
    pub fn transitive_capabilities(&self, id: &str) -> Option<RequiredCapabilities> {
        self.transitive_capabilities.get(id).copied()
    }

    /// Return direct callees invoked by an operation via `Expr::Call`.
    #[must_use]
    pub fn callees(&self, id: &str) -> Option<&[&'static str]> {
        self.direct_callees
            .get(id)
            .map(|callees| callees.as_slice())
    }

    /// Return whether an operation participates in an unclosed recursive cycle or unresolved call.
    #[must_use]
    pub fn is_unclosed_or_cyclic(&self, id: &str) -> bool {
        self.unclosed_or_cyclic.contains(id)
    }

    /// Return the deterministic 64-bit closure identity.
    #[must_use]
    pub fn closure_identity(&self) -> u64 {
        self.closure_identity
    }

    /// Calculate effective composite version for an operation.
    #[must_use]
    pub fn composite_version(&self, id: &str, base_version: u32) -> Option<u64> {
        let eff = self.transitive_effects(id)?;
        let caps = self.transitive_capabilities(id)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-foundation::call_graph_closure::composite_version::v1\n");
        hasher.update(id.as_bytes());
        hasher.update(&base_version.to_le_bytes());
        hasher.update(&[
            eff.reads as u8,
            eff.writes as u8,
            eff.atomics as u8,
            eff.synchronizes as u8,
        ]);
        hasher.update(&[
            caps.subgroup_ops as u8,
            caps.f16 as u8,
            caps.bf16 as u8,
            caps.f64 as u8,
            caps.async_dispatch as u8,
            caps.indirect_dispatch as u8,
            caps.tensor_ops as u8,
            caps.trap as u8,
            caps.distributed_collectives as u8,
        ]);
        hasher.update(&caps.static_storage_bytes.to_le_bytes());
        hasher.update(&self.closure_identity.to_le_bytes());
        let hash_bytes = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash_bytes.as_bytes()[..8]);
        Some(u64::from_le_bytes(bytes))
    }
}

/// Compute strongly connected components for the call graph adjacency map.
fn compute_sccs(adj: &BTreeMap<&'static str, Vec<&'static str>>) -> Vec<Vec<&'static str>> {
    struct Tarjan<'a> {
        adj: &'a BTreeMap<&'static str, Vec<&'static str>>,
        index: usize,
        indices: BTreeMap<&'static str, usize>,
        on_stack: BTreeSet<&'static str>,
        stack: Vec<&'static str>,
        sccs: Vec<Vec<&'static str>>,
    }

    impl Tarjan<'_> {
        /// Visit `v` and return its lowlink.
        ///
        /// The lowlink travels back as the return value rather than through a
        /// second map. Every read of it was a lookup for a key this function
        /// had just written, so the map could only ever answer, and eight
        /// `expect` calls said so in prose instead of in the types.
        fn strongconnect(&mut self, v: &'static str) -> usize {
            let v_index = self.index;
            self.indices.insert(v, v_index);
            self.index += 1;
            self.stack.push(v);
            self.on_stack.insert(v);

            let mut v_low = v_index;
            let adj = self.adj;
            for &w in adj.get(v).map_or(&[][..], Vec::as_slice) {
                match self.indices.get(w).copied() {
                    None if adj.contains_key(w) => v_low = v_low.min(self.strongconnect(w)),
                    Some(w_index) if self.on_stack.contains(w) => v_low = v_low.min(w_index),
                    _ => {}
                }
            }

            if v_low == v_index {
                let mut scc = Vec::new();
                while let Some(w) = self.stack.pop() {
                    self.on_stack.remove(w);
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                self.sccs.push(scc);
            }
            v_low
        }
    }

    let mut tarjan = Tarjan {
        adj,
        index: 0,
        indices: BTreeMap::new(),
        on_stack: BTreeSet::new(),
        stack: Vec::new(),
        sccs: Vec::new(),
    };

    for &node in adj.keys() {
        if !tarjan.indices.contains_key(node) {
            tarjan.strongconnect(node);
        }
    }

    tarjan.sccs
}

fn compute_closure_identity(
    reg_map: &BTreeMap<&'static str, &OperationRegistration>,
    direct_callees: &BTreeMap<&'static str, Vec<&'static str>>,
    transitive_effects: &BTreeMap<&'static str, OperationEffects>,
    transitive_capabilities: &BTreeMap<&'static str, RequiredCapabilities>,
    unclosed_or_cyclic: &BTreeSet<&'static str>,
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-foundation::call_graph_closure::v1\n");
    for (&id, reg) in reg_map {
        hasher.update(id.as_bytes());
        hasher.update(&reg.semantic_version.to_le_bytes());
        if let Some(callees) = direct_callees.get(id) {
            for callee in callees {
                hasher.update(b"->");
                hasher.update(callee.as_bytes());
            }
        }
        if let Some(eff) = transitive_effects.get(id) {
            hasher.update(&[
                eff.reads as u8,
                eff.writes as u8,
                eff.atomics as u8,
                eff.synchronizes as u8,
            ]);
        }
        if let Some(caps) = transitive_capabilities.get(id) {
            hasher.update(&[
                caps.subgroup_ops as u8,
                caps.f16 as u8,
                caps.bf16 as u8,
                caps.f64 as u8,
                caps.async_dispatch as u8,
                caps.indirect_dispatch as u8,
                caps.tensor_ops as u8,
                caps.trap as u8,
                caps.distributed_collectives as u8,
            ]);
            hasher.update(&caps.static_storage_bytes.to_le_bytes());
        }
        hasher.update(&[unclosed_or_cyclic.contains(id) as u8]);
    }
    let hash_bytes = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash_bytes.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}
