use crate::facts::{DataflowEdge, PlanningFacts};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CandidatePlan {
    pub(crate) node_groups: Vec<u32>,
    pub(crate) fused_edges: Vec<DataflowEdge>,
    /// Launch width this candidate proposes for every group whose members all
    /// tolerate one, or `None` to launch every group at its declared width.
    pub(crate) workgroup_width: Option<u32>,
}

impl CandidatePlan {
    pub(crate) fn baseline(node_count: usize) -> Self {
        Self {
            node_groups: (0..node_count)
                .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
                .collect(),
            fused_edges: Vec::new(),
            workgroup_width: None,
        }
    }

    pub(crate) fn from_edges(node_count: usize, edges: &[DataflowEdge]) -> Self {
        let mut parent: Vec<usize> = (0..node_count).collect();
        for edge in edges {
            let from = edge.from.0 as usize;
            let to = edge.to.0 as usize;
            if from >= node_count || to >= node_count {
                continue;
            }
            let from_root = root(&mut parent, from);
            let to_root = root(&mut parent, to);
            if from_root != to_root {
                let first = from_root.min(to_root);
                let second = from_root.max(to_root);
                parent[second] = first;
            }
        }

        let mut roots = Vec::<usize>::new();
        let mut node_groups = Vec::with_capacity(node_count);
        for node in 0..node_count {
            let root = root(&mut parent, node);
            let group = match roots.iter().position(|candidate| *candidate == root) {
                Some(group) => group,
                None => {
                    roots.push(root);
                    roots.len() - 1
                }
            };
            node_groups.push(u32::try_from(group).unwrap_or(u32::MAX));
        }
        let mut fused_edges = edges.to_vec();
        fused_edges.sort_by_key(|edge| (edge.from, edge.to, edge.value));
        fused_edges.dedup();
        Self {
            node_groups,
            fused_edges,
            workgroup_width: None,
        }
    }

    /// Same grouping launched at `width` instead of the declared widths.
    pub(crate) fn with_workgroup_width(&self, width: Option<u32>) -> Self {
        Self {
            node_groups: self.node_groups.clone(),
            fused_edges: self.fused_edges.clone(),
            workgroup_width: width,
        }
    }

    pub(crate) fn group_count(&self) -> usize {
        self.node_groups
            .iter()
            .copied()
            .max()
            .map_or(0, |group| group as usize + 1)
    }

    /// Nodes belonging to one fusion group, in node order.
    pub(crate) fn group_members(&self, group: u32) -> impl Iterator<Item = usize> + '_ {
        self.node_groups
            .iter()
            .enumerate()
            .filter(move |(_, member)| **member == group)
            .map(|(node, _)| node)
    }

    /// Workgroup dimensions this candidate launches one group with.
    ///
    /// A proposed width applies only when every member of the group tolerates
    /// one; a single member that observes its launch width holds the whole group
    /// at the declared shape, because the group emits one module.
    pub(crate) fn group_workgroup(&self, group: u32, facts: &PlanningFacts) -> [u32; 3] {
        let declared = self
            .group_members(group)
            .filter_map(|node| facts.node_declared_workgroup.get(node).copied())
            .next()
            .unwrap_or([1, 1, 1]);
        let Some(width) = self.workgroup_width else {
            return declared;
        };
        let uniform = self
            .group_members(group)
            .all(|node| facts.node_accepts_width.get(node).copied().unwrap_or(false));
        if uniform {
            [width, 1, 1]
        } else {
            declared
        }
    }

    /// Invocations per workgroup this candidate launches one group with.
    pub(crate) fn group_invocations(&self, group: u32, facts: &PlanningFacts) -> u64 {
        let workgroup = self.group_workgroup(group, facts);
        u64::from(workgroup[0])
            .saturating_mul(u64::from(workgroup[1]))
            .saturating_mul(u64::from(workgroup[2]))
            .max(1)
    }
}

fn root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}
