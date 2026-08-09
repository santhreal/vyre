use crate::facts::DataflowEdge;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CandidatePlan {
    pub(crate) node_groups: Vec<u32>,
    pub(crate) fused_edges: Vec<DataflowEdge>,
}

impl CandidatePlan {
    pub(crate) fn baseline(node_count: usize) -> Self {
        Self {
            node_groups: (0..node_count)
                .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
                .collect(),
            fused_edges: Vec::new(),
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
        }
    }

    pub(crate) fn group_count(&self) -> usize {
        self.node_groups
            .iter()
            .copied()
            .max()
            .map_or(0, |group| group as usize + 1)
    }
}

fn root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}
