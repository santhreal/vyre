//! The registered operations the audit judges, and the walk that summarizes one.
//!
//! Every check reads the same summary: how many nodes an operation owns, how
//! many it composes from a registered child, which children those are, and the
//! buffer signature and fingerprint the shape comparisons need. Building it once
//! is what lets a check be a rule over `OpInfo` rather than a second traversal
//! of the IR.

use super::*;

pub(super) const MAX_LEGO_AUDIT_SOURCE_BYTES: u64 = 2_097_152;

/// Splits one violation line into the problem and its corrective action.
///
/// Every law states the corrective action in the same line, either after `Fix:`
/// or as the final sentence, so the split keeps one finding actionable when it is
/// read on its own.
pub(super) fn violation(text: String) -> Finding {
    let body = text
        .trim_start()
        .trim_start_matches(|character: char| {
            character == '\u{2717}' || character == '\u{26a0}' || character == ' '
        })
        .to_string();
    if let Some((problem, fix)) = body.split_once(" Fix: ") {
        return Finding::new(problem.trim(), fix.trim());
    }
    match body.rsplit_once(". ") {
        Some((problem, fix)) => Finding::new(problem, fix),
        None => Finding::new(
            body,
            "extract the shared work into a registered primitive and compose it through vyre_foundation::composition::wrap_child_region",
        ),
    }
}

/// One registered op with everything the audit needs.
pub(crate) struct OpInfo {
    pub(crate) id: String,
    pub(crate) source_file: String,
    pub(crate) category: Option<String>,
    pub(crate) laws: BTreeSet<String>,
    pub(crate) semantic_version: u32,
    pub(crate) tolerance: u32,
    pub(crate) effects: vyre_foundation::operation::OperationEffects,
    pub(crate) capabilities: String,
    pub(crate) required_caps: vyre_foundation::program_caps::RequiredCapabilities,
    pub(crate) callees: BTreeSet<String>,
    // Kept for future audit passes that need to re-walk the raw IR
    // (e.g. to verify that Region source_region chains are stable
    // under re-optimization). The current fingerprint/own_nodes/
    // composed_nodes/children summary is already derived from the
    // Program up-front, so downstream prints don't re-read it.
    #[allow(dead_code)]
    pub(crate) program: Program,
    pub(crate) tier: Tier,
    pub(crate) buffer_signature: Vec<String>,
    pub(crate) fingerprint: Vec<u8>,
    pub(crate) semantic_fingerprint: [u8; 32],
    pub(crate) own_nodes: usize,
    pub(crate) composed_nodes: usize,
    pub(crate) children: BTreeSet<String>, // op_ids this op invokes via Region.source_region
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Tier {
    T2,   // vyre-primitives::hardware::*
    T2_5, // every other vyre-primitives::*
    T3,   // vyre-libs::*
    Other,
}

pub(super) fn tier_of(op_id: &str) -> Tier {
    if op_id.starts_with("vyre-primitives::hardware::") {
        Tier::T2
    } else if op_id.starts_with("vyre-primitives::") {
        Tier::T2_5
    } else if op_id.starts_with("vyre-libs::") {
        Tier::T3
    } else {
        Tier::Other
    }
}

fn build_registered_info(
    entry: vyre_foundation::operation::SemanticOperation,
    program: Program,
) -> OpInfo {
    let mut info = build_info(entry.id, program);
    info.source_file = entry.source_file.to_string();
    info.category = entry.category.map(str::to_string);
    info.laws = entry.laws.iter().map(|law| (*law).to_string()).collect();
    info.semantic_version = entry.semantic_version;
    info.tolerance = entry.tolerance();
    info.effects = vyre_foundation::operation::OperationEffects::from_program(&info.program);
    info.required_caps = vyre_foundation::program_caps::scan(&info.program);
    info.capabilities = format!("{:?}", info.required_caps);
    info
}

pub(crate) fn collect_ops(report: &mut Report) -> Vec<OpInfo> {
    let mut ops = Vec::new();
    for entry in vyre_registry_link::operation::live_operation_registry().iter() {
        match entry.program() {
            Some(program) => ops.push(build_registered_info(entry, program)),
            None => report.find(Finding::new(
                format!(
                    "registered operation `{}` provides no neutral builder, so every composition law skips it",
                    entry.id
                ),
                "register a neutral builder for it, or withdraw the registration; an audit over fewer operations than the registry holds passes for the wrong reason",
            )),
        }
    }
    solve_call_graph_transitive_effects(&mut ops);
    ops
}

fn collect_calls_expr(expr: &Expr, calls: &mut BTreeSet<String>) {
    if let Expr::Call { op_id, args } = expr {
        calls.insert(op_id.as_str().to_string());
        for arg in args {
            collect_calls_expr(arg, calls);
        }
    } else {
        for child in vyre_foundation::visit::expr_children(expr).iter() {
            collect_calls_expr(child, calls);
        }
    }
}

fn collect_calls_node(node: &Node, calls: &mut BTreeSet<String>) {
    for operand in vyre_foundation::visit::node_operands(node)
        .into_iter()
        .flatten()
    {
        collect_calls_expr(operand, calls);
    }
    for operand in vyre_foundation::visit::node_variadic_operands(node) {
        collect_calls_expr(operand, calls);
    }
    if let Node::Region {
        generator,
        source_region,
        ..
    } = node
    {
        if source_region.is_some() && generator.as_str().contains("::") {
            calls.insert(generator.as_str().to_string());
        }
    }
    for body in vyre_foundation::visit::child_bodies(node) {
        for child in body {
            collect_calls_node(child, calls);
        }
    }
}

pub(super) fn collect_callees(program: &Program) -> BTreeSet<String> {
    let mut calls = BTreeSet::new();
    for node in program.entry() {
        collect_calls_node(node, &mut calls);
    }
    calls
}

fn merge_capabilities(
    target: &mut vyre_foundation::program_caps::RequiredCapabilities,
    other: &vyre_foundation::program_caps::RequiredCapabilities,
) -> bool {
    let mut changed = false;
    if !target.subgroup_ops && other.subgroup_ops {
        target.subgroup_ops = true;
        changed = true;
    }
    if !target.f16 && other.f16 {
        target.f16 = true;
        changed = true;
    }
    if !target.bf16 && other.bf16 {
        target.bf16 = true;
        changed = true;
    }
    if !target.f64 && other.f64 {
        target.f64 = true;
        changed = true;
    }
    if !target.async_dispatch && other.async_dispatch {
        target.async_dispatch = true;
        changed = true;
    }
    if !target.indirect_dispatch && other.indirect_dispatch {
        target.indirect_dispatch = true;
        changed = true;
    }
    if !target.tensor_ops && other.tensor_ops {
        target.tensor_ops = true;
        changed = true;
    }
    if !target.trap && other.trap {
        target.trap = true;
        changed = true;
    }
    if !target.distributed_collectives && other.distributed_collectives {
        target.distributed_collectives = true;
        changed = true;
    }
    if other.local_single_rank_collectives > target.local_single_rank_collectives {
        target.local_single_rank_collectives = other.local_single_rank_collectives;
        changed = true;
    }
    if other.transport_collectives > target.transport_collectives {
        target.transport_collectives = other.transport_collectives;
        changed = true;
    }
    for i in 0..3 {
        if other.max_workgroup_size[i] > target.max_workgroup_size[i] {
            target.max_workgroup_size[i] = other.max_workgroup_size[i];
            changed = true;
        }
    }
    if other.static_storage_bytes > target.static_storage_bytes {
        target.static_storage_bytes = other.static_storage_bytes;
        changed = true;
    }
    changed
}

fn set_all_capabilities(target: &mut vyre_foundation::program_caps::RequiredCapabilities) -> bool {
    let mut changed = false;
    if !target.subgroup_ops {
        target.subgroup_ops = true;
        changed = true;
    }
    if !target.f16 {
        target.f16 = true;
        changed = true;
    }
    if !target.bf16 {
        target.bf16 = true;
        changed = true;
    }
    if !target.f64 {
        target.f64 = true;
        changed = true;
    }
    if !target.async_dispatch {
        target.async_dispatch = true;
        changed = true;
    }
    if !target.indirect_dispatch {
        target.indirect_dispatch = true;
        changed = true;
    }
    if !target.tensor_ops {
        target.tensor_ops = true;
        changed = true;
    }
    if !target.trap {
        target.trap = true;
        changed = true;
    }
    if !target.distributed_collectives {
        target.distributed_collectives = true;
        changed = true;
    }
    changed
}

fn detect_unclosed_cyclic_nodes(ops: &[OpInfo], index_map: &HashMap<String, usize>) -> Vec<bool> {
    let n = ops.len();
    let mut adj = vec![Vec::new(); n];
    let mut has_unresolved = vec![false; n];

    for (i, op) in ops.iter().enumerate() {
        for callee in &op.callees {
            if let Some(&callee_idx) = index_map.get(callee) {
                adj[i].push(callee_idx);
            } else {
                has_unresolved[i] = true;
            }
        }
    }

    // Tarjan's SCC algorithm for cycle detection
    let mut index = 0usize;
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![usize::MAX; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut is_cyclic = vec![false; n];

    fn strongconnect(
        v: usize,
        adj: &[Vec<usize>],
        index: &mut usize,
        indices: &mut [usize],
        lowlinks: &mut [usize],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        is_cyclic: &mut [bool],
    ) {
        indices[v] = *index;
        lowlinks[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &adj[v] {
            if indices[w] == usize::MAX {
                strongconnect(w, adj, index, indices, lowlinks, on_stack, stack, is_cyclic);
                lowlinks[v] = lowlinks[v].min(lowlinks[w]);
            } else if on_stack[w] {
                lowlinks[v] = lowlinks[v].min(indices[w]);
            }
        }

        if lowlinks[v] == indices[v] {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            if scc.len() > 1 {
                for &node in &scc {
                    is_cyclic[node] = true;
                }
            } else if scc.len() == 1 {
                let u = scc[0];
                if adj[u].contains(&u) {
                    is_cyclic[u] = true;
                }
            }
        }
    }

    for i in 0..n {
        if indices[i] == usize::MAX {
            strongconnect(
                i,
                &adj,
                &mut index,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut stack,
                &mut is_cyclic,
            );
        }
    }

    // Reachability: if any node can reach a cyclic or unresolved node, mark it
    let mut reaches_unclosed = vec![false; n];
    for i in 0..n {
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(i);
        visited[i] = true;

        while let Some(curr) = queue.pop_front() {
            if is_cyclic[curr] || has_unresolved[curr] {
                reaches_unclosed[i] = true;
                break;
            }
            for &nxt in &adj[curr] {
                if !visited[nxt] {
                    visited[nxt] = true;
                    queue.push_back(nxt);
                }
            }
        }
    }

    reaches_unclosed
}

pub(crate) fn solve_call_graph_transitive_effects(ops: &mut [OpInfo]) {
    let index_map: HashMap<String, usize> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| (op.id.clone(), i))
        .collect();

    let unclosed = detect_unclosed_cyclic_nodes(ops, &index_map);
    for (i, &is_unclosed) in unclosed.iter().enumerate() {
        if is_unclosed {
            ops[i].effects.reads = true;
            ops[i].effects.writes = true;
            ops[i].effects.atomics = true;
            ops[i].effects.synchronizes = true;
            set_all_capabilities(&mut ops[i].required_caps);
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..ops.len() {
            let callees = ops[i].callees.clone();
            for callee_id in callees {
                if let Some(&callee_idx) = index_map.get(&callee_id) {
                    let callee_effects = ops[callee_idx].effects;
                    let callee_caps = ops[callee_idx].required_caps.clone();

                    if !ops[i].effects.reads && callee_effects.reads {
                        ops[i].effects.reads = true;
                        changed = true;
                    }
                    if !ops[i].effects.writes && callee_effects.writes {
                        ops[i].effects.writes = true;
                        changed = true;
                    }
                    if !ops[i].effects.atomics && callee_effects.atomics {
                        ops[i].effects.atomics = true;
                        changed = true;
                    }
                    if !ops[i].effects.synchronizes && callee_effects.synchronizes {
                        ops[i].effects.synchronizes = true;
                        changed = true;
                    }
                    if merge_capabilities(&mut ops[i].required_caps, &callee_caps) {
                        changed = true;
                    }
                } else {
                    if !ops[i].effects.reads {
                        ops[i].effects.reads = true;
                        changed = true;
                    }
                    if !ops[i].effects.writes {
                        ops[i].effects.writes = true;
                        changed = true;
                    }
                    if !ops[i].effects.atomics {
                        ops[i].effects.atomics = true;
                        changed = true;
                    }
                    if !ops[i].effects.synchronizes {
                        ops[i].effects.synchronizes = true;
                        changed = true;
                    }
                    if set_all_capabilities(&mut ops[i].required_caps) {
                        changed = true;
                    }
                }
            }
        }
    }

    for op in ops {
        op.capabilities = format!("{:?}", op.required_caps);
        let mut hasher = blake3::Hasher::new();
        hasher.update(&op.semantic_fingerprint);
        // Structured deterministic effects
        hasher.update(&[
            op.effects.reads as u8,
            op.effects.writes as u8,
            op.effects.atomics as u8,
            op.effects.synchronizes as u8,
        ]);
        // Structured deterministic capabilities (distinct barriers, collectives, traps, async)
        hasher.update(&[
            op.required_caps.subgroup_ops as u8,
            op.required_caps.f16 as u8,
            op.required_caps.bf16 as u8,
            op.required_caps.f64 as u8,
            op.required_caps.async_dispatch as u8,
            op.required_caps.indirect_dispatch as u8,
            op.required_caps.tensor_ops as u8,
            op.required_caps.trap as u8,
            op.required_caps.distributed_collectives as u8,
            op.required_caps.local_single_rank_collectives as u8,
            op.required_caps.transport_collectives as u8,
        ]);
        for dim in op.required_caps.max_workgroup_size {
            hasher.update(&dim.to_le_bytes());
        }
        hasher.update(&op.required_caps.static_storage_bytes.to_le_bytes());
        hasher.update(&op.semantic_version.to_le_bytes());
        hasher.update(&op.tolerance.to_le_bytes());
        for child in &op.children {
            hasher.update(child.as_bytes());
        }
        for callee in &op.callees {
            hasher.update(callee.as_bytes());
        }
        for law in &op.laws {
            hasher.update(law.as_bytes());
        }
        op.semantic_fingerprint = *hasher.finalize().as_bytes();
    }
}

pub(super) fn build_info(id: &'static str, program: Program) -> OpInfo {
    let tier = tier_of(id);
    let mut state = Walk::default();
    for node in program.entry() {
        walk(node, false, &mut state);
    }
    let callees = collect_callees(&program);
    let required_caps = vyre_foundation::program_caps::scan(&program);
    let capabilities = format!("{required_caps:?}");
    let semantic_fingerprint = program
        .clone()
        .with_entry_op_id("vyre-semantic-owner")
        .fingerprint();
    OpInfo {
        id: id.to_string(),
        source_file: String::new(),
        category: None,
        laws: BTreeSet::new(),
        semantic_version: 1,
        tolerance: 0,
        effects: vyre_foundation::operation::OperationEffects::from_program(&program),
        capabilities,
        required_caps,
        callees,
        buffer_signature: buffer_signature(&program),
        fingerprint: fingerprint_program(&program),
        semantic_fingerprint,
        own_nodes: state.own_nodes,
        composed_nodes: state.composed_nodes,
        children: state.children,
        program,
        tier,
    }
}

pub(super) fn buffer_signature(program: &Program) -> Vec<String> {
    program
        .buffers()
        .iter()
        .map(|buffer| {
            format!(
                "binding={}:access={:?}:kind={:?}:element={:?}:count={}:output={}:live_out={}:range={:?}",
                buffer.binding(),
                buffer.access(),
                buffer.kind(),
                buffer.element(),
                buffer.count(),
                buffer.is_output(),
                buffer.is_pipeline_live_out(),
                buffer.output_byte_range(),
            )
        })
        .collect()
}

#[derive(Default)]
pub(super) struct Walk {
    own_nodes: usize,
    composed_nodes: usize,
    children: BTreeSet<String>,
}

/// A node is own work unless it is, or sits inside, a region that names the
/// operation behind it. The naming region counts as composed because naming an
/// operation is what composition is: when every operation started tagging its own
/// entry region, the tag itself read as one more node of own work and lowered the
/// measured fraction of every operation that took the fix.
pub(super) fn walk(node: &Node, inside_composed: bool, state: &mut Walk) {
    let names_a_composition = matches!(
        node,
        Node::Region {
            source_region: Some(_),
            ..
        }
    );
    if inside_composed || names_a_composition {
        state.composed_nodes += 1;
    } else {
        state.own_nodes += 1;
    }
    match node {
        Node::Region {
            source_region,
            body,
            generator,
        } => {
            let now_composed = inside_composed || source_region.is_some();
            // Also count `generator` as a child op-id if it matches a
            // known op id (not all generators are children, but when
            // the generator string collides with a registered op id
            // that's a strong hint).
            if source_region.is_some() && generator.as_str().contains("::") {
                state.children.insert(generator.as_str().to_string());
            }
            for child in body.iter() {
                walk(child, now_composed, state);
            }
        }
        Node::Loop { body, .. } => {
            for child in body {
                walk(child, inside_composed, state);
            }
        }
        Node::Block(children) => {
            for child in children {
                walk(child, inside_composed, state);
            }
        }
        Node::If {
            then, otherwise, ..
        } => {
            for child in then {
                walk(child, inside_composed, state);
            }
            for child in otherwise {
                walk(child, inside_composed, state);
            }
        }
        _ => {}
    }
}

/// Two op ids share a sub-dialect when their first TWO `::` segments
/// match. `vyre-libs::math::square` and `vyre-libs::math::broadcast`
/// both live under `vyre-libs::math`, so structural similarity there
/// is expected (same shape of elementwise unary op).
pub(super) fn same_subdialect(a: &str, b: &str) -> bool {
    let a_prefix: Vec<&str> = a.split("::").take(3).collect();
    let b_prefix: Vec<&str> = b.split("::").take(3).collect();
    a_prefix.len() >= 3 && b_prefix.len() >= 3 && a_prefix[..2] == b_prefix[..2]
}

pub(super) fn read_text_bounded(path: &std::path::Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_LEGO_AUDIT_SOURCE_BYTES, "lego audit")
}

pub(super) fn workspace_root() -> Option<std::path::PathBuf> {
    Some(xtask::checkout::checkout_root())
}

/// Node count below which an operation is allowed to be flat.
///
/// A handful of nodes cannot be decomposed into anything. Checks 2, 6 and 8 all
/// measure that floor the same way and all three read it from here.
pub(super) const COMPOSITION_FLOOR_NODES: usize = 20;

/// Whether the composition rules judge `op` at all.
///
/// An internal phase generator is not a surface operation, a declared pure-IR
/// leaf has already been reviewed, and an operation under the flat-size floor
/// has nothing to decompose. A check that also narrows by tier does that itself,
/// because the three checks disagree on which tiers they judge.
pub(super) fn under_composition_rules(op: &OpInfo) -> bool {
    !is_internal_phase_op(&op.id)
        && !is_declared_tier3_leaf(&op.id)
        && op.own_nodes + op.composed_nodes >= COMPOSITION_FLOOR_NODES
}

/// Whether the unordered pair `(a, b)` is reported here for the first time.
///
/// Checks 1 and 10 both pair operations off, and both owe one row per pair
/// rather than one per direction, so the key is ordered by id before it is
/// recorded.
pub(super) fn first_report_of_pair(
    reported: &mut BTreeSet<(String, String)>,
    a: &str,
    b: &str,
) -> bool {
    let key = if a < b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    };
    reported.insert(key)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn test_op(id: &'static str, callees: &[&str]) -> OpInfo {
        let mut info = test_ops::op(id, Tier::T3, &[]);
        info.callees = callees.iter().map(|c| c.to_string()).collect();
        info
    }

    /// WHY: Section 182.9.5 requires nested writes in a callee to propagate transitively to the parent.
    #[test]
    fn nested_write_propagates_and_changes_fingerprint() {
        let mut child = test_op("child", &[]);
        child.effects.writes = true;
        let parent = test_op("parent", &["child"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent, child];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.writes,
            "parent must inherit write effect from child"
        );
        assert_ne!(
            ops[0].semantic_fingerprint, initial_fp,
            "parent fingerprint must change"
        );
    }

    /// WHY: Section 182.9.5 requires nested atomics in a callee to propagate transitively to the parent.
    #[test]
    fn nested_atomic_propagates_and_changes_fingerprint() {
        let mut child = test_op("child", &[]);
        child.effects.atomics = true;
        let parent = test_op("parent", &["child"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent, child];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.atomics,
            "parent must inherit atomic effect from child"
        );
        assert_ne!(
            ops[0].semantic_fingerprint, initial_fp,
            "parent fingerprint must change"
        );
    }

    /// WHY: Section 182.9.5 requires nested synchronization barriers in a callee to propagate transitively.
    #[test]
    fn nested_barrier_propagates_and_changes_fingerprint() {
        let mut child = test_op("child", &[]);
        child.effects.synchronizes = true;
        let parent = test_op("parent", &["child"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent, child];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.synchronizes,
            "parent must inherit barrier/synchronizes effect from child"
        );
        assert_ne!(
            ops[0].semantic_fingerprint, initial_fp,
            "parent fingerprint must change"
        );
    }

    /// WHY: Section 182.9.5 requires nested trap capability in a callee to propagate transitively.
    #[test]
    fn nested_trap_capability_propagates_and_changes_fingerprint() {
        let mut child = test_op("child", &[]);
        child.required_caps.trap = true;
        let parent = test_op("parent", &["child"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent, child];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].required_caps.trap,
            "parent must inherit trap capability from child"
        );
        assert_ne!(
            ops[0].semantic_fingerprint, initial_fp,
            "parent fingerprint must change"
        );
    }

    /// WHY: Section 182.9.5 requires nested async capability in a callee to propagate transitively.
    #[test]
    fn nested_async_capability_propagates_and_changes_fingerprint() {
        let mut child = test_op("child", &[]);
        child.required_caps.async_dispatch = true;
        let parent = test_op("parent", &["child"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent, child];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].required_caps.async_dispatch,
            "parent must inherit async capability from child"
        );
        assert_ne!(
            ops[0].semantic_fingerprint, initial_fp,
            "parent fingerprint must change"
        );
    }

    /// WHY: Section 182.9.5 requires an unresolved callee to default to the strongest applicable effects and capabilities.
    #[test]
    fn unresolved_callee_defaults_strongest_and_changes_fingerprint() {
        let parent = test_op("parent", &["unresolved::callee"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.reads
                && ops[0].effects.writes
                && ops[0].effects.atomics
                && ops[0].effects.synchronizes
        );
        assert!(
            ops[0].required_caps.trap
                && ops[0].required_caps.async_dispatch
                && ops[0].required_caps.subgroup_ops
        );
        assert_ne!(ops[0].semantic_fingerprint, initial_fp);
    }

    /// WHY: Section 182.9.5 requires self-cyclic callees (A -> A) to default to strongest effects and capabilities fail-closed.
    #[test]
    fn self_cycle_callee_defaults_strongest_and_changes_fingerprint() {
        let self_cyclic = test_op("self_cycle", &["self_cycle"]);
        let initial_fp = self_cyclic.semantic_fingerprint;

        let mut ops = vec![self_cyclic];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.reads
                && ops[0].effects.writes
                && ops[0].effects.atomics
                && ops[0].effects.synchronizes
        );
        assert!(ops[0].required_caps.trap && ops[0].required_caps.async_dispatch);
        assert_ne!(ops[0].semantic_fingerprint, initial_fp);
    }

    /// WHY: Section 182.9.5 requires multi-node cyclic callees (A -> B -> A) to default to strongest effects and capabilities fail-closed.
    #[test]
    fn multi_node_cycle_callees_default_strongest_and_changes_fingerprint() {
        let op_a = test_op("op_a", &["op_b"]);
        let op_b = test_op("op_b", &["op_a"]);
        let initial_fp_a = op_a.semantic_fingerprint;
        let initial_fp_b = op_b.semantic_fingerprint;

        let mut ops = vec![op_a, op_b];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.reads
                && ops[0].effects.writes
                && ops[0].effects.atomics
                && ops[0].effects.synchronizes
        );
        assert!(
            ops[1].effects.reads
                && ops[1].effects.writes
                && ops[1].effects.atomics
                && ops[1].effects.synchronizes
        );
        assert!(ops[0].required_caps.trap && ops[0].required_caps.async_dispatch);
        assert!(ops[1].required_caps.trap && ops[1].required_caps.async_dispatch);
        assert_ne!(ops[0].semantic_fingerprint, initial_fp_a);
        assert_ne!(ops[1].semantic_fingerprint, initial_fp_b);
    }

    /// WHY: Section 182.9.5 requires a DAG parent reaching a multi-node cycle (P -> A where A -> B -> A) to inherit strongest effects.
    #[test]
    fn dag_parent_reaching_multi_node_cycle_defaults_strongest() {
        let parent = test_op("parent", &["op_a"]);
        let op_a = test_op("op_a", &["op_b"]);
        let op_b = test_op("op_b", &["op_a"]);
        let initial_fp = parent.semantic_fingerprint;

        let mut ops = vec![parent, op_a, op_b];
        solve_call_graph_transitive_effects(&mut ops);

        assert!(
            ops[0].effects.reads
                && ops[0].effects.writes
                && ops[0].effects.atomics
                && ops[0].effects.synchronizes
        );
        assert!(ops[0].required_caps.trap && ops[0].required_caps.async_dispatch);
        assert_ne!(ops[0].semantic_fingerprint, initial_fp);
    }
}
