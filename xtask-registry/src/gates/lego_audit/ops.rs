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
    info.capabilities = format!(
        "{:?}",
        vyre_foundation::program_caps::scan(&info.program)
    );
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
    ops
}

pub(super) fn build_info(id: &'static str, program: Program) -> OpInfo {
    let tier = tier_of(id);
    let mut state = Walk::default();
    for node in program.entry() {
        walk(node, false, &mut state);
    }
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
        effects: vyre_foundation::operation::OperationEffects::default(),
        capabilities: String::new(),
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
