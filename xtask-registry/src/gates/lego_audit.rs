//! `cargo xtask lego-audit`  -  deeper LEGO-block enforcement.
//!
//! Gate 1 (`cargo xtask gate1`) is the floor: loops ≤ 4 AND nodes ≤ 200
//! OR composed_fraction ≥ 60%. That's table stakes. vyre's thesis is
//! composition, so the real measurement is harder.
//!
//! This xtask runs ten stricter audits:
//!
//! 1. **No-reinvention check**  -  IR fingerprint every op body; any two
//!    ops with >80% fingerprint overlap where one doesn't invoke the
//!    other get flagged as duplication.
//! 2. **Depth-of-composition**  -  Tier 3 operations must place at least 25%
//!    of their nodes under registered children or appear in the explicit
//!    reviewed pure-IR leaf set.
//! 3. **Primitive-coverage**  -  every Tier 2.5 primitive should have
//!    ≥ 2 callers. Orphans are reported as one-release adoption advisories.
//!    Synthetic catalog consumers remain hard failures and never count.
//! 4. **Cross-dialect reach-through**  -  Tier 3 dialects importing
//!    private items from sibling Tier 3 dialects. That coupling
//!    belongs in Tier 2.5; flag it.
//! 5. **Large-file advisory**  -  files over a per-file source-line
//!    review guideline are reported as notes for a split-by-responsibility
//!    review. This is advisory and never fails the audit; the hard size
//!    ceiling is the `file-size` gate.
//! 6. **Composition-chain coverage**  -  every non-leaf registered op must
//!    render at least one child Region. Explicit pure-IR leaves and tiny
//!    operations are exempt.
//! 7. **Trend**  -  compare per-op `composed_fraction` to the previous
//!    tag; fail CI if it regresses. The thesis is "composition gets
//!    deeper over time," not "stagnates."
//! 8. **Composability**  -  flag non-leaf Tier 3 islands with no upstream
//!    caller and no downstream child operations.
//! 9. **Name-stem collision**  -  ≥ 4 ops sharing a leaf-prefix stem
//!    requires a discoverable namespace, merge, or explicit reviewed family.
//! 10. **Operand-shape advisory**  -  identical fingerprint prefixes and
//!     bigram-cosine ≥ 0.55 identify registered operations for semantic review.
//!
//! Exit code 0 when every hard check passes. Advisories remain visible.
//! Intended to run in CI after Gate 1.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;

const MAX_LEGO_AUDIT_SOURCE_BYTES: u64 = 2_097_152;
const PRIMITIVE_ADMISSION_PATH: &str = "docs/optimization/PRIMITIVE_ADMISSION.toml";

#[derive(Debug, serde::Deserialize)]
struct PrimitiveAdmissionRegistry {
    schema_version: u32,
    minimum_independent_callers: usize,
    #[serde(default)]
    exception: Vec<PrimitiveAdmissionException>,
}

#[derive(Debug, serde::Deserialize)]
struct PrimitiveAdmissionException {
    family: String,
    owner: String,
    reason: String,
    review_boundary: String,
}

use vyre::ir::{Expr, Node, Program};
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

use xtask::gates::dedup_report::{
    duplicate_family_report, duplicate_report_generator_command, duplicate_report_json_path,
    duplicate_severity, registered_op_duplicate_family_id, registered_op_duplicate_subject,
    registered_op_owner_lane, structural_similarity, write_duplicate_report_json,
    DuplicateEvidence, DuplicateFamilyFinding, DuplicateFamilyReport, DuplicateSubject,
};
use xtask::gates::implementation_family::{
    known_distinct_implementation_families, reviewed_distinct_operations,
    same_implementation_family, IMPLEMENTATION_FAMILY_ROWS, REVIEWED_DISTINCT_OPERATIONS,
};
use xtask::gates::use_paths::{collect_use_paths, is_test_source_path};

const FINGERPRINT_SIM_THRESHOLD: f64 = 0.88;
/// Line count at which a source file is flagged for a split-by-responsibility
/// *review*. Crossing it is a guideline prompt, not a law and not a build
/// failure. The hard god-file ceiling (ratcheted, with a per-file exception
/// list) is enforced by the `file-size` gate.
const LARGE_FILE_ADVISORY_LINES: usize = 500;
const MIN_CALLERS_FOR_PRIMITIVE: usize = 2;

/// Entry point for the `lego-audit` subcommand.
/// Audits registered composition against the ten LEGO-block laws.
pub struct LegoAudit;

impl Gate for LegoAudit {
    fn name(&self) -> &'static str {
        "lego-audit"
    }

    fn help(&self) -> &'static str {
        "Hold registered composition to the ten composition laws; --write records the composition baseline"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.note(format!("{} op(s) audited", ops.len()));
        if ctx.write {
            write_composition_baseline(&ctx.root, &ops).map_err(|error| {
                GateError::new(
                    format!("failed to write the composition baseline: {error}"),
                    "make audits/lego-composition.tsv writable, then run the gate again",
                )
            })?;
            report.note("wrote the composition baseline");
        }
        if let Some(path) = ctx.flag("--duplicate-report-json") {
            let path = duplicate_report_json_path(
                "--duplicate-report-json",
                Some(path),
                "--duplicate-report-json requires a path",
            )
            .map_err(|error| {
                GateError::new(error, "pass a writable path after --duplicate-report-json")
            })?;
            let generator_command = duplicate_report_generator_command("lego-audit", &path);
            let duplicates = lego_duplicate_report(&ops, &generator_command);
            write_duplicate_report_json(&path, &duplicates).map_err(|error| {
                GateError::new(
                    format!(
                        "could not write the duplicate family report `{}`: {error}",
                        path.display()
                    ),
                    "pass a writable path after --duplicate-report-json",
                )
            })?;
            report.note(format!(
                "wrote the duplicate family report to {}",
                path.display()
            ));
        }

        check_0_every_exemption_is_live(&mut report, &ops);
        check_1_no_reinvention(&mut report, &ops);
        check_2_depth_of_composition(&mut report, &ops);
        check_3_primitive_coverage(&mut report, &ops);
        check_4_cross_dialect_reachthrough(&mut report);
        check_5_god_files(&mut report);
        check_6_composition_chain_coverage(&mut report, &ops);
        check_7_trend(&mut report, &ops);
        check_8_composability(&mut report, &ops);
        check_9_name_stem_collision(&mut report, &ops);
        check_10_operand_shape_duplicate(&mut report, &ops);
        Ok(report)
    }
}

/// Enforces canonical primitive adoption and its recorded exceptions.
pub struct PrimitiveAdmissionGate;

impl Gate for PrimitiveAdmissionGate {
    fn name(&self) -> &'static str {
        "primitive-admission-gate"
    }

    fn help(&self) -> &'static str {
        "Enforce canonical primitive adoption and its recorded exceptions"
    }

    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        report.note(format!("{} op(s) audited", ops.len()));
        check_3_primitive_coverage(&mut report, &ops);
        Ok(report)
    }
}

/// Splits one violation line into the problem and its corrective action.
///
/// Every law states the corrective action in the same line, either after `Fix:`
/// or as the final sentence, so the split keeps one finding actionable when it is
/// read on its own.
fn violation(text: String) -> Finding {
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

fn tier_of(op_id: &str) -> Tier {
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

pub(crate) fn collect_ops(report: &mut Report) -> Vec<OpInfo> {
    let mut ops = Vec::new();
    for entry in vyre_registry_link::operation::live_operation_registry().iter() {
        match entry.program() {
            Some(program) => ops.push(build_info(entry.id, program)),
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

fn build_info(id: &'static str, program: Program) -> OpInfo {
    let tier = tier_of(id);
    let mut state = Walk::default();
    for node in program.entry() {
        walk(node, false, &mut state);
    }
    OpInfo {
        id: id.to_string(),
        buffer_signature: buffer_signature(&program),
        fingerprint: fingerprint_program(&program),
        own_nodes: state.own_nodes,
        composed_nodes: state.composed_nodes,
        children: state.children,
        program,
        tier,
    }
}

fn buffer_signature(program: &Program) -> Vec<String> {
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
struct Walk {
    own_nodes: usize,
    composed_nodes: usize,
    children: BTreeSet<String>,
}

/// A node is own work unless it is, or sits inside, a region that names the
/// operation behind it. The naming region counts as composed because naming an
/// operation is what composition is: when every operation started tagging its own
/// entry region, the tag itself read as one more node of own work and lowered the
/// measured fraction of every operation that took the fix.
fn walk(node: &Node, inside_composed: bool, state: &mut Walk) {
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

/// Build a compact byte sequence representing the node-kind tree
/// structure of a Program's body. Two programs with identical
/// structural shape produce identical fingerprints; one-byte edits
/// produce minor differences. Used for check 1 similarity scoring.
fn fingerprint_program(program: &Program) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for node in program.entry() {
        fingerprint_node(node, &mut out);
    }
    out
}

fn fingerprint_node(node: &Node, out: &mut Vec<u8>) {
    match node {
        Node::Let { value, .. } => {
            out.push(0x01);
            fingerprint_expr(value, out);
        }
        Node::Assign { value, .. } => {
            out.push(0x02);
            fingerprint_expr(value, out);
        }
        Node::Store { index, value, .. } => {
            out.push(0x03);
            fingerprint_expr(index, out);
            fingerprint_expr(value, out);
        }
        Node::If {
            cond,
            then,
            otherwise,
            ..
        } => {
            out.push(0x04);
            fingerprint_expr(cond, out);
            out.push(0xFE);
            for n in then {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
            for n in otherwise {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
        }
        Node::Loop { from, to, body, .. } => {
            out.push(0x05);
            fingerprint_expr(from, out);
            fingerprint_expr(to, out);
            out.push(0xFE);
            for n in body {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
        }
        Node::Return => out.push(0x06),
        Node::Block(nodes) => {
            out.push(0x07);
            for n in nodes {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
        }
        Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        } => out.push(0x08),
        Node::Region {
            source_region,
            body,
            generator,
        } => {
            out.push(0x09);
            if source_region.is_some() {
                out.extend_from_slice(&fingerprint_name(generator.as_str()));
            } else {
                for n in body.iter() {
                    fingerprint_node(n, out);
                }
            }
            out.push(0xFF);
        }
        Node::IndirectDispatch { .. } => out.push(0x0A),
        Node::AsyncLoad { offset, size, .. } => {
            out.push(0x0B);
            fingerprint_expr(offset, out);
            fingerprint_expr(size, out);
        }
        Node::AsyncStore { offset, size, .. } => {
            out.push(0x0C);
            fingerprint_expr(offset, out);
            fingerprint_expr(size, out);
        }
        Node::AsyncWait { .. } => out.push(0x0D),
        Node::Trap { address, .. } => {
            out.push(0x0E);
            fingerprint_expr(address, out);
        }
        Node::Resume { .. } => out.push(0x0F),
        _ => out.push(0x80),
    }
}

fn fingerprint_expr(expr: &Expr, out: &mut Vec<u8>) {
    match expr {
        Expr::LitU32(value) => {
            out.push(0x21);
            out.push(literal_bucket_u32(*value));
        }
        Expr::LitI32(value) => {
            out.push(0x22);
            out.push(literal_bucket_u32(*value as u32));
        }
        Expr::LitF32(value) => {
            out.push(0x23);
            out.push(literal_bucket_u32(value.to_bits()));
        }
        Expr::LitBool(value) => {
            out.push(0x24);
            out.push(u8::from(*value));
        }
        Expr::Var(_) => out.push(0x25),
        Expr::Load { index, .. } => {
            out.push(0x26);
            fingerprint_expr(index, out);
        }
        Expr::BufLen { .. } => out.push(0x27),
        Expr::InvocationId { axis } => {
            out.push(0x28);
            out.push(*axis);
        }
        Expr::WorkgroupId { axis } => {
            out.push(0x29);
            out.push(*axis);
        }
        Expr::LocalId { axis } => {
            out.push(0x2A);
            out.push(*axis);
        }
        Expr::BinOp { op, left, right } => {
            out.push(0x2B);
            out.push(fingerprint_name(&format!("bin::{op:?}"))[0]);
            fingerprint_expr(left, out);
            fingerprint_expr(right, out);
        }
        Expr::UnOp { op, operand } => {
            out.push(0x2C);
            out.push(fingerprint_name(&format!("un::{op:?}"))[0]);
            fingerprint_expr(operand, out);
        }
        Expr::Call { op_id, args } => {
            out.push(0x2D);
            out.push(fingerprint_name(op_id.as_str())[0]);
            for arg in args {
                fingerprint_expr(arg, out);
            }
            out.push(0xFD);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            out.push(0x2E);
            fingerprint_expr(cond, out);
            fingerprint_expr(true_val, out);
            fingerprint_expr(false_val, out);
        }
        Expr::Cast { target, value } => {
            out.push(0x2F);
            out.push(fingerprint_name(&format!("cast::{target:?}"))[0]);
            fingerprint_expr(value, out);
        }
        Expr::Fma { a, b, c } => {
            out.push(0x30);
            fingerprint_expr(a, out);
            fingerprint_expr(b, out);
            fingerprint_expr(c, out);
        }
        Expr::Atomic {
            op,
            index,
            expected,
            value,
            ordering,
            ..
        } => {
            out.push(0x31);
            out.push(fingerprint_name(&format!("atomic::{op:?}::{ordering:?}"))[0]);
            fingerprint_expr(index, out);
            if let Some(expected) = expected.as_deref() {
                fingerprint_expr(expected, out);
            }
            out.push(0xFC);
            fingerprint_expr(value, out);
        }
        Expr::SubgroupBallot { cond } => {
            out.push(0x32);
            fingerprint_expr(cond, out);
        }
        Expr::SubgroupShuffle { value, lane } => {
            out.push(0x33);
            fingerprint_expr(value, out);
            fingerprint_expr(lane, out);
        }
        Expr::SubgroupReduce { value, .. } => {
            out.push(0x34);
            fingerprint_expr(value, out);
        }
        Expr::SubgroupLocalId => out.push(0x35),
        Expr::SubgroupSize => out.push(0x36),
        Expr::Opaque(extension) => {
            out.push(0x37);
            out.push(extension.stable_fingerprint()[0]);
        }
        _ => out.push(0xBF),
    }
}

fn literal_bucket_u32(value: u32) -> u8 {
    match value {
        0 => 0,
        1 => 1,
        2..=4 => 2,
        5..=31 => 3,
        32..=255 => 4,
        256..=4096 => 5,
        _ => 6,
    }
}

fn fingerprint_name(name: &str) -> [u8; 4] {
    let mut hash = 0x811C_9DC5u32;
    for byte in name.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash.to_le_bytes()
}

/// Check 1: flag pairs of ops with near-identical fingerprints whose
/// Region chains don't indicate one calls the other.
///
/// Uses bigram-frequency cosine similarity  -  captures ordered
/// structure, not just node-kind sets.
fn check_1_no_reinvention(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(format!(
        "[1/10] No-reinvention check (bigram cosine ≥ {FINGERPRINT_SIM_THRESHOLD:.2})"
    ));
    let pairs = no_reinvention_pairs(ops);
    for (sim, a, b) in &pairs {
        report.find(violation(format!("  ✗ reinvention: `{}` and `{}` are {:.0}% structurally similar (cross-dialect) but neither composes the other. Extract the shared body into a Tier 2.5 primitive.",
            a.id,
            b.id,
            sim * 100.0)));
    }
    if pairs.is_empty() {
        report.note("  ✓ no cross-dialect duplication".to_string());
    }
    pairs.len()
}

fn no_reinvention_pairs(ops: &[OpInfo]) -> Vec<(f64, &OpInfo, &OpInfo)> {
    let mut pairs = Vec::new();
    let mut reported: BTreeSet<(String, String)> = BTreeSet::new();
    for (i, a) in ops.iter().enumerate() {
        if is_internal_phase_op(&a.id) {
            continue;
        }
        // Only compare NON-TRIVIAL ops  -  trivial kernels share the
        // same "single invocation, loop, store" skeleton and their
        // structural similarity is expected. The audit targets ops
        // with real body content.
        if a.fingerprint.len() < 40 {
            continue;
        }
        for b in ops.iter().skip(i + 1) {
            if is_internal_phase_op(&b.id) {
                continue;
            }
            // The "extract to Tier 2.5" remedy only applies when a higher
            // tier is reinventing substrate work. Similarity among two
            // primitives may indicate a future lower-level helper, but it is
            // not a Tier-3 LEGO violation and should not fail this audit.
            if a.tier != Tier::T3 && b.tier != Tier::T3 {
                continue;
            }
            if b.fingerprint.len() < 40 {
                continue;
            }
            if a.children.contains(&b.id) || b.children.contains(&a.id) {
                continue;
            }
            if a.children.iter().any(|child| b.children.contains(child)) {
                continue;
            }
            if same_implementation_family(&a.id, &b.id)
                || known_distinct_implementation_families(&a.id, &b.id)
            {
                continue;
            }
            let sim = structural_similarity(&a.fingerprint, &b.fingerprint);
            if sim < FINGERPRINT_SIM_THRESHOLD {
                continue;
            }
            // Skip comparisons inside the same sub-dialect (math::*
            // vs math::* is often legitimate  -  same loop pattern over
            // same data type, different semantics).
            if same_subdialect(&a.id, &b.id) {
                continue;
            }
            let key = if a.id < b.id {
                (a.id.clone(), b.id.clone())
            } else {
                (b.id.clone(), a.id.clone())
            };
            if !reported.insert(key) {
                continue;
            }
            pairs.push((sim, a, b));
        }
    }
    pairs
}

/// Id fragments that mark an op as one phase of a larger composition rather than
/// an op a caller reaches for.
const PHASE_MARKERS: [&str; 6] = [
    "::hidden_projection",
    "::output_projection",
    "::softmax_stats",
    "::weight_write",
    "::v_cycle_phase",
    "::power_iteration_phase",
];

fn is_internal_phase_op(id: &str) -> bool {
    PHASE_MARKERS.iter().any(|marker| id.contains(marker))
}

/// Explicit domain-owned Category-A leaves.
///
/// These operations emit pure, backend-neutral IR but have no lower registered
/// composition unit. Keeping this list explicit prevents an arbitrary flat
/// Tier-3 operation from bypassing the depth gate.
const DECLARED_TIER3_LEAVES: [&str; 9] = [
    "vyre-libs::nn::top_k",
    "vyre-libs::math::reduce_variance",
    "vyre-libs::nn::softmax_top_k",
    "vyre-libs::nn::flash_attention",
    "vyre-libs::nn::linear_4bit_affine_grouped",
    "vyre-libs::math::fft::scale_conjugate_inverse",
    "vyre-libs::math::fft::pointwise_complex_multiply_conjugate",
    "vyre-libs::math::linalg::matmul_strassen_2x2",
    "vyre-libs::math::fft::fft_radix2",
];

fn is_declared_tier3_leaf(id: &str) -> bool {
    DECLARED_TIER3_LEAVES.contains(&id)
}

/// What a dead exemption row costs, and how to close it.
const DEAD_EXEMPTION_FIX: &str =
    "delete the row: an exemption that matches no registered op exempts nothing, and it reads as coverage of an op that no longer exists";

/// Every exemption row must match something the tree still holds.
///
/// An exemption is a rule that a named subject is judged elsewhere: a phase of a
/// larger composition, a declared pure-IR leaf, an op whose shape comes from a
/// shared builder, a pair whose shapes were read side by side and judged
/// distinct, a directory that is plumbing rather than a dialect, or a leaf-stem
/// family that is acknowledged rather than renamed. A row naming something that
/// was renamed or deleted stops exempting anything, and nothing says so: the
/// list keeps its length, the audit keeps passing, and a reader takes the row as
/// evidence the subject is covered. Two rows were already in that state when the
/// check was written, and six more turned up when the plumbing directories and
/// the leaf-stem families came under the same rule.
///
/// A stem row is held to the condition it suppresses rather than to the mere
/// existence of the stem, because a family that shrank below the collision
/// threshold no longer needs acknowledging.
fn check_0_every_exemption_is_live(report: &mut Report, ops: &[OpInfo]) {
    let libs_src = xtask::checkout::checkout_root().join("vyre-libs").join("src");
    for dir in dead_plumbing_rows(&libs_src) {
        report.find(Finding::new(
            format!("no directory `vyre-libs/src/{dir}` answers to the shared-plumbing row"),
            "delete the row: a plumbing row that matches no directory exempts nothing, and it reads as if a cross-dialect edge into it were already reviewed",
        ));
    }
    for marker in PHASE_MARKERS {
        if !ops.iter().any(|op| op.id.contains(marker)) {
            report.find(Finding::new(
                format!("no registered op id contains the phase marker `{marker}`"),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
    for leaf in DECLARED_TIER3_LEAVES {
        if !ops.iter().any(|op| op.id == leaf) {
            report.find(Finding::new(
                format!("no registered op answers to the declared Tier-3 leaf `{leaf}`"),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
    for (id, family) in IMPLEMENTATION_FAMILY_ROWS {
        if !ops.iter().any(|op| &op.id == id) {
            report.find(Finding::new(
                format!("no registered op answers to `{id}`, claimed by the implementation family `{family}`"),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
    for (one, other, _) in REVIEWED_DISTINCT_OPERATIONS {
        for id in [one, other] {
            if !ops.iter().any(|op| &op.id == id) {
                report.find(Finding::new(
                    format!("no registered op answers to `{id}`, half of the reviewed-distinct pair with `{}`", if id == one { other } else { one }),
                    DEAD_EXEMPTION_FIX,
                ));
            }
        }
    }
    let colliding = colliding_stems(ops);
    for stem in KNOWN_STEM_FAMILIES {
        if !colliding.contains_key(stem) {
            report.find(Finding::new(
                format!(
                    "the stem allowlist row `{stem}` suppresses nothing: fewer than \
                     {STEM_COLLISION_MIN} ops share it, or they already live under a `{stem}` \
                     namespace segment"
                ),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
}

/// Two op ids share a sub-dialect when their first TWO `::` segments
/// match. `vyre-libs::math::square` and `vyre-libs::math::broadcast`
/// both live under `vyre-libs::math`, so structural similarity there
/// is expected (same shape of elementwise unary op).
fn same_subdialect(a: &str, b: &str) -> bool {
    let a_prefix: Vec<&str> = a.split("::").take(3).collect();
    let b_prefix: Vec<&str> = b.split("::").take(3).collect();
    a_prefix.len() >= 3 && b_prefix.len() >= 3 && a_prefix[..2] == b_prefix[..2]
}

/// Check 2: per-op composition depth  -  for Tier 3 ops, composed_nodes
/// should dominate own_nodes.
fn check_2_depth_of_composition(report: &mut Report, ops: &[OpInfo]) -> usize {
    let mut flagged = 0usize;
    report.note("[2/10] Depth-of-composition (Tier 3 ops compose ≥25% registered child nodes or declare a pure-IR leaf)".to_string());
    for op in ops {
        if op.tier != Tier::T3 {
            continue;
        }
        if is_internal_phase_op(&op.id) {
            continue;
        }
        if is_declared_tier3_leaf(&op.id) {
            continue;
        }
        let total = op.own_nodes + op.composed_nodes;
        if total < 20 {
            continue; // Small ops are allowed to be flat.
        }
        if op.children.is_empty() || op.composed_nodes.saturating_mul(4) < total {
            report.find(violation(format!("  ✗ {} Tier 3 op has own={} composed={} and {} child op(s)  -  registered child composition is below 25%. Wrap sub-bodies in vyre_foundation::composition::wrap_child_region(<primitive_id>, ...), or explicitly classify an irreducible pure-IR leaf.",
                op.id, op.own_nodes, op.composed_nodes, op.children.len())));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note(
            "  ✓ Tier 3 ops meet registered-child depth or declare reviewed pure-IR leaves"
                .to_string(),
        );
    }
    flagged
}

fn is_synthetic_catalog_consumer(op_id: &str) -> bool {
    op_id.starts_with("vyre-libs::catalog::")
}

fn primitive_caller_counts(ops: &[OpInfo]) -> HashMap<String, usize> {
    let mut caller_counts = HashMap::new();
    for op in ops
        .iter()
        .filter(|op| !is_synthetic_catalog_consumer(&op.id))
    {
        for child in &op.children {
            if tier_of(child) == Tier::T2_5 {
                *caller_counts.entry(child.clone()).or_insert(0) += 1;
            }
        }
    }
    caller_counts
}

fn primitive_family(op_id: &str) -> Option<&str> {
    op_id
        .strip_prefix("vyre-primitives::")
        .and_then(|suffix| suffix.split("::").next())
}

fn load_primitive_admission_registry() -> Result<PrimitiveAdmissionRegistry, String> {
    let root = workspace_root().ok_or_else(|| "workspace root is unavailable".to_string())?;
    let path = root.join(PRIMITIVE_ADMISSION_PATH);
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let registry: PrimitiveAdmissionRegistry =
        toml::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))?;
    if registry.schema_version != 1 {
        return Err(format!(
            "{} must declare schema_version = 1",
            path.display()
        ));
    }
    if registry.minimum_independent_callers != MIN_CALLERS_FOR_PRIMITIVE {
        return Err(format!(
            "{} minimum_independent_callers={} disagrees with the audit floor {}",
            path.display(),
            registry.minimum_independent_callers,
            MIN_CALLERS_FOR_PRIMITIVE
        ));
    }
    Ok(registry)
}

fn validate_primitive_admission(
    report: &mut Report,
    ops: &[OpInfo],
    caller_counts: &HashMap<String, usize>,
    registry: PrimitiveAdmissionRegistry,
) -> (usize, usize) {
    let mut flagged = 0usize;
    let mut exceptions = BTreeMap::new();
    for exception in registry.exception {
        if exception.family.trim().is_empty()
            || exception.owner.trim().is_empty()
            || exception.reason.trim().is_empty()
            || exception.review_boundary.trim().is_empty()
        {
            report.find(violation(format!("  ✗ primitive admission exception `{}` has an empty family, owner, reason, or review boundary",
                exception.family)));
            flagged += 1;
            continue;
        }
        if exceptions
            .insert(exception.family.clone(), exception)
            .is_some()
        {
            report.find(violation(
                "  ✗ duplicate primitive admission exception family".to_string(),
            ));
            flagged += 1;
        }
    }

    let mut under_adopted_families = BTreeSet::new();
    for op in ops {
        if op.tier != Tier::T2_5 {
            continue;
        }
        let callers = caller_counts.get(&op.id).copied().unwrap_or(0);
        if callers >= MIN_CALLERS_FOR_PRIMITIVE {
            continue;
        }
        let Some(family) = primitive_family(&op.id) else {
            report.find(violation(format!("  ✗ {} has no canonical primitive family. Fix: use `vyre-primitives::<family>::...`.",
                op.id)));
            flagged += 1;
            continue;
        };
        under_adopted_families.insert(family.to_string());
        if !exceptions.contains_key(family) {
            report.find(violation(format!("  ✗ {} has only {} caller(s) and family `{family}` has no owner-reviewed exception in {PRIMITIVE_ADMISSION_PATH}.",
                op.id, callers)));
            flagged += 1;
        }
    }

    for family in exceptions.keys() {
        if !under_adopted_families.contains(family) {
            report.find(violation(format!("  ✗ primitive admission exception `{family}` is stale because every family member meets the caller floor")));
            flagged += 1;
        }
    }
    (flagged, under_adopted_families.len())
}

/// Check 3: every Tier 2.5 primitive needs at least two independent callers
/// or an explicit owner-reviewed exception for its current family.
fn check_3_primitive_coverage(report: &mut Report, ops: &[OpInfo]) -> usize {
    let mut flagged = 0usize;
    let mut exceptions_used = 0usize;
    report.note(format!("[3/10] Primitive coverage (Tier 2.5 primitives need ≥ {MIN_CALLERS_FOR_PRIMITIVE} callers)"));
    for op in ops
        .iter()
        .filter(|op| is_synthetic_catalog_consumer(&op.id))
    {
        report.find(violation(format!("  ✗ {} is a synthetic catalog consumer. Fix: exercise the primitive directly and record only product composition edges.",
            op.id)));
        flagged += 1;
    }

    let registry = match load_primitive_admission_registry() {
        Ok(registry) => registry,
        Err(error) => {
            report.find(violation(format!(
                "  ✗ primitive admission registry is invalid: {error}"
            )));
            return flagged + 1;
        }
    };
    let caller_counts = primitive_caller_counts(ops);
    let (admission_failures, reviewed_families) =
        validate_primitive_admission(report, ops, &caller_counts, registry);
    flagged += admission_failures;
    exceptions_used += reviewed_families;
    if flagged == 0 {
        report.note(format!("  ✓ no synthetic consumers; under-adopted primitives are covered by {exceptions_used} owner-reviewed family exception(s)"));
    }
    flagged
}

/// Enforce only the canonical primitive adoption and exception contract.
/// Check 6: composition-chain coverage  -  every non-leaf op should have
/// at least one child Region with a `source_region` pointing at
/// another registered op. Ops that explicitly declare leaf status in the
/// canonical operation contract are exempt.
fn check_6_composition_chain_coverage(report: &mut Report, ops: &[OpInfo]) -> usize {
    let mut flagged = 0usize;
    report.note("[6/10] Composition-chain coverage (non-leaf ops must have ≥ 1 child Region with source_region)".to_string());
    for op in ops {
        // Tier 2 intrinsics and Tier 2.5 primitives are leaves unless
        // their own bodies choose to compose deeper primitives.
        if matches!(op.tier, Tier::T2 | Tier::T2_5) {
            continue;
        }
        if is_internal_phase_op(&op.id) {
            continue;
        }
        if is_declared_tier3_leaf(&op.id) {
            continue;
        }
        // Tiny ops are trivially allowed to be flat.
        if op.own_nodes + op.composed_nodes < 20 {
            continue;
        }
        if op.children.is_empty() {
            report.find(violation(format!("  ⚠ {} has no registered child Regions  -  either mark it a leaf primitive or wrap inlined sub-bodies via vyre_foundation::composition::wrap_child_region(<child_op_id>, ...).",
                op.id)));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note(
            "  ✓ every non-leaf op names at least one child op in its Region chain".to_string(),
        );
    }
    flagged
}

/// Walk `vyre-libs/src/<dialect>/**/*.rs` and report every `use` that reaches
/// into `crate::<other_dialect>::...` or `vyre_libs::<other_dialect>::...`.
///
/// A dialect owns its own surface and depends downward on `vyre-primitives`. An
/// edge to a sibling is allowed, and some are necessary: a linear layer is a
/// bias-matmul. What is not allowed is naming that edge from inside the
/// importing file, three levels into the other dialect's module tree, where
/// nothing collects it. The edge is declared once at the crate root, in
/// `vyre_libs::prelude`, and imported from there.
///
/// The check is structural  -  it parses Rust use trees with `syn`, so grouped
/// imports, aliases and globs are audited consistently without relying on
/// line-oriented grep. It reads paths, not visibility: a `pub use` re-export is
/// skipped because re-exporting is how the seam itself is written, and every
/// other import is judged by where it points.
///
/// `lego-quick` asks a weaker question over the same subject: whether some
/// feature gating the importing dialect enables one gating the imported
/// dialect. Feature aggregates make that true by accident, so it passes edges
/// this check reports. The two are not one measurement with two answers, and
/// collapsing them onto the stricter rule is open work.
fn check_4_cross_dialect_reachthrough(report: &mut Report) -> usize {
    report.note("[4/10] Cross-dialect reach-through (a dialect names a sibling edge in vyre_libs::prelude, not from inside its own module tree)".to_string());
    let checkout = xtask::checkout::checkout_root();
    let libs_root = Some(checkout.join("vyre-libs").join("src"));
    let Some(libs_root) = libs_root.filter(|p| p.is_dir()) else {
        report.find(violation(
            "  ⚠ vyre-libs/src not reachable from xtask. Fix: invoke from the workspace root."
                .to_string(),
        ));
        return 0;
    };
    let (dialects, list_errors) = list_dialect_dirs(&libs_root);
    if !list_errors.is_empty() {
        for error in &list_errors {
            report.find(violation(format!("  ✗ {error}")));
        }
        return list_errors.len();
    }
    if dialects.len() < 2 {
        report.note("  ✓ fewer than 2 dialects present; nothing to cross.".to_string());
        return 0;
    }
    let mut flagged = 0usize;
    for dialect in &dialects {
        let dialect_name = dialect.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let sources = xtask::tree_walk::pruned_by(dialect, |name| {
            !xtask::tree_walk::BUILD_OUTPUT_AND_VCS.contains(&name)
        });
        for entry in sources {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    report.find(violation(format!("  ✗ {}: failed to read dialect directory: {error}. Fix: make the checked source tree fully readable.",
                        dialect.display())));
                    flagged += 1;
                    continue;
                }
            };
            let path = entry.into_path();
            if is_test_source_path(&path) {
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let text = match read_text_bounded(&path) {
                Ok(text) => text,
                Err(error) => {
                    report.find(violation(format!("  ✗ {}: failed to read Rust source for reach-through audit: {error}. Fix: make the checked source tree fully readable.",
                        path.display())));
                    flagged += 1;
                    continue;
                }
            };
            let relative = path
                .strip_prefix(&checkout)
                .unwrap_or(&path)
                .display()
                .to_string();
            let Ok(file) = syn::parse_file(&text) else {
                report.find(violation(format!("  ✗ {relative}: failed to parse Rust source for reach-through audit. Fix: keep checked-in Rust source syntactically parseable.")));
                flagged += 1;
                continue;
            };
            for use_path in collect_use_paths(&file) {
                if use_path.is_public {
                    continue;
                }
                for other in &dialects {
                    let other_name = other.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if other_name == dialect_name || other_name.is_empty() {
                        continue;
                    }
                    if use_path.imports_dialect(other_name) {
                        report.find(violation(format!(
                            "  ✗ {relative} line {}: `{}` → reaches into the `{other_name}` \
                             dialect. Fix: re-export the item from vyre-libs/src/prelude.rs and \
                             import it as `crate::prelude::…`, or hoist the shared piece into \
                             vyre-primitives.",
                            use_path.line,
                            use_path.segments.join("::")
                        )));
                        flagged += 1;
                    }
                }
            }
        }
    }
    if flagged == 0 {
        report.note(
            "  ✓ every cross-dialect edge is named in vyre_libs::prelude, not reached into"
                .to_string(),
        );
    }
    flagged
}

/// Directories under `vyre-libs/src` that are shared plumbing rather than a
/// dialect, so a dialect importing from one is not a cross-dialect edge.
///
/// Only a directory can appear here, because `list_dialect_dirs` reads the
/// dialect set from the directories under `vyre-libs/src` and a single-file
/// module is never in that set to begin with. Five rows named single-file
/// modules or a path that no longer exists and were removed for that reason:
/// `region`, `tensor_ref`, `buffer_names`, `descriptor` and `test_support`.
/// `check_0_every_exemption_is_live` holds each remaining row to an existing
/// directory, so the next row that goes the same way fails instead of reading
/// as coverage.
const SHARED_PLUMBING_DIRS: [&str; 1] = ["builder"];

/// Shared-plumbing rows that name no directory under `libs_src`.
fn dead_plumbing_rows(libs_src: &std::path::Path) -> Vec<&'static str> {
    SHARED_PLUMBING_DIRS
        .into_iter()
        .filter(|dir| !libs_src.join(dir).is_dir())
        .collect()
}

fn list_dialect_dirs(root: &std::path::Path) -> (Vec<std::path::PathBuf>, Vec<String>) {
    let read_dir = match std::fs::read_dir(root) {
        Ok(read_dir) => read_dir,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!(
                    "{}: failed to read dialect root: {error}. Fix: make vyre-libs/src fully readable.",
                    root.display()
                )],
            );
        }
    };
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for entry in read_dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                errors.push(format!(
                    "{}: failed to read dialect root entry: {error}. Fix: make vyre-libs/src fully readable.",
                    root.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if SHARED_PLUMBING_DIRS.contains(&name) {
            continue;
        }
        out.push(path);
    }
    (out, errors)
}

fn check_5_god_files(report: &mut Report) -> usize {
    report.note(format!("[5/10] Large-file advisory (files over {LARGE_FILE_ADVISORY_LINES} lines flagged for split-by-responsibility review; non-blocking)"));
    let Some(root) = workspace_root() else {
        // A missing workspace root is a real environment failure, not a
        // size advisory, so it still fails the audit.
        report.find(violation("  ✗ workspace root not reachable from xtask. Fix: run from the vyre workspace checkout.".to_string()));
        return 1;
    };

    let mut advisories = 0usize;
    let mut errors = 0usize;
    for entry in walkdir::WalkDir::new(&root)
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "target" | "target-codex" | "target-fusion-fix"
            )
        })
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.find(violation(format!("  ✗ walkdir failed while scanning source files: {error}. Fix: make the checked source tree fully readable.")));
                errors += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let text = match read_text_bounded(path) {
            Ok(text) => text,
            Err(error) => {
                report.find(violation(format!("  ✗ {} could not be read for the large-file advisory: {error}. Fix: make the checked source tree fully readable.",
                    path.strip_prefix(&root).unwrap_or(path).display())));
                errors += 1;
                continue;
            }
        };
        let line_count = text.lines().count();
        if line_count > LARGE_FILE_ADVISORY_LINES {
            report.note(format!("  • {} has {line_count} lines. Review: does this file carry more than one responsibility? If so, split it (advisory, not a failure).",
                path.strip_prefix(&root).unwrap_or(path).display()));
            advisories += 1;
        }
    }
    if advisories == 0 {
        report.note(format!(
            "  ✓ no Rust source file is over the {LARGE_FILE_ADVISORY_LINES}-line review guideline"
        ));
    } else {
        report.note(format!("  • {advisories} file(s) over the {LARGE_FILE_ADVISORY_LINES}-line guideline flagged for review (non-blocking)"));
    }
    // Only genuine I/O errors fail this check; the size guideline is advisory.
    errors
}

fn read_text_bounded(path: &std::path::Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(path, MAX_LEGO_AUDIT_SOURCE_BYTES, "lego audit")
}

const COMPOSITION_REGRESSION_EPSILON: f64 = 1.0e-9;

fn composition_regressed(old_fraction: f64, new_fraction: f64) -> bool {
    new_fraction + COMPOSITION_REGRESSION_EPSILON < old_fraction
}

fn check_7_trend(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[7/10] Composition trend (current composed_fraction must not regress from the latest available baseline)".to_string());
    let Some(root) = workspace_root() else {
        report.find(violation("  ✗ workspace root not reachable from xtask. Fix: run from the vyre workspace checkout.".to_string()));
        return 1;
    };
    let Some(tag) = previous_tag(&root) else {
        report.note("  ✓ no previous git tag found; trend check has no baseline".to_string());
        return 0;
    };
    let (previous, baseline_name) = if let Some(previous) =
        previous_composition_baseline(&root, &tag)
    {
        (previous, tag.clone())
    } else if let Some(current_baseline) = current_composition_baseline(&root) {
        report.note(format!("  • previous tag `{tag}` predates composition baselines; comparing against the checked-in bootstrap baseline"));
        (current_baseline, "audits/lego-composition.tsv".to_string())
    } else {
        report.find(violation(format!("  ✗ previous tag `{tag}` has no composition baseline and no bootstrap baseline is checked in. Fix: run `./cargo_full run --bin xtask -- lego-audit --write-baseline` and commit audits/lego-composition.tsv.")));
        return 1;
    };

    let current = composition_fractions(ops);
    let mut flagged = 0usize;
    for (op_id, old_fraction) in previous {
        let Some(new_fraction) = current.get(&op_id) else {
            continue;
        };
        if composition_regressed(old_fraction, *new_fraction) {
            report.find(violation(format!("  ✗ {op_id} composed_fraction regressed from {:.1}% to {:.1}%. Fix: restore Region composition or extract shared work to Tier 2.5.",
                old_fraction * 100.0,
                new_fraction * 100.0)));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note(format!(
            "  ✓ no composed_fraction regressions against `{baseline_name}`"
        ));
    }
    flagged
}

fn workspace_root() -> Option<std::path::PathBuf> {
    Some(xtask::checkout::checkout_root())
}

fn composition_fractions(ops: &[OpInfo]) -> BTreeMap<String, f64> {
    ops.iter()
        .map(|op| {
            let total = op.own_nodes + op.composed_nodes;
            let fraction = if total == 0 {
                1.0
            } else {
                op.composed_nodes as f64 / total as f64
            };
            (op.id.clone(), fraction)
        })
        .collect()
}

const COMPOSITION_BASELINE_PATH: &str = "audits/lego-composition.tsv";

fn write_composition_baseline(root: &std::path::Path, ops: &[OpInfo]) -> io::Result<()> {
    let path = root.join(COMPOSITION_BASELINE_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut rendered = String::from("# op_id\tcomposed_fraction\n");
    for (op_id, fraction) in composition_fractions(ops) {
        rendered.push_str(&format!("{op_id}\t{fraction:.12}\n"));
    }
    std::fs::write(&path, rendered)?;
    Ok(())
}

fn current_composition_baseline(root: &std::path::Path) -> Option<BTreeMap<String, f64>> {
    let text = std::fs::read_to_string(root.join(COMPOSITION_BASELINE_PATH)).ok()?;
    parse_composition_baseline(&text)
}

fn parse_composition_baseline(text: &str) -> Option<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let mut cols = line.split('\t');
        let Some(op_id) = cols.next().filter(|op_id| !op_id.is_empty()) else {
            continue;
        };
        let Some(fraction) = cols.next().and_then(|raw| raw.parse::<f64>().ok()) else {
            continue;
        };
        if fraction.is_finite() && (0.0..=1.0).contains(&fraction) {
            out.insert(op_id.to_string(), fraction);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn previous_tag(root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["describe", "--tags", "--abbrev=0", "HEAD^"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tag = String::from_utf8(output.stdout).ok()?;
    let tag = tag.trim();
    (!tag.is_empty()).then(|| tag.to_string())
}

fn previous_composition_baseline(
    root: &std::path::Path,
    tag: &str,
) -> Option<BTreeMap<String, f64>> {
    let output = std::process::Command::new("git")
        .args(["show", &format!("{tag}:audits/lego-composition.tsv")])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    parse_composition_baseline(&text)
}

// ============================================================
// Check 8: composability  -  flag islands.
// ============================================================
//
// An op O is an "island" when no other op composes it AND O composes
// nothing of its own. Islands fail the LEGO thesis: they are leaves
// with no upstream consumer, which means either (a) they were shipped
// on speculation and never wired in, or (b) they reinvent something a
// caller already has inline. Both cases want the user to look.
//
// Tier-2 intrinsics and Tier-2.5 primitives are terminal building blocks.
// Explicit Tier-3 leaves and tiny flat ops follow the same contract.

const ISLAND_MIN_NODES: usize = 20;

fn check_8_composability(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[8/10] Composability (every non-leaf op must be composed by ≥ 1 caller OR compose ≥ 1 child op)".to_string());
    let mut callers: HashMap<String, usize> = HashMap::new();
    for op in ops {
        for child in &op.children {
            *callers.entry(child.clone()).or_insert(0) += 1;
        }
    }
    let mut flagged = 0usize;
    for op in ops {
        if matches!(op.tier, Tier::T2 | Tier::T2_5) {
            continue;
        }
        if is_internal_phase_op(&op.id) {
            continue;
        }
        if is_declared_tier3_leaf(&op.id) {
            continue;
        }
        if op.own_nodes + op.composed_nodes < ISLAND_MIN_NODES {
            continue;
        }
        let upstream = callers.get(&op.id).copied().unwrap_or(0);
        let downstream = op.children.len();
        if upstream == 0 && downstream == 0 {
            report.find(violation(format!("  ⚠ {} is an island: {} upstream caller(s), {} child op(s), {} total nodes. Fix: either wire it as a child of a caller, or wrap its body via vyre_foundation::composition::wrap_child_region(<existing_primitive>, ...).",
                op.id,
                upstream,
                downstream,
                op.own_nodes + op.composed_nodes)));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note("  ✓ no island ops".to_string());
    }
    flagged
}

// ============================================================
// Check 9: name-stem collision  -  discoverability.
// ============================================================
//
// When N ops share a stem (`matmul`, `matmul_tiled`, `matmul_strassen`,
// `matmul_one_level`), a writer searching for "matmul" sees a wall of
// near-synonyms. The gate forces either (a) a discoverable family name
// (e.g. `matmul::tiled`, `matmul::strassen` namespacing), (b) merging
// near-duplicates, or (c) acknowledging the family with an explicit
// allowlist entry. Threshold: ≥ 4 ops sharing the leaf-prefix stem.

const STEM_COLLISION_MIN: usize = 4;

/// Stems whose family is acknowledged rather than renamed.
///
/// A row belongs here when the family is real and cannot be moved into a
/// namespace segment without renaming an op id that is already registered.
/// `dominator` is that case: `dominator_frontier` and `dominator_tree` predate
/// the two phase operations the fixpoint now composes, and `dominator::tree`
/// would rename both.
///
/// `check_0_every_exemption_is_live` holds each row to a stem that would be
/// reported without it, so a row outliving its family fails instead of reading
/// as a reviewed decision. The `opt` row was already dead when that rule landed.
const KNOWN_STEM_FAMILIES: [&str; 14] = [
    "and",
    "ast",
    "attention",
    "c",
    "c11",
    "csr",
    "dominator",
    "i4x8",
    "int4",
    "linear",
    "matmul",
    "python312",
    "quest",
    "workgroup",
];

fn is_known_stem_family(stem: &str) -> bool {
    KNOWN_STEM_FAMILIES.contains(&stem)
}

/// Stems the collision rule would report if the allowlist were empty.
///
/// A stem qualifies when at least `STEM_COLLISION_MIN` ops share it and they do
/// not already live under a namespace segment of that name, which is the family
/// being explicit by construction. Both the rule and the liveness check read
/// this, so an allowlist row is judged against the condition it suppresses
/// rather than against a second copy of it.
fn colliding_stems(ops: &[OpInfo]) -> BTreeMap<String, Vec<String>> {
    let mut buckets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for op in ops {
        if is_internal_phase_op(&op.id) {
            continue;
        }
        let leaf = op.id.rsplit("::").next().unwrap_or(&op.id);
        let stem = leaf_stem(leaf);
        if stem.is_empty() {
            continue;
        }
        buckets
            .entry(stem.to_string())
            .or_default()
            .push(op.id.clone());
    }
    buckets.retain(|stem, ids| {
        ids.len() >= STEM_COLLISION_MIN
            && !ids
                .iter()
                .all(|id| id.contains(&format!("::{stem}::")) || id.ends_with(&format!("::{stem}")))
    });
    buckets
}

fn check_9_name_stem_collision(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(format!(
        "[9/10] Name-stem collision (≥ {STEM_COLLISION_MIN} ops sharing a leaf-prefix stem)"
    ));
    let mut flagged = 0usize;
    for (stem, ids) in colliding_stems(ops) {
        if is_known_stem_family(&stem) {
            continue;
        }
        report.find(violation(format!("  ⚠ {} ops share leaf-stem `{stem}`: {}. Fix: namespace the family (e.g. `{stem}::tiled`, `{stem}::strassen`), merge near-duplicates, or add a stem allowlist entry.",
            ids.len(),
            ids.join(", "))));
        flagged += 1;
    }
    if flagged == 0 {
        report.note(format!(
            "  ✓ no leaf-stem collisions ≥ {STEM_COLLISION_MIN}"
        ));
    }
    flagged
}

/// Reduce a leaf identifier to its discoverability stem: drop the
/// trailing `_<suffix>` segment so `matmul`, `matmul_tiled`,
/// `matmul_strassen`, `matmul_one_level` all map to `matmul`.
fn leaf_stem(leaf: &str) -> &str {
    match leaf.find('_') {
        Some(idx) => &leaf[..idx],
        None => leaf,
    }
}

// ============================================================
// Check 10: unreviewed shape pair  -  catches false negatives of check 1.
// ============================================================
//
// Check 1 fires when bigram-cosine ≥ 0.88. False negatives slip when
// two ops share the same operand-type tuple AND the same fingerprint
// prefix (the first ~16 bytes of the IR-shape fingerprint, which
// captures the entry node-kind sequence). These are the "same
// problem, slightly reordered" duplicates that bigram cosine misses.
//
// WHY the score reads only past the prefix: the bucket key already
// fixes those bytes identical for every pair in the bucket, so scoring
// them again measures the key and reports similarity the check itself
// created. Two ops whose entries agree and whose remainders diverge
// scored above the threshold on the strength of the agreement that put
// them in one bucket. The remainder is the only evidence the key did
// not already spend, so the remainder is what the score reads, and a
// body that ends inside the key window carries no such evidence and is
// not compared at all.

const PREFIX_LEN: usize = 16;
const OPERAND_DUP_MIN_COSINE: f64 = 0.55;

fn check_10_operand_shape_duplicate(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(format!("[10/10] Operand-shape advisory (same fingerprint prefix, then cosine ≥ {OPERAND_DUP_MIN_COSINE:.2} past that prefix)"));
    let pairs = operand_shape_duplicate_pairs(ops);
    for (cos, a, b) in &pairs {
        report.find(violation(format!("  ⚠ unreviewed shape pair: `{}` and `{}` share their entry shape and {:.0}% cosine over the rest of the body. Fix: extract the shared body to one builder and record both in `IMPLEMENTATION_FAMILY_ROWS`, or read the two algorithms side by side and record the pair in `REVIEWED_DISTINCT_OPERATIONS` with the reason the shape cannot express.",
            a.id,
            b.id,
            cos * 100.0)));
    }
    if pairs.is_empty() {
        report.note("  ✓ every shape pair is reviewed".to_string());
    }
    0
}

/// The part of a fingerprint the bucket key did not already fix.
fn fingerprint_past_prefix(fingerprint: &[u8]) -> &[u8] {
    fingerprint.get(PREFIX_LEN..).unwrap_or(&[])
}

fn operand_shape_duplicate_pairs(ops: &[OpInfo]) -> Vec<(f64, &OpInfo, &OpInfo)> {
    let mut buckets: HashMap<Vec<u8>, Vec<&OpInfo>> = HashMap::new();
    for op in ops {
        if is_internal_phase_op(&op.id) {
            continue;
        }
        if op.fingerprint.len() < PREFIX_LEN {
            continue;
        }
        let prefix: Vec<u8> = op.fingerprint[..PREFIX_LEN].to_vec();
        buckets.entry(prefix).or_default().push(op);
    }
    let mut pairs = Vec::new();
    let mut reported: BTreeSet<(String, String)> = BTreeSet::new();
    for ops_in_bucket in buckets.values() {
        if ops_in_bucket.len() < 2 {
            continue;
        }
        for (i, a) in ops_in_bucket.iter().enumerate() {
            for b in ops_in_bucket.iter().skip(i + 1) {
                if a.children.contains(&b.id) || b.children.contains(&a.id) {
                    continue;
                }
                if same_implementation_family(&a.id, &b.id)
                    || known_distinct_implementation_families(&a.id, &b.id)
                    || reviewed_distinct_operations(&a.id, &b.id).is_some()
                {
                    continue;
                }
                if same_subdialect(&a.id, &b.id) {
                    continue;
                }
                let cos = structural_similarity(
                    fingerprint_past_prefix(&a.fingerprint),
                    fingerprint_past_prefix(&b.fingerprint),
                );
                if cos < OPERAND_DUP_MIN_COSINE {
                    continue;
                }
                let key = if a.id < b.id {
                    (a.id.clone(), b.id.clone())
                } else {
                    (b.id.clone(), a.id.clone())
                };
                if !reported.insert(key) {
                    continue;
                }
                pairs.push((cos, *a, *b));
            }
        }
    }
    pairs
}

fn lego_duplicate_report(ops: &[OpInfo], generator_command: &str) -> DuplicateFamilyReport {
    let mut families = Vec::new();
    families.extend(
        no_reinvention_pairs(ops)
            .into_iter()
            .map(|(score, left, right)| {
                lego_duplicate_family("lego-audit:no-reinvention", score, left, right)
            }),
    );
    families.extend(
        operand_shape_duplicate_pairs(ops)
            .into_iter()
            .map(|(score, left, right)| {
                lego_duplicate_family("lego-audit:operand-shape", score, left, right)
            }),
    );
    duplicate_family_report(generator_command, "registered-op-lego-audit", families)
}

fn lego_duplicate_family(
    detector: &str,
    score: f64,
    left: &OpInfo,
    right: &OpInfo,
) -> DuplicateFamilyFinding {
    DuplicateFamilyFinding {
        family_id: registered_op_duplicate_family_id(&left.id, &right.id),
        detector: detector.to_string(),
        severity: duplicate_severity(score),
        score,
        left: lego_duplicate_subject(left),
        right: lego_duplicate_subject(right),
        import_owner: if left.tier <= right.tier {
            registered_op_owner_lane(&left.id).to_string()
        } else {
            registered_op_owner_lane(&right.id).to_string()
        },
        import_target: if left.tier <= right.tier {
            left.id.clone()
        } else {
            right.id.clone()
        },
        evidence: DuplicateEvidence {
            similarity_metric: "lego-ir-structural-similarity",
            left_metric: format!(
                "tier={:?}:own_nodes={}:composed_nodes={}:fingerprint_bytes={}",
                left.tier,
                left.own_nodes,
                left.composed_nodes,
                left.fingerprint.len()
            ),
            right_metric: format!(
                "tier={:?}:own_nodes={}:composed_nodes={}:fingerprint_bytes={}",
                right.tier,
                right.own_nodes,
                right.composed_nodes,
                right.fingerprint.len()
            ),
            dedup_action: "extract_shared_tier_2_5_primitive_or_compose_existing_op",
        },
    }
}

fn lego_duplicate_subject(op: &OpInfo) -> DuplicateSubject {
    registered_op_duplicate_subject(&op.id, &op.fingerprint, op.own_nodes + op.composed_nodes)
}

#[cfg(test)]
mod dedup_contract_tests {
    use super::*;
    use std::path::PathBuf;

    fn op(id: &str, tier: Tier, children: &[&str]) -> OpInfo {
        OpInfo {
            id: id.to_string(),
            program: Program::empty(),
            tier,
            buffer_signature: Vec::new(),
            fingerprint: vec![1; 64],
            own_nodes: 1,
            composed_nodes: 0,
            children: children.iter().map(|child| (*child).to_string()).collect(),
        }
    }

    fn op_with_fingerprint(id: &str, fingerprint: Vec<u8>) -> OpInfo {
        let mut info = op(id, Tier::T3, &[]);
        info.fingerprint = fingerprint;
        info
    }

    /// WHY: the bucket key fixes the first `PREFIX_LEN` bytes identical for
    /// every pair in a bucket. Scoring those bytes again measures the key, so a
    /// pair whose bodies diverge everywhere the key did not reach used to score
    /// above the threshold on the strength of the agreement that bucketed it.
    /// This test fails the moment the score reads the whole fingerprint again:
    /// with a 16-byte shared entry and remainders that share no bigram, whole
    /// fingerprint cosine is over 0.55 and remainder cosine is 0.
    #[test]
    fn a_pair_that_agrees_only_where_the_bucket_key_reaches_is_not_a_duplicate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xB1, 0xB3, 0xB1, 0xB3, 0xB1, 0xB3, 0xB1, 0xB3]);
        let ops = vec![
            op_with_fingerprint("vyre-libs::alpha::left", left),
            op_with_fingerprint("vyre-primitives::beta::right", right),
        ];
        assert!(operand_shape_duplicate_pairs(&ops).is_empty());
    }

    /// WHY: a body that ends inside the key window leaves no evidence the key
    /// did not already spend, so it cannot be judged either way. Two four-node
    /// operations used to pair at 88% because the key had made them identical.
    #[test]
    fn a_body_that_ends_inside_the_bucket_key_is_not_compared() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let ops = vec![
            op_with_fingerprint("vyre-libs::alpha::left", entry.clone()),
            op_with_fingerprint("vyre-primitives::beta::right", entry),
        ];
        assert!(operand_shape_duplicate_pairs(&ops).is_empty());
    }

    /// WHY: the correction must keep the duplicates it was built to find. Two
    /// bodies that agree past the key still pair.
    #[test]
    fn a_pair_that_agrees_past_the_bucket_key_is_still_a_duplicate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA3]);
        let ops = vec![
            op_with_fingerprint("vyre-libs::alpha::left", left),
            op_with_fingerprint("vyre-primitives::beta::right", right),
        ];
        let pairs = operand_shape_duplicate_pairs(&ops);
        assert_eq!(pairs.len(), 1, "the pair past the key must still be found");
        assert!(pairs[0].0 >= OPERAND_DUP_MIN_COSINE);
    }

    /// IR duplicate analysis judges exactly the registrations that carry a
    /// program.
    ///
    /// WHY: `collect_ops` fingerprints a program, so a registration without one
    /// cannot be compared and has to be left out rather than fingerprinted as
    /// an empty body. Set equality is what makes this non-vacuous: dropping a
    /// program-carrying operation would shrink the analysis in silence, and
    /// admitting a signature-only one would compare it against nothing. This
    /// used to assert that a signature-only registration exists, which the
    /// design then removed: `OperationRegistry` refuses the dotted
    /// host-capability ids that were the last of them, so that assertion could
    /// only ever fail.
    #[test]
    fn ir_duplicate_analysis_judges_exactly_the_operations_that_carry_a_program() {
        let registry = vyre_registry_link::operation::live_operation_registry();
        let mut expected: Vec<&str> = registry
            .iter()
            .filter(|entry| entry.program().is_some())
            .map(|entry| entry.id)
            .collect();
        expected.sort_unstable();
        let ops = collect_ops(&mut Report::clean());
        let mut analysed: Vec<&str> = ops.iter().map(|op| op.id.as_str()).collect();
        analysed.sort_unstable();
        assert_eq!(
            analysed, expected,
            "Fix: the duplicate analysis must judge every registration that carries a program and no registration that does not"
        );
    }

    /// This test prevents generated consumer_a/consumer_b aliases from satisfying the two-caller primitive promotion rule.
    #[test]
    fn synthetic_catalog_consumers_do_not_count_as_primitive_callers() {
        let primitive_id = "vyre-primitives::math::shared_step";
        let ops = vec![
            op(primitive_id, Tier::T2_5, &[]),
            op("vyre-libs::math::real_consumer", Tier::T3, &[primitive_id]),
            op(
                "vyre-libs::catalog::math::shared_step::consumer_a",
                Tier::T3,
                &[primitive_id],
            ),
            op(
                "vyre-libs::catalog::math::shared_step::consumer_b",
                Tier::T3,
                &[primitive_id],
            ),
        ];

        assert_eq!(primitive_caller_counts(&ops).get(primitive_id), Some(&1));
    }

    /// This adversarial test reserves the complete catalog namespace so renamed synthetic aliases cannot bypass caller filtering.
    #[test]
    fn every_catalog_namespace_entry_is_synthetic() {
        assert!(is_synthetic_catalog_consumer(
            "vyre-libs::catalog::graph::frontier::production"
        ));
        assert!(!is_synthetic_catalog_consumer("vyre-libs::graph::frontier"));
    }

    /// This contract test keeps discoverability stems stable across multi-suffix operation names.
    #[test]
    fn leaf_stem_drops_first_underscore_suffix() {
        assert_eq!(leaf_stem("matmul"), "matmul");
        assert_eq!(leaf_stem("matmul_tiled"), "matmul");
        assert_eq!(leaf_stem("matmul_strassen_one_level"), "matmul");
        assert_eq!(leaf_stem("fft_radix2"), "fft");
        assert_eq!(leaf_stem(""), "");
    }

    /// This regression test keeps reviewed pure-IR leaves explicit instead of exempting every flat Tier-3 operation.
    #[test]
    fn declared_leaf_classification_is_exact() {
        assert!(is_declared_tier3_leaf("vyre-libs::nn::top_k"));
        assert!(!is_declared_tier3_leaf("vyre-libs::nn::unknown_flat_op"));
    }

    /// This policy test requires low primitive adoption to match an explicit,
    /// owner-reviewed family exception instead of disappearing into prose.
    #[test]
    fn primitive_coverage_requires_registered_family_exception() {
        let ops = collect_ops(&mut Report::clean());
        assert!(ops
            .iter()
            .any(|op| primitive_family(&op.id) == Some("math")));
        assert_eq!(check_3_primitive_coverage(&mut Report::clean(), &ops), 0);
    }

    /// A newly under-adopted family fails closed until its owner records a
    /// concrete exception or real callers meet the promotion floor.
    #[test]
    fn unregistered_primitive_family_fails_admission() {
        let ops = vec![op(
            "vyre-primitives::unreviewed::new_primitive",
            Tier::T2_5,
            &[],
        )];
        let mut exceptions = load_primitive_admission_registry().expect("registry");
        exceptions
            .exception
            .retain(|exception| exception.family == "unreviewed");
        assert_eq!(
            validate_primitive_admission(
                &mut Report::clean(),
                &ops,
                &primitive_caller_counts(&ops),
                exceptions
            )
            .0,
            1
        );
    }

    /// This adversarial test ensures synthetic catalog wrappers remain hard failures even though low adoption is advisory.
    #[test]
    fn synthetic_primitive_consumers_remain_hard_failures() {
        let mut ops = collect_ops(&mut Report::clean());
        ops.push(op(
            "vyre-libs::catalog::math::new_primitive::consumer_a",
            Tier::T3,
            &["vyre-primitives::math::new_primitive"],
        ));
        assert_eq!(check_3_primitive_coverage(&mut Report::clean(), &ops), 1);
    }

    /// This boundary test locks the minimum material composition ratio at exactly 25%.
    #[test]
    fn quarter_composed_tier3_operation_passes_depth_gate() {
        let mut composed = op(
            "vyre-libs::nn::reviewed_orchestrator",
            Tier::T3,
            &["vyre-primitives::nn::child"],
        );
        composed.own_nodes = 75;
        composed.composed_nodes = 25;
        assert_eq!(
            check_2_depth_of_composition(&mut Report::clean(), &[composed]),
            0
        );
    }

    /// This negative twin prevents a nominal child edge from hiding an almost entirely inlined Tier-3 implementation.
    #[test]
    fn below_quarter_composed_tier3_operation_fails_depth_gate() {
        let mut inlined = op(
            "vyre-libs::nn::inlined_orchestrator",
            Tier::T3,
            &["vyre-primitives::nn::child"],
        );
        inlined.own_nodes = 76;
        inlined.composed_nodes = 24;
        assert_eq!(
            check_2_depth_of_composition(&mut Report::clean(), &[inlined]),
            1
        );
    }

    /// This discoverability test preserves explicit acknowledgement of intentional operation families.
    #[test]
    fn known_stem_families_are_explicit() {
        assert!(is_known_stem_family("matmul"));
        assert!(is_known_stem_family("int4"));
        assert!(!is_known_stem_family("unreviewed"));
    }

    /// This adversarial parser test rejects malformed and out-of-range baseline rows while preserving exact valid fractions.
    #[test]
    fn composition_baseline_parser_accepts_only_bounded_finite_rows() {
        let parsed = parse_composition_baseline(
            "# op_id\tcomposed_fraction\nvalid::op\t0.25\nbad\tNaN\nhigh\t1.1\nmissing\n",
        )
        .expect("Fix: one valid composition baseline row must parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed.get("valid::op"), Some(&0.25));
    }

    /// This numeric boundary test prevents baseline serialization rounding from becoming a false composition regression.
    #[test]
    fn composition_regression_tolerates_serialization_rounding_only() {
        assert!(!composition_regressed(0.913043478261, 21.0 / 23.0));
        assert!(composition_regressed(0.913043478261, 0.90));
    }

    /// WHY: this preserves the explicit duplicate-report output path contract.
    /// The gate reads the flag off `GateCtx` and resolves it through the shared
    /// helper, so the test exercises both halves rather than a parser that no
    /// longer exists.
    #[test]
    fn duplicate_report_json_arg_accepts_path() {
        let ctx = xtask::gate::GateCtx::new(
            PathBuf::from("."),
            vec![
                "--with-repo".to_string(),
                "--duplicate-report-json".to_string(),
                "release/evidence/dedup/lego-duplicates.json".to_string(),
            ],
        );
        let resolved = duplicate_report_json_path(
            "--duplicate-report-json",
            ctx.flag("--duplicate-report-json"),
            "--duplicate-report-json requires a path",
        );
        assert_eq!(
            resolved.ok(),
            Some(PathBuf::from("release/evidence/dedup/lego-duplicates.json"))
        );
    }

    /// WHY: the shared-plumbing list is consumed by a directory filter, so a row
    /// naming a single-file module or a path that was removed is skipped by
    /// nothing and still reads as a reviewed exemption. Five of the six rows were
    /// in that state. The check runs against a directory it is handed rather than
    /// the checkout, so both directions are proved: a row with a directory behind
    /// it is live, and one without it is reported.
    ///
    /// What this does not catch: a directory that exists but is a dialect rather
    /// than plumbing. That judgement is the reviewer's and the row carries it.
    #[test]
    fn a_plumbing_row_without_a_directory_behind_it_is_dead() {
        let libs_src = tempfile::tempdir().expect("temporary vyre-libs/src");

        assert_eq!(
            dead_plumbing_rows(libs_src.path()),
            SHARED_PLUMBING_DIRS.to_vec(),
            "every row is dead against a tree that holds none of them"
        );

        for dir in SHARED_PLUMBING_DIRS {
            std::fs::create_dir(libs_src.path().join(dir)).expect("plumbing directory");
        }
        assert_eq!(
            dead_plumbing_rows(libs_src.path()),
            Vec::<&str>::new(),
            "no row is dead once every one of them names a directory"
        );

        let first = SHARED_PLUMBING_DIRS[0];
        std::fs::remove_dir(libs_src.path().join(first)).expect("remove one plumbing directory");
        assert_eq!(
            dead_plumbing_rows(libs_src.path()),
            vec![first],
            "the row whose directory went away is the one reported"
        );
    }
}
