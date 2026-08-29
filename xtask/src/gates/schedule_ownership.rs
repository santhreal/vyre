//! One crate selects the execution schedule; every other crate receives it.
//!
//! A schedule decision is a choice of geometry, persistence, topology, layout or
//! placement. `docs/CRATE_OWNERSHIP.toml` names one crate in the
//! `compiler-boundary` layer, and that crate is the only one allowed to make
//! one. The foundation states neutral facts, lowering and the concrete backends
//! realize what was selected, drivers report capabilities and measurements, and
//! the runtime submits the artifact it was handed.
//!
//! Four owners of launch geometry shipped at once before this landed: a
//! `SchedulingPolicy` in the foundation whose persistence predicates ignored
//! their arguments and answered `true`, an optimizer pass that rewrote a
//! program's workgroup size from adapter facts before the search saw it, a
//! driver tuner that widened a launch the artifact had frozen, and the search
//! itself. Three could disagree with the bytes the artifact authenticated, and a
//! launch cannot recover from a workgroup it did not declare.
//!
//! Nothing here is a list of names. The decision types are the field types of
//! the selected plan and the selected schedule, and the geometry fields are the
//! extent-typed fields of a selected phase, both read from source at run time. A
//! field added to either turns this gate red until its owner is recorded.
//!
//! Three rules over one call graph:
//!
//! 1. A function outside the compiler boundary that constructs a decision value
//!    is a finding unless it, or a caller within three levels, receives one.
//!    Constructing without receiving is choosing; constructing from a value the
//!    caller was handed is realizing.
//! 2. Writing a geometry field outside the compiler boundary is a finding,
//!    whether through an assignment or a generated setter, because the artifact
//!    froze that field and a write after the freeze runs bytes nobody
//!    authenticated. A body that forwards an extent it was handed is realizing;
//!    one that writes an extent out, or that holds a device capability record
//!    while it writes, chose the shape.
//! 3. Returning geometry from a body that takes a device capability record is a
//!    finding under the same call-graph exemption, outside the layers that
//!    report device facts. Mapping what the hardware grants onto a launch shape
//!    is a cost model, whatever the body does with the facts, and there is one
//!    cost model.
//!
//! A macro body is opaque to the walk except for `vec!`, whose elements are
//! parsed and visited. `matches!` stays opaque on purpose: its second argument
//! is a pattern, and reading a decision is not making one.
//!
//! Patterns are not constructions. `match mode { ExecutionMode::Static => .. }`
//! reads a decision, and the walk reaches expressions only, so a reader is never
//! reported as a selector.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use quote::ToTokens;
use syn::spanned::Spanned;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// How many caller levels a realization route may pass a decision down.
const CALLER_DEPTH: usize = 3;

/// The layer that owns schedule selection.
const OWNER_LAYER: &str = "compiler-boundary";

/// The layer whose product is fixtures rather than a production route.
///
/// A harness builds a decision so a consumer can be tested against one. It is
/// not reachable from a compile, and it is where a fixture constructor belongs
/// when production source has to give one up.
const FIXTURE_LAYER: &str = "test-tooling";

/// The layers whose product is device facts and realized launches.
///
/// A backend probes the hardware and reports what it grants, and it prepares the
/// width the artifact declared against what the target admits. Both answers are
/// spelled with the same three axes a chosen shape is, so a return type cannot
/// tell a ceiling from a decision, and rule 3 would read every reported limit as
/// a cost model. Rules 1 and 2 still bind these layers: a backend that builds a
/// decision, or rewrites the geometry the artifact froze, is a finding wherever
/// it sits.
const DEVICE_LAYERS: &[&str] = &["backend-neutral", "concrete-backend"];

/// Reports a second route to a schedule decision.
pub struct ScheduleOwnership;

impl crate::gate::GateBehavior for ScheduleOwnership {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let registry = Registry::read(&tree)?;
        let (decisions, unclassified) = decision_types(&tree)?;
        let geometry = geometry_fields(&tree)?;
        let setters = geometry_setters(&tree)?;
        let mut report = Report::clean();
        report.note(format!(
            "{} decision types derived from the plan dimensions and the neutral schedule: {}",
            decisions.len(),
            joined(&decisions)
        ));
        report.note(format!(
            "geometry fields derived from the selected phase: {}",
            joined(&geometry)
        ));
        report.note(format!(
            "geometry setters derived from the IR: {}",
            joined(&setters)
        ));
        report.note(format!(
            "{} owns selection; {} other layers receive it",
            registry.owner_package, registry.other_layers
        ));
        for finding in unclassified {
            report.find(finding);
        }
        let rules = Rules {
            decisions,
            geometry,
            setters,
        };
        let mut judged = 0_usize;
        for path in tree.all_rust() {
            let Some(crate_directory) = registry.subject_crate(&path) else {
                continue;
            };
            if is_test_path(&path) || registry.layer(crate_directory) == FIXTURE_LAYER {
                continue;
            }
            let text = tree.read(&path)?;
            let Ok(file) = syn::parse_file(&text) else {
                continue;
            };
            judged += 1;
            let functions = collect(&file, &rules);
            for finding in findings(&path, crate_directory, &functions, &rules, &registry) {
                report.find(finding);
            }
        }
        report.cover_complete("production sources outside the compiler boundary", judged);
        Ok(report)
    }
}

/// A sorted set as one line of prose.
fn joined(set: &BTreeSet<String>) -> String {
    set.iter().cloned().collect::<Vec<_>>().join(", ")
}

/// What the rules are derived from, resolved once per run.
struct Rules {
    /// Types that are a schedule decision.
    decisions: BTreeSet<String>,
    /// Fields of a selected phase that carry geometry.
    geometry: BTreeSet<String>,
    /// IR methods that write a launch extent into a program.
    setters: BTreeSet<String>,
}

impl Rules {
    /// Whether a signature names a decision type.
    fn signature_receives(&self, signature: &str) -> bool {
        self.decisions
            .iter()
            .any(|decision| signature.contains(decision.as_str()))
    }

    /// Whether a signature takes a decision by mutable reference.
    ///
    /// Receiving a selected value and copying it forward is realization.
    /// Receiving it mutably and writing its geometry is a retune, which the
    /// artifact identity cannot follow, so the realization exemption stops here.
    fn signature_mutates_decision(&self, signature: &str) -> bool {
        let compact = signature.replace(' ', "");
        self.decisions
            .iter()
            .any(|decision| compact.contains(&format!("&mut{decision}")))
    }

    /// Whether a method writes a launch extent into a program.
    fn is_geometry_setter(&self, method: &str) -> bool {
        self.setters.contains(method)
    }

    /// Whether a signature takes a device capability record.
    ///
    /// The stems are the shapes a capability record is spelled with in this
    /// tree. A crate that chooses a launch shape from what the hardware grants
    /// has a cost model; a crate that states the shape of its own program from
    /// its own problem size is declaring a search input.
    fn signature_takes_device_facts(&self, signature: &str) -> bool {
        const DEVICE_FACT_STEMS: &[&str] = &[
            "Caps",
            "Capabilities",
            "Limits",
            "Facts",
            "Profile",
            "Measurement",
        ];
        DEVICE_FACT_STEMS
            .iter()
            .any(|stem| signature.contains(stem))
    }

    /// Whether a return type is geometry.
    fn returns_geometry(&self, output: &str, name: &str) -> bool {
        let compact = output.replace(' ', "");
        if compact.contains("[u32;3]") {
            return true;
        }
        compact == "->u32"
            && self
                .geometry
                .iter()
                .any(|field| name.contains(field.as_str()))
    }
}

/// Crate layers as the ownership registry declares them.
struct Registry {
    /// Crate directory to declared layer.
    layers: BTreeMap<String, String>,
    /// Package name of the crate in the owning layer.
    owner_package: String,
    /// Directory of the crate in the owning layer.
    owner_directory: String,
    /// How many layers other than the owning one hold crates.
    other_layers: usize,
}

impl Registry {
    /// Read the declared layers, failing closed when no crate owns selection.
    fn read(tree: &Tree) -> Result<Self, GateError> {
        let table = tree.read_toml("docs/CRATE_OWNERSHIP.toml")?;
        let crates = table
            .get("crate")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                GateError::new(
                    "docs/CRATE_OWNERSHIP.toml declares no [[crate]] entries",
                    "declare every workspace member with its path and layer",
                )
            })?;
        let mut layers = BTreeMap::new();
        let mut owner = None;
        for entry in crates {
            let read = |key: &str| entry.get(key).and_then(toml::Value::as_str);
            let (Some(package), Some(path), Some(layer)) =
                (read("package"), read("path"), read("layer"))
            else {
                continue;
            };
            if layer == OWNER_LAYER {
                owner = Some((package.to_string(), path.to_string()));
            }
            layers.insert(path.to_string(), layer.to_string());
        }
        let (owner_package, owner_directory) = owner.ok_or_else(|| {
            GateError::new(
                format!("no crate declares the `{OWNER_LAYER}` layer"),
                format!("declare the schedule-selection owner with layer = \"{OWNER_LAYER}\""),
            )
        })?;
        for layer in DEVICE_LAYERS {
            if !layers.values().any(|declared| declared == layer) {
                return Err(GateError::new(
                    format!("no crate declares the `{layer}` layer"),
                    "declare the device-facing layers, or update the gate when a layer is renamed",
                ));
            }
        }
        let other_layers = layers
            .values()
            .filter(|layer| layer.as_str() != OWNER_LAYER)
            .collect::<BTreeSet<_>>()
            .len();
        Ok(Self {
            layers,
            owner_package,
            owner_directory,
            other_layers,
        })
    }

    /// The crate directory judged for this path, or `None` when the path is the
    /// owner's own source or belongs to no registered crate.
    fn subject_crate(&self, path: &Path) -> Option<&str> {
        let text = path.to_str()?;
        let mut best: Option<&str> = None;
        for directory in self.layers.keys() {
            if text.starts_with(&format!("{directory}/src/"))
                && best.is_none_or(|current| directory.len() > current.len())
            {
                best = Some(directory.as_str());
            }
        }
        let directory = best?;
        if directory == self.owner_directory {
            return None;
        }
        Some(directory)
    }

    /// Whether a crate directory sits in a layer that reports device facts.
    fn is_device_layer(&self, directory: &str) -> bool {
        DEVICE_LAYERS.contains(&self.layer(directory))
    }

    /// The declared layer of a crate directory.
    fn layer(&self, directory: &str) -> &str {
        self.layers
            .get(directory)
            .map_or("unregistered", String::as_str)
    }
}

/// The five dimensions a schedule decides, as the plan spells its own fields.
///
/// A field of the selected plan whose name carries one of these terms holds a
/// decision. The terms are the dimensions, not the types: the types are read off
/// the declaration, so renaming one changes nothing here.
const DIMENSION_TERMS: &[&str] = &[
    "topology",
    "schedule",
    "fusion",
    "barrier",
    "materialization",
    "execution",
    "geometry",
    "layout",
    "placement",
    "persistence",
];

/// What the plan records about how it was selected rather than what it selected.
const PROVENANCE_TERMS: &[&str] = &[
    "derivation",
    "certificate",
    "candidates",
    "pareto",
    "budget",
    "work",
    "cost",
    "pruned",
    "measurement",
];

/// Where the plan and the neutral schedule are declared.
const PLAN_FILE: &str = "vyre-megakernel/src/schema/plan.rs";
/// Where the neutral schedule and its phases are declared.
const SCHEDULE_FILE: &str = "vyre-foundation/src/schedule/mod.rs";

/// Decision types and one finding per plan field nobody classified.
///
/// A field is a decision or it is provenance. A field that is neither turns this
/// gate red the day it lands, because a stage that records something new about
/// execution needs an owner before it needs a reader.
fn decision_types(tree: &Tree) -> Result<(BTreeSet<String>, Vec<Finding>), GateError> {
    let mut decisions = BTreeSet::new();
    let mut unclassified = Vec::new();
    let plan = tree.read(PLAN_FILE)?;
    let fields = declared_fields(&plan, "SelectedPlan").ok_or_else(|| {
        GateError::new(
            format!("`SelectedPlan` is not declared in {PLAN_FILE}"),
            "point the roster at the file that declares the selected plan",
        )
    })?;
    for (name, ty) in fields {
        if DIMENSION_TERMS.iter().any(|term| name.contains(term)) {
            collect_type_idents(&ty, &mut decisions);
        } else if !PROVENANCE_TERMS.iter().any(|term| name.contains(term)) {
            unclassified.push(Finding::in_file(
                PLAN_FILE,
                format!(
                    "`SelectedPlan.{name}` is neither one of the dimensions a schedule decides                      nor a record of how it was selected, so no crate owns it"
                ),
                "name the field after the dimension it decides, or after the provenance it                  records, so the ownership rule can judge who may build it",
            ));
        }
    }
    let schedule = tree.read(SCHEDULE_FILE)?;
    let fields = declared_fields(&schedule, "SelectedSchedule").ok_or_else(|| {
        GateError::new(
            format!("`SelectedSchedule` is not declared in {SCHEDULE_FILE}"),
            "point the roster at the file that declares the neutral selected schedule",
        )
    })?;
    decisions.insert("SelectedSchedule".to_string());
    for (name, ty) in fields {
        if name == "version" || name.contains("identity") {
            continue;
        }
        collect_type_idents(&ty, &mut decisions);
    }
    Ok((decisions, unclassified))
}

/// Geometry setters, derived from the IR the compiler freezes geometry into.
///
/// A method that takes a three-axis extent and writes it into a program states
/// the launch shape. The names are read from the IR rather than listed, so a
/// second setter added beside the first is judged the day it lands.
fn geometry_setters(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    const IR_ROOT: &str = "vyre-foundation/src/ir_inner/";
    let mut setters = BTreeSet::new();
    for path in tree.all_rust() {
        let Some(text) = path.to_str() else { continue };
        if !text.starts_with(IR_ROOT) {
            continue;
        }
        let source = tree.read(&path)?;
        let Ok(file) = syn::parse_file(&source) else {
            continue;
        };
        collect_setters(&file.items, &mut setters);
    }
    if setters.is_empty() {
        return Err(GateError::new(
            "the IR declares no method that writes a three-axis launch extent",
            "keep the program's launch-geometry setter in the IR, where the compiler freezes it",
        ));
    }
    Ok(setters)
}

/// Methods named `set_*` that take a three-axis extent.
fn collect_setters(items: &[syn::Item], out: &mut BTreeSet<String>) {
    for item in items {
        match item {
            syn::Item::Impl(block) => {
                for member in &block.items {
                    let syn::ImplItem::Fn(function) = member else {
                        continue;
                    };
                    let name = function.sig.ident.to_string();
                    if !name.starts_with("set_") {
                        continue;
                    }
                    let takes_extent = function.sig.inputs.iter().any(|input| {
                        let syn::FnArg::Typed(typed) = input else {
                            return false;
                        };
                        typed.ty.to_token_stream().to_string().replace(' ', "") == "[u32;3]"
                    });
                    if takes_extent {
                        out.insert(name);
                    }
                }
            }
            syn::Item::Mod(module) => {
                if let Some((_, items)) = module.content.as_ref() {
                    collect_setters(items, out);
                }
            }
            _ => {}
        }
    }
}

/// Geometry fields, derived from the extent-typed fields of a selected phase.
fn geometry_fields(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let text = tree.read("vyre-foundation/src/schedule/mod.rs")?;
    let fields = declared_fields(&text, "SchedulePhase").ok_or_else(|| {
        GateError::new(
            "`SchedulePhase` is not declared in vyre-foundation/src/schedule/mod.rs",
            "point the geometry rule at the file that declares a selected phase",
        )
    })?;
    let geometry: BTreeSet<String> = fields
        .into_iter()
        .filter(|(name, ty)| is_geometry(name, ty))
        .map(|(name, _)| name)
        .collect();
    if geometry.is_empty() {
        return Err(GateError::new(
            "no field of `SchedulePhase` carries an extent, so the geometry rule judges nothing",
            "keep the selected phase's grid, workgroup and width fields extent-typed",
        ));
    }
    Ok(geometry)
}

/// Whether a declared field carries geometry.
///
/// A three-axis extent is a grid, a workgroup or a tile whatever it is called,
/// and a scalar width is a lane count.
fn is_geometry(name: &str, ty: &syn::Type) -> bool {
    let text = ty.to_token_stream().to_string().replace(' ', "");
    text == "[u32;3]" || (text == "u32" && name.contains("width"))
}

/// The declared fields of one struct, in declaration order.
fn declared_fields(text: &str, declaration: &str) -> Option<Vec<(String, syn::Type)>> {
    let file = syn::parse_file(text).ok()?;
    for item in file.items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        if item.ident != declaration {
            continue;
        }
        return Some(
            item.fields
                .into_iter()
                .filter_map(|field| Some((field.ident?.to_string(), field.ty)))
                .collect(),
        );
    }
    None
}

/// Every named type inside `ty`, wrappers unwrapped and primitives dropped.
fn collect_type_idents(ty: &syn::Type, out: &mut BTreeSet<String>) {
    match ty {
        syn::Type::Path(path) => {
            for segment in &path.path.segments {
                let name = segment.ident.to_string();
                if !name.chars().next().is_some_and(char::is_uppercase) {
                    continue;
                }
                if is_wrapper(&name) {
                    if let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments {
                        for argument in &arguments.args {
                            if let syn::GenericArgument::Type(inner) = argument {
                                collect_type_idents(inner, out);
                            }
                        }
                    }
                    continue;
                }
                out.insert(name);
            }
        }
        syn::Type::Reference(reference) => collect_type_idents(&reference.elem, out),
        _ => {}
    }
}

/// Whether a type only carries another one.
fn is_wrapper(name: &str) -> bool {
    matches!(name, "Vec" | "Option" | "Box" | "BTreeMap" | "BTreeSet")
}

/// One function of one file, with what it receives and what it decides.
struct FunctionRecord {
    /// Name as declared, without a path.
    name: String,
    /// Declared signature, with the receiver type appended when there is one.
    signature: String,
    /// Whether the return type is geometry.
    returns_geometry: bool,
    /// Line the declaration sits on.
    line: u32,
    /// Decision enum variants this body mints, with the line of the mint.
    mints: Vec<(String, u32)>,
    /// Decision struct types this body constructs, with the line of the construction.
    constructs: Vec<(String, u32)>,
    /// Geometry field writes in this body, with their line.
    writes: Vec<(String, u32)>,
    /// Names this body calls.
    calls: BTreeSet<String>,
    /// Values this body binds, which are values it built rather than received.
    locals: BTreeSet<String>,
}

/// Every production function of a parsed file.
fn collect(file: &syn::File, rules: &Rules) -> Vec<FunctionRecord> {
    let mut walker = Walker {
        rules,
        records: Vec::new(),
        current: None,
        self_type: Vec::new(),
    };
    syn::visit::visit_file(&mut walker, file);
    walker.records
}

/// Collects one record per production function, skipping test code.
struct Walker<'rules> {
    rules: &'rules Rules,
    records: Vec<FunctionRecord>,
    current: Option<FunctionRecord>,
    self_type: Vec<String>,
}

impl Walker<'_> {
    /// Start a record, finishing whatever nested function was open.
    fn enter(&mut self, signature: &syn::Signature) {
        self.leave();
        let name = signature.ident.to_string();
        let receiver = signature
            .inputs
            .iter()
            .any(|input| matches!(input, syn::FnArg::Receiver(_)));
        let text = signature_text(signature, self.self_type.last().filter(|_| receiver));
        let output = signature.output.to_token_stream().to_string();
        self.current = Some(FunctionRecord {
            returns_geometry: self.rules.returns_geometry(&output, &name),
            name,
            signature: text,
            line: line_of(signature),
            mints: Vec::new(),
            constructs: Vec::new(),
            writes: Vec::new(),
            calls: BTreeSet::new(),
            locals: BTreeSet::new(),
        });
    }

    /// Close the open record.
    fn leave(&mut self) {
        if let Some(record) = self.current.take() {
            self.records.push(record);
        }
    }

    /// The signature of the body being walked, empty outside one.
    fn signature(&self) -> String {
        self.current
            .as_ref()
            .map(|record| record.signature.clone())
            .unwrap_or_default()
    }

    /// Whether a setter call writes into a program this body was handed.
    ///
    /// A program the body built and is still assembling declares its own launch
    /// shape, and the declared shape is an input to the search. A program that
    /// arrived as a parameter or lives in a field already carries the shape the
    /// artifact froze, so writing it there is a rewrite.
    fn rewrites_a_received_program(&self, expr: &syn::ExprMethodCall) -> bool {
        let Some(root) = receiver_root(&expr.receiver) else {
            return false;
        };
        self.current
            .as_ref()
            .is_some_and(|record| !record.locals.contains(&root))
    }

    /// Record a geometry write.
    fn note_write(&mut self, field: String, line: u32) {
        if let Some(record) = self.current.as_mut() {
            record.writes.push((field, line));
        }
    }
}

impl<'ast> syn::visit::Visit<'ast> for Walker<'_> {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if is_test_gated(&item.attrs) {
            return;
        }
        syn::visit::visit_item_mod(self, item);
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        self.self_type
            .push(item.self_ty.to_token_stream().to_string());
        syn::visit::visit_item_impl(self, item);
        self.self_type.pop();
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        if is_test_gated(&item.attrs) {
            return;
        }
        self.enter(&item.sig);
        syn::visit::visit_block(self, &item.block);
        self.leave();
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if is_test_gated(&item.attrs) {
            return;
        }
        self.enter(&item.sig);
        syn::visit::visit_block(self, &item.block);
        self.leave();
    }

    fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
        let line = line_of(expr);
        if let Some((type_name, mints_variant)) = classify_decision_path(&expr.path) {
            if let Some(record) = self.current.as_mut() {
                if mints_variant {
                    record.mints.push((type_name, line));
                } else {
                    record.constructs.push((type_name, line));
                }
            }
        }
        let decides = expr
            .path
            .segments
            .last()
            .is_some_and(|last| self.rules.decisions.contains(&last.ident.to_string()));
        if decides {
            for field in &expr.fields {
                if let syn::Member::Named(name) = &field.member {
                    let name = name.to_string();
                    if self.rules.geometry.contains(&name) && !is_neutral_extent(&field.expr) {
                        self.note_write(name, line_of(&field.expr));
                    }
                }
            }
        }
        syn::visit::visit_expr_struct(self, expr);
    }

    /// A pattern reads a decision and never makes one.
    ///
    /// `syn` aliases `PatPath` to `ExprPath`, and its `visit_pat` dispatches a
    /// unit-variant pattern to `visit_expr_path`, so `match topology {
    /// Topology::Sparse => .. }` arrives at the construction hook and every arm
    /// is charged as a second route to the decision. The module doc has claimed
    /// since it landed that the walk reaches expressions only; this is what
    /// makes that true. Guards and arm bodies are expressions and stay on the
    /// walk, so a guard that writes geometry is still judged.
    fn visit_pat(&mut self, _pattern: &'ast syn::Pat) {}

    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        if expr.path.segments.len() < 2 {
            syn::visit::visit_expr_path(self, expr);
            return;
        }
        if let Some((type_name, mints_variant)) = classify_decision_path(&expr.path) {
            let line = line_of(expr);
            if let Some(record) = self.current.as_mut() {
                if mints_variant {
                    record.mints.push((type_name, line));
                } else {
                    record.constructs.push((type_name, line));
                }
            }
        }
        syn::visit::visit_expr_path(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = expr.func.as_ref() {
            if let Some(last) = path.path.segments.last() {
                if let Some(record) = self.current.as_mut() {
                    record.calls.insert(last.ident.to_string());
                }
            }
        }
        syn::visit::visit_expr_call(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method = expr.method.to_string();
        let writes_a_chosen_extent = expr.args.iter().any(states_extent)
            || self.rules.signature_takes_device_facts(&self.signature());
        if self.rules.is_geometry_setter(&method)
            && self.rewrites_a_received_program(expr)
            && writes_a_chosen_extent
        {
            self.note_write(method.clone(), line_of(expr));
        }
        if let Some(record) = self.current.as_mut() {
            record.calls.insert(method);
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_assign(&mut self, expr: &'ast syn::ExprAssign) {
        if let syn::Expr::Field(field) = expr.left.as_ref() {
            if let syn::Member::Named(name) = &field.member {
                let name = name.to_string();
                if self.rules.geometry.contains(&name) && states_extent(&expr.right) {
                    self.note_write(name, line_of(expr));
                }
            }
        }
        syn::visit::visit_expr_assign(self, expr);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac.path.is_ident("vec") {
            if let Ok(elements) = mac.parse_body_with(
                syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
            ) {
                for element in &elements {
                    syn::visit::visit_expr(self, element);
                }
            }
        }
        syn::visit::visit_macro(self, mac);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        let mut bound = BTreeSet::new();
        bindings(&local.pat, &mut bound);
        if let Some(record) = self.current.as_mut() {
            record.locals.extend(bound);
        }
        syn::visit::visit_local(self, local);
    }
}

/// Every name a pattern binds.
fn bindings(pat: &syn::Pat, out: &mut BTreeSet<String>) {
    match pat {
        syn::Pat::Ident(ident) => {
            out.insert(ident.ident.to_string());
        }
        syn::Pat::Tuple(tuple) => {
            for element in &tuple.elems {
                bindings(element, out);
            }
        }
        syn::Pat::Type(typed) => bindings(&typed.pat, out),
        syn::Pat::Reference(reference) => bindings(&reference.pat, out),
        _ => {}
    }
}

/// The base name a method receiver chains off.
fn receiver_root(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => Some(path.path.segments.first()?.ident.to_string()),
        syn::Expr::Field(field) => receiver_root(&field.base),
        syn::Expr::MethodCall(call) => receiver_root(&call.receiver),
        syn::Expr::Index(index) => receiver_root(&index.expr),
        syn::Expr::Reference(reference) => receiver_root(&reference.expr),
        syn::Expr::Unary(unary) => receiver_root(&unary.expr),
        _ => None,
    }
}

/// The decision type a path names, and whether the path mints a variant of it.
///
/// A module prefix is skipped, so `candidate::Topology::Sparse` reads the same
/// as `Topology::Sparse`. Reading only the first segment missed both, because a
/// lowercase module name failed the uppercase test and the path was dropped
/// whole. The type is the last uppercase-initial segment, or the one before it
/// when that segment is itself uppercase-initial, which is what distinguishes
/// minting `Topology::Sparse` from constructing `topology::Plan`.
fn classify_decision_path(path: &syn::Path) -> Option<(String, bool)> {
    let idents: Vec<String> = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    let last_upper = idents
        .iter()
        .rposition(|ident| ident.chars().next().is_some_and(char::is_uppercase))?;
    if last_upper >= 1
        && idents[last_upper - 1]
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
    {
        return Some((idents[last_upper - 1].clone(), true));
    }
    Some((idents[last_upper].clone(), false))
}

/// The line an expression starts on.
fn line_of(node: &impl Spanned) -> u32 {
    u32::try_from(node.span().start().line).unwrap_or(0)
}

/// Whether a geometry initializer states no shape of its own.
///
/// A phase built from another phase's field, or from a value handed in, carries
/// the shape it was given. An extent written out states one.
fn is_neutral_extent(expr: &syn::Expr) -> bool {
    matches!(expr, syn::Expr::Field(_) | syn::Expr::Path(_))
}

/// Whether an expression writes out a launch extent.
///
/// An assignment of a decoded map, a cloned field or a returned buffer carries
/// whatever it was handed. Three axes written out is a shape someone chose.
fn states_extent(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Array(array) => array.elems.len() == 3,
        syn::Expr::Reference(reference) => states_extent(&reference.expr),
        _ => false,
    }
}

/// What a function receives: its parameters and, in an impl, its receiver type.
///
/// The self type counts only for a method with a receiver. A method operates on
/// a value it was handed; an associated function with no receiver builds one out
/// of whatever it can reach, which is the shape a selector has.
///
/// The return type is left out on purpose. A function that returns a decision is
/// the one being judged, so counting its own output as something it was handed
/// would exempt every selector in the tree.
///
/// The decision roster is applied to this text later, so the walk itself stays
/// free of the roster and can be tested on its own.
fn signature_text(signature: &syn::Signature, self_type: Option<&String>) -> String {
    let mut text = signature
        .inputs
        .iter()
        .map(|input| input.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(self_type) = self_type {
        text.push(' ');
        text.push_str(self_type);
    }
    text
}

/// Whether attributes gate an item to test builds.
fn is_test_gated(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let text = attr.to_token_stream().to_string().replace(' ', "");
        text.contains("cfg(test)") || text == "#[test]" || text.contains("cfg(any(test")
    })
}

/// Whether a path holds test rather than production source.
fn is_test_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.contains("/tests/")
        || text.contains("/benches/")
        || text.contains("_tests.rs")
        || text.contains("/test_")
        || text.contains("fixtures")
}

/// Findings for one file.
fn findings(
    path: &Path,
    crate_directory: &str,
    functions: &[FunctionRecord],
    rules: &Rules,
    registry: &Registry,
) -> Vec<Finding> {
    let layer = registry.layer(crate_directory);
    let owner = &registry.owner_package;
    let by_name: BTreeMap<&str, &FunctionRecord> = functions
        .iter()
        .map(|record| (record.name.as_str(), record))
        .collect();
    let mut findings = Vec::new();
    for record in functions {
        let realizes = receives_within_depth(record, &by_name, rules, 0);
        for (minted, line) in &record.mints {
            if !rules.decisions.contains(minted) {
                continue;
            }
            findings.push(Finding::at(
                path.to_path_buf(),
                *line,
                format!(
                    "`{crate_directory}` is in the `{layer}` layer and mints a `{minted}` \
                     variant in `{}`, so it selects execution on a second route",
                    record.name
                ),
                format!(
                    "take the selected schedule as an input and realize it, or return a fact \
                     `{owner}` consumes; selection is `{owner}`'s alone"
                ),
            ));
        }
        for (constructed, line) in &record.constructs {
            if !rules.decisions.contains(constructed) || realizes {
                continue;
            }
            findings.push(Finding::at(
                path.to_path_buf(),
                *line,
                format!(
                    "`{crate_directory}` is in the `{layer}` layer and builds a `{constructed}` \
                     in `{}` without receiving one, so it selects execution on a second route",
                    record.name
                ),
                format!(
                    "take the selected schedule as an input and realize it, or return a fact \
                     `{owner}` consumes; selection is `{owner}`'s alone"
                ),
            ));
        }
        let retunes = rules.signature_mutates_decision(&record.signature);
        for (field, line) in &record.writes {
            if realizes && !retunes {
                continue;
            }
            findings.push(Finding::at(
                path.to_path_buf(),
                *line,
                format!(
                    "`{crate_directory}` writes the selected `{field}` in `{}`, which restates \
                     geometry the artifact already froze",
                    record.name
                ),
                format!(
                    "carry the geometry the artifact declares; a shape `{owner}` did not \
                     authenticate runs bytes no identity covers"
                ),
            ));
        }
        let ranks_hardware = rules.signature_takes_device_facts(&record.signature)
            && !registry.is_device_layer(crate_directory);
        if let (true, true, false) = (record.returns_geometry, ranks_hardware, realizes) {
            findings.push(Finding::at(
                path.to_path_buf(),
                record.line,
                format!(
                    "`{crate_directory}` maps a device fact to geometry in `{}`, which is a \
                     second cost model",
                    record.name
                ),
                format!(
                    "report the device limit as a fact and let `{owner}` rank shapes against it; \
                     one cost model orders every candidate"
                ),
            ));
        }
    }
    findings
}

/// Whether this function or a caller within `CALLER_DEPTH` receives a decision.
fn receives_within_depth(
    record: &FunctionRecord,
    by_name: &BTreeMap<&str, &FunctionRecord>,
    rules: &Rules,
    depth: usize,
) -> bool {
    if rules.signature_receives(&record.signature) {
        return true;
    }
    if depth >= CALLER_DEPTH {
        return false;
    }
    by_name.values().any(|caller| {
        caller.name != record.name
            && caller.calls.contains(&record.name)
            && receives_within_depth(caller, by_name, rules, depth + 1)
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn registry() -> Registry {
        let layers = [
            ("vyre-megakernel", OWNER_LAYER),
            ("vyre-foundation", "foundation"),
            ("vyre-runtime", "runtime"),
        ]
        .into_iter()
        .map(|(path, layer)| (path.to_string(), layer.to_string()))
        .collect();
        Registry {
            layers,
            owner_package: "vyre-megakernel".to_string(),
            owner_directory: "vyre-megakernel".to_string(),
            other_layers: 2,
        }
    }

    fn rules() -> Rules {
        Rules {
            decisions: ["ExecutionMode", "SchedulePhase", "SelectedSchedule"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            geometry: ["grid", "workgroup", "vector_width"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            setters: ["set_workgroup_size"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }

    fn judge(source: &str) -> Vec<Finding> {
        let file = syn::parse_file(source).expect("Fix: keep the fixture parseable Rust.");
        let rules = rules();
        let functions = collect(&file, &rules);
        findings(
            &PathBuf::from("vyre-runtime/src/thing.rs"),
            "vyre-runtime",
            &functions,
            &rules,
            &registry(),
        )
    }

    /// WHY: the rule the gate exists for. A crate that builds a decision it was
    /// never handed has chosen execution, whatever the function is called.
    #[test]
    fn a_construction_with_no_received_decision_is_a_finding() {
        let findings = judge(
            "fn pick(node_count: usize) -> ExecutionMode {\n    if node_count > 64 { ExecutionMode::Persistent } else { ExecutionMode::Static }\n}\n",
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("ExecutionMode")),
            "the finding names the decision: {}",
            Finding::messages(&findings)
        );
        assert!(
            findings[0].fix.contains("vyre-megakernel"),
            "the fix names the owner: {}",
            findings[0].fix
        );
    }

    /// WHY: realization looks like selection at one call site. A helper building
    /// a phase for a caller that holds the selected schedule is lowering, and
    /// reporting it would make the gate red exactly where the correct pattern is.
    #[test]
    fn a_construction_inside_a_helper_of_a_receiver_is_realization() {
        let findings = judge(
            "fn realize(schedule: &SelectedSchedule) -> Vec<SchedulePhase> {\n    vec![build(schedule.phases[0].grid)]\n}\nfn build(grid: [u32; 3]) -> SchedulePhase {\n    SchedulePhase { grid }\n}\n",
        );
        assert!(
            findings.is_empty(),
            "a helper of a receiver realizes: {}",
            Finding::messages(&findings)
        );
    }

    /// WHY: the depth bound is what keeps the walk inside one file's call graph.
    /// A helper nobody hands a decision to is a second selector however deep it
    /// sits.
    #[test]
    fn a_helper_reached_only_from_a_non_receiver_is_still_a_finding() {
        let findings = judge(
            "fn entry(node_count: u32) -> SchedulePhase {\n    build(node_count)\n}\nfn build(node_count: u32) -> SchedulePhase {\n    SchedulePhase { grid: [node_count, 1, 1] }\n}\n",
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("build")),
            "{}",
            Finding::messages(&findings)
        );
    }

    /// WHY: every consumer of a decision matches on it. Counting a pattern as a
    /// construction would report the runtime for reading the mode it was handed,
    /// which is the one thing it is supposed to do.
    #[test]
    fn a_match_on_a_decision_is_not_a_construction() {
        let findings = judge(
            "fn submit(mode: ExecutionMode) -> u32 {\n    match mode { ExecutionMode::Static => 0, ExecutionMode::Persistent => 1 }\n}\n",
        );
        assert!(
            findings.is_empty(),
            "reading is not selecting: {}",
            Finding::messages(&findings)
        );
    }

    /// WHY: the write is the failure that cannot be recovered from. The artifact
    /// authenticated one workgroup; a launch at another runs bytes no identity
    /// covers.
    #[test]
    fn an_assignment_to_a_selected_geometry_field_is_a_finding() {
        let findings = judge(
            "fn widen(phase: &mut SchedulePhase, width: u32) {\n    phase.workgroup = [width, 1, 1];\n}\n",
        );
        assert_eq!(findings.len(), 1, "{}", Finding::messages(&findings));
        assert!(
            findings[0].message.contains("workgroup"),
            "{}",
            findings[0].message
        );
    }

    /// WHY: a generated setter writes the same field an assignment does, and the
    /// pass that rewrote a program's workgroup size reached it that way.
    #[test]
    fn a_generated_geometry_setter_is_the_same_write() {
        let findings = judge("fn tune(program: &mut Program, width: u32) {\n    program.set_workgroup_size([width, 1, 1]);\n}\n");
        let realized = judge("fn lower(selected: &SelectedSchedule, program: &mut Program) {\n    program.set_workgroup_size(selected.workgroup);\n}\n");
        assert!(
            realized.is_empty(),
            "copying a selected extent into a program is realization: {}",
            Finding::messages(&realized)
        );
        let retuned = judge("fn widen(phase: &mut SchedulePhase, program: &mut Program) {\n    program.set_workgroup_size([64, 1, 1]);\n}\n");
        assert_eq!(
            retuned.len(),
            1,
            "rewriting a received decision is a retune, not realization: {}",
            Finding::messages(&retuned)
        );
        let declared = judge("fn build(width: u32) -> Program {\n    let mut program = Program::wrapped();\n    program.set_workgroup_size([width, 1, 1]);\n    program\n}\n");
        assert!(
            declared.is_empty(),
            "a program under construction declares its own shape: {}",
            Finding::messages(&declared)
        );
        assert_eq!(findings.len(), 1, "{}", Finding::messages(&findings));
        assert!(
            findings[0].message.contains("set_workgroup_size"),
            "{}",
            findings[0].message
        );
        assert!(
            !judge("fn keep(program: &mut Program) {\n    program.set_buffer_count(4);\n}\n")
                .iter()
                .any(|finding| finding.message.contains("set_buffer_count")),
            "a setter for another field is not geometry"
        );
    }

    /// WHY: a crate that answers how wide to launch while holding what the
    /// device grants has a cost model, and the one cost model belongs to the
    /// owner. A shape derived only from the program's own problem size is a
    /// declared input to the search, not a second selector.
    #[test]
    fn geometry_returned_from_a_device_fact_is_a_second_cost_model() {
        let findings = judge(
            "fn select_workgroup_x(problem: u32, limits: LaunchGeometryLimits) -> [u32; 3] {\n    if problem > 4096 { [256, 1, 1] } else { [64, 1, 1] }\n}\n",
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("second cost model")),
            "{}",
            Finding::messages(&findings)
        );
        let carried = judge(
            "fn carry(declared: [u32; 3], limits: LaunchGeometryLimits, count: u32) -> [u32; 3] {\n    if count > 0 { declared } else { declared }\n}\n",
        );
        assert!(
            !carried.is_empty(),
            "holding a capability record while answering with a shape is the cost model, \
             whatever the body does with the record: {}",
            Finding::messages(&carried)
        );
        let own_shape = judge(
            "fn strided_grid(elements: u32) -> [u32; 3] {\n    if elements > 1024 { [1024, 1, 1] } else { [elements, 1, 1] }\n}\n",
        );
        assert!(
            own_shape.is_empty(),
            "a program stating its own declared shape from its own problem size is a search \
             input: {}",
            Finding::messages(&own_shape)
        );
    }

    /// WHY: the roster is derived so a field added to the plan cannot arrive
    /// without an owner. A hardcoded list goes stale in silence, which is the
    /// same as having no rule.
    #[test]
    fn the_roster_comes_from_the_declared_fields() {
        let source = "pub struct SelectedPlan {\n    pub topology: crate::candidate::ExecutionTopology,\n    pub fusion: Vec<FusionRecord>,\n    pub pareto_frontier: u32,\n    pub measurement: Option<PlanMeasurement>,\n}\n";
        let fields = declared_fields(source, "SelectedPlan").expect("the declaration is present");
        let mut decisions = BTreeSet::new();
        for (_, ty) in &fields {
            collect_type_idents(ty, &mut decisions);
        }
        assert!(decisions.contains("ExecutionTopology"), "{decisions:?}");
        assert!(
            decisions.contains("FusionRecord"),
            "Vec unwraps: {decisions:?}"
        );
        assert!(
            decisions.contains("PlanMeasurement"),
            "Option unwraps: {decisions:?}"
        );
        assert!(
            !decisions.contains("Vec"),
            "a wrapper is no decision: {decisions:?}"
        );
        assert!(
            !decisions.contains("u32"),
            "a primitive is no decision: {decisions:?}"
        );
    }

    /// WHY: the geometry fields are read from the phase rather than named here,
    /// so renaming one keeps the rule pointed at it, and a field that carries no
    /// extent never becomes geometry by being added.
    #[test]
    fn geometry_is_the_extent_typed_fields_of_a_phase() {
        let source = "pub struct SchedulePhase {\n    pub id: SchedulePhaseId,\n    pub grid: [u32; 3],\n    pub workgroup: [u32; 3],\n    pub vector_width: u32,\n    pub source_regions: Vec<u32>,\n}\n";
        let fields = declared_fields(source, "SchedulePhase").expect("the declaration is present");
        let geometry: BTreeSet<String> = fields
            .into_iter()
            .filter(|(name, ty)| is_geometry(name, ty))
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            geometry,
            ["grid", "vector_width", "workgroup"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        );
    }

    /// WHY: a test may build any decision it likes, and judging test code would
    /// have made the gate red on fixtures instead of on production routes.
    #[test]
    fn a_test_gated_module_is_not_judged() {
        let findings = judge(
            "#[cfg(test)]\nmod tests {\n    fn fixture() -> ExecutionMode {\n        ExecutionMode::Static\n    }\n}\n",
        );
        assert!(
            findings.is_empty(),
            "test code is not a route: {}",
            Finding::messages(&findings)
        );
    }

    /// WHY: the owner is read from the registry, so moving selection to another
    /// crate is a registry edit rather than an edit here. Judging the owner would
    /// report every legitimate selection in the tree.
    #[test]
    fn the_owning_crate_is_not_judged_and_every_other_one_is() {
        let registry = registry();
        assert_eq!(
            registry.subject_crate(Path::new("vyre-megakernel/src/select.rs")),
            None
        );
        assert_eq!(
            registry.subject_crate(Path::new("vyre-runtime/src/submit.rs")),
            Some("vyre-runtime")
        );
        assert_eq!(
            registry.subject_crate(Path::new("docs/ARCHITECTURE.md")),
            None
        );
        assert_eq!(registry.layer("vyre-runtime"), "runtime");
    }

    /// WHY: a match arm pattern reads a decision and must never be counted as a
    /// construction (such as matching on execution mode variants). In contrast,
    /// a function constructing a decision variant is a finding, and a match guard that
    /// writes a geometry field remains an observable rule 2 violation.
    #[test]
    fn match_arm_patterns_are_not_constructions_but_variant_constructs_and_guard_writes_are() {
        let reading_source = r#"
            pub enum ExecutionMode { Static, Persistent }
            pub struct ModeDecision { pub mode: ExecutionMode }

            impl ModeDecision {
                fn reason_code(&self) -> &'static str {
                    match self.mode {
                        ExecutionMode::Static => "static",
                        ExecutionMode::Persistent => "persistent",
                    }
                }
            }
        "#;
        let constructing_source = r#"
            pub enum ExecutionMode { Static, Persistent }

            fn select_mode(node_count: usize) -> ExecutionMode {
                if node_count > 64 {
                    ExecutionMode::Persistent
                } else {
                    ExecutionMode::Static
                }
            }
        "#;
        let guard_writing_source = r#"
            pub enum ExecutionMode { Static, Persistent }

            fn match_with_guard_write(mode: ExecutionMode, phase: &mut SchedulePhase) -> u32 {
                match mode {
                    ExecutionMode::Persistent if { phase.workgroup = [128, 1, 1]; true } => 1,
                    _ => 0,
                }
            }
        "#;

        let rules = rules();

        // 1. Matching only to read variants produces no findings
        let reading_file = syn::parse_file(reading_source).expect("valid rust");
        let reading_funcs = collect(&reading_file, &rules);
        let reading_findings = findings(
            &PathBuf::from("vyre-driver/src/megakernel_execution.rs"),
            "vyre-driver",
            &reading_funcs,
            &rules,
            &registry(),
        );
        assert!(
            reading_findings.is_empty(),
            "reading decision variants in match patterns must produce no findings: {}",
            Finding::messages(&reading_findings)
        );

        // 2. Constructing decision variants produces findings
        let constructing_file = syn::parse_file(constructing_source).expect("valid rust");
        let constructing_funcs = collect(&constructing_file, &rules);
        let constructing_findings = findings(
            &PathBuf::from("vyre-driver/src/megakernel_execution.rs"),
            "vyre-driver",
            &constructing_funcs,
            &rules,
            &registry(),
        );
        assert_eq!(
            constructing_findings.len(),
            2,
            "constructing decision variants without receiving one must produce findings: {}",
            Finding::messages(&constructing_findings)
        );

        // 3. Match guard writing geometry field produces a rule 2 finding
        let guard_file = syn::parse_file(guard_writing_source).expect("valid rust");
        let guard_funcs = collect(&guard_file, &rules);
        let guard_findings = findings(
            &PathBuf::from("vyre-driver/src/megakernel_execution.rs"),
            "vyre-driver",
            &guard_funcs,
            &rules,
            &registry(),
        );
        assert!(
            guard_findings
                .iter()
                .any(|f| f.message.contains("workgroup")),
            "geometry write in match guard must be reported: {}",
            Finding::messages(&guard_findings)
        );
    }

    /// WHY: minting a decision variant is selection even when the function or its callers
    /// receive a decision. Forwarding a received decision without naming a variant is realization.
    #[test]
    fn minting_a_decision_variant_is_reported_even_when_receiving_a_decision() {
        let minting_receiver = judge(
            "fn stabilize(previous: ExecutionMode, count: usize) -> ExecutionMode {\n    if count > 64 { ExecutionMode::Persistent } else { previous }\n}\n",
        );
        assert!(
            minting_receiver
                .iter()
                .any(|f| f.message.contains("ExecutionMode")),
            "minting a variant is selection even when receiving a decision: {}",
            Finding::messages(&minting_receiver)
        );

        let forwarding_receiver = judge(
            "fn forward(mode: ExecutionMode) -> ExecutionMode {\n    mode\n}\n",
        );
        assert!(
            forwarding_receiver.is_empty(),
            "forwarding a received decision without naming a variant is realization: {}",
            Finding::messages(&forwarding_receiver)
        );
    }
}
