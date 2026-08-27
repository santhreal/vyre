//! Loop-carrier detection: uncarriered assigns are flagged and carrier summaries match the descriptor walk.
use std::collections::{BTreeMap, BTreeSet};
use vyre_debug::fixtures::loop_carry_smoke;
use vyre_debug::{carrier_summary, find_uncarriered_assigns};
use vyre_foundation::ir::{Expr, Node, Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;

#[path = "program_fixtures/mod.rs"]
mod program_fixtures;

fn lower_library_program(program: &Program) -> vyre_lower::KernelDescriptor {
    let graph = ProgramGraph::from_program("vyre-debug::loop-carrier", program.clone())
        .expect("library fixture must form a valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("library fixture must have a valid logical domain");
    let schedule = vyre_megakernel::baseline_schedule(&logical);
    let phase = schedule.phases[0].id;
    vyre_lower::lower_scheduled(program, &schedule, phase)
        .expect("library fixture must lower through its selected schedule")
        .into_descriptor()
}

#[test]
fn find_uncarriered_assigns_smoke_program_returns_empty() {
    let p = loop_carry_smoke();
    let desc = vyre_lower::lower_physical(&p)
        .map(|lowered| lowered.into_descriptor())
        .unwrap();
    let uncarriered = find_uncarriered_assigns(&p, &desc);
    assert!(uncarriered.is_empty());
}

#[test]
fn find_uncarriered_assigns_flags_a_loop_with_no_carrier() {
    let p = program_fixtures::program_over_out(
        [64, 1, 1],
        vec![
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(10),
                vec![Node::assign("x", Expr::add(Expr::var("x"), Expr::u32(1)))],
            ),
        ],
    );
    let mut desc = vyre_lower::lower_physical(&p)
        .map(|lowered| lowered.into_descriptor())
        .unwrap();

    // Manually strip LoopCarrier from the descriptor.
    // `LoopCarrierFinal` was consolidated into `LoopCarrier` upstream;
    // the strip set is now just the two remaining variants.
    fn strip_carriers(body: &mut vyre_lower::KernelBody) {
        body.ops.retain(|op| {
            !matches!(
                op.kind,
                vyre_lower::KernelOpKind::LoopCarrier { .. }
                    | vyre_lower::KernelOpKind::LoopCarrierEnd { .. }
            )
        });
        for child in &mut body.child_bodies {
            strip_carriers(child);
        }
    }
    strip_carriers(&mut desc.body);

    let uncarriered = find_uncarriered_assigns(&p, &desc);
    assert_eq!(uncarriered.len(), 1);
    assert_eq!(uncarriered[0].name, "x");
    assert!(!uncarriered[0].has_carrier_op);
}

#[test]
fn carrier_summary_counts_match_descriptor_walk() {
    let p = vyre_libs::parsing::python::lex::python312_lexer("hs", "tt", "ts", "tl", "tc", 4);
    let desc = lower_library_program(&p);
    let summary = carrier_summary(&desc);

    // Walk the descriptor directly and build the ground-truth maps.
    // Semantics:
    //   carrier_reads  <- LoopCarrier       (read of carrier slot)
    //   carrier_writes <- LoopCarrierInit   (seed write before loop)
    //   carrier_finals <- LoopCarrierEnd    (commit write at iteration end)
    let mut reads = BTreeMap::new();
    let mut writes = BTreeMap::new();
    let mut finals = BTreeMap::new();
    fn walk_body(
        body: &vyre_lower::KernelBody,
        r: &mut BTreeMap<String, usize>,
        w: &mut BTreeMap<String, usize>,
        f: &mut BTreeMap<String, usize>,
    ) {
        for op in &body.ops {
            match &op.kind {
                vyre_lower::KernelOpKind::LoopCarrier { name } => {
                    *r.entry(name.to_string()).or_insert(0) += 1;
                }
                vyre_lower::KernelOpKind::LoopCarrierInit { name } => {
                    *w.entry(name.to_string()).or_insert(0) += 1;
                }
                vyre_lower::KernelOpKind::LoopCarrierEnd { name } => {
                    *f.entry(name.to_string()).or_insert(0) += 1;
                }
                _ => {}
            }
        }
        for child in &body.child_bodies {
            walk_body(child, r, w, f);
        }
    }
    walk_body(&desc.body, &mut reads, &mut writes, &mut finals);
    assert_eq!(summary.carrier_reads, reads, "carrier_reads mismatch");
    assert_eq!(summary.carrier_writes, writes, "carrier_writes mismatch");
    assert_eq!(summary.carrier_finals, finals, "carrier_finals mismatch");
    // A program with loops must have non-empty finals (LoopCarrierEnd ops exist).
    assert!(
        !summary.carrier_finals.is_empty(),
        "carrier_finals is empty on a descriptor with loop-carried variables; \
         expected LoopCarrierEnd ops to be counted here"
    );
}

#[test]
fn carrier_summary_includes_function_locals() {
    let p = vyre_libs::parsing::python::lex::python312_lexer("hs", "tt", "ts", "tl", "tc", 4);
    let desc = lower_library_program(&p);
    let summary = carrier_summary(&desc);
    // Derive the expected names from the descriptor the summary just walked
    // rather than pinning one lexer's variable spelling: a local recorded here
    // is a naga local named after a carried variable, so every carrier name the
    // walk found is the only vocabulary a local may be built from.
    let carried = summary
        .carrier_reads
        .keys()
        .chain(summary.carrier_writes.keys())
        .chain(summary.carrier_finals.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    assert!(
        !carried.is_empty(),
        "the lexer descriptor carries no loop variable, so this test proves nothing"
    );
    assert!(
        !summary.function_locals.is_empty(),
        "carrier_finals named {carried:?} but no emitted local was recorded"
    );
    assert!(
        summary
            .function_locals
            .iter()
            .any(|local| carried.iter().any(|name| local.contains(name.as_str()))),
        "no recorded local names a carried variable; carried: {carried:?}, locals: {:?}",
        summary.function_locals
    );
}
