//! The class closed here: a rewrite that speculates a memory read across a
//! guard it never proved.
//!
//! Loop-invariant hoisting moves a binding to the point above the loop, where
//! it runs once whatever the trip count is. For arithmetic that costs a
//! register. For a load it changes which memory the program touches: a loop the
//! header does not prove is entered may run zero times, and a read the body
//! would have skipped becomes a read the program performs. The reference
//! interpreter absorbs an out-of-range read as a zero fill and records it; a
//! device does not, so the absorption hides the divergence rather than removing
//! it.
//!
//! The rule is therefore split by what a header proves. Literal bounds that
//! prove at least one iteration license the load. Literal bounds that prove
//! none, and bounds that prove nothing, do not. Arithmetic is unaffected in all
//! three cases, which is what keeps this a bound on speculation rather than a
//! way of switching the pass off.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_licm::LoopLicm;
use vyre_foundation::transform::licm::apply_licm;

/// A loop over `from..to` binding `base` to a read-only load and a second name
/// to invariant arithmetic, then storing both.
fn loop_over(from: Expr, to: Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::loop_for(
            "i",
            from,
            to,
            vec![
                Node::let_bind("base", Expr::load("src", Expr::u32(0))),
                Node::let_bind("bump", Expr::add(Expr::u32(1), Expr::u32(2))),
                Node::store(
                    "out",
                    Expr::var("i"),
                    Expr::add(Expr::var("base"), Expr::var("bump")),
                ),
            ],
        )],
    )
}

/// Names bound at the top level of the entry body, which is where a hoist lands.
fn hoisted_names(program: &Program) -> Vec<String> {
    let nodes = match program.entry() {
        [Node::Region {
            generator, body, ..
        }] if generator.as_str() == Program::ROOT_REGION_GENERATOR => body.as_ref(),
        nodes => nodes,
    };
    nodes
        .iter()
        .filter_map(|node| match node {
            Node::Let { name, .. } => Some(name.as_str().to_string()),
            _ => None,
        })
        .collect()
}

/// WHY: this is the capability the rule has to keep. Bounds that prove the body
/// runs license the load, so gating on entry must not cost the hoist that
/// motivated the read-only rule in the first place.
#[test]
fn literal_bounds_that_prove_an_iteration_release_the_load() {
    let result = LoopLicm::transform(loop_over(Expr::u32(0), Expr::u32(4)));
    assert_eq!(hoisted_names(&result.program), vec!["base", "bump"]);
    result
        .program
        .validate()
        .expect("the hoisted program must still validate");
}

/// WHY: goes red against hoisting on an unproven header. A runtime bound may be
/// empty, and the hoisted load is then a read the program never performed. The
/// arithmetic beside it still leaves, which is what distinguishes this from
/// refusing to hoist at all.
#[test]
fn a_runtime_bound_holds_the_load_and_releases_the_arithmetic() {
    let result = LoopLicm::transform(loop_over(Expr::u32(0), Expr::var("count")));
    assert_eq!(hoisted_names(&result.program), vec!["bump"]);
    assert!(
        result.changed,
        "the invariant arithmetic still leaves a loop of unknown trip count"
    );
}

/// WHY: the third thing a header can prove. An empty literal range runs the body
/// never, so nothing in it may be speculated, and the pass must not treat
/// "literal" as "proven entered".
#[test]
fn literal_bounds_that_prove_no_iteration_hold_the_load() {
    let result = LoopLicm::transform(loop_over(Expr::u32(4), Expr::u32(4)));
    assert_eq!(hoisted_names(&result.program), vec!["bump"]);
}

/// WHY: the pipeline adapter and the pass are one owner of this rule, and a
/// second entry point that speculates where the first refuses is the defect
/// class this whole file exists for.
#[test]
fn the_pipeline_adapter_holds_the_same_line() {
    let after = apply_licm(&loop_over(Expr::u32(0), Expr::var("count")));
    assert_eq!(hoisted_names(&after), vec!["bump"]);

    let entered = apply_licm(&loop_over(Expr::u32(0), Expr::u32(4)));
    assert_eq!(hoisted_names(&entered), vec!["base", "bump"]);
}
