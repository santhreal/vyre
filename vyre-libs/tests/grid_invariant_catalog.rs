//! A registered program that needs more than one workgroup names the grid.
//!
//! WHY: a concrete backend derives a 1D grid from the largest writable buffer,
//! so a program whose writable footprint exceeds one workgroup's lanes runs in
//! several workgroups. A body that names no grid-varying index then performs the
//! same global writes in every workgroup. That is invisible in the reference,
//! which runs workgroups one after another so redundant work converges, and it
//! is a data race on a device: a stage that reads back a buffer an earlier stage
//! wrote will rescale one lane and lose another. Workgroup-private scratch makes
//! it worse, because each workgroup then recomputes the whole pipeline from its
//! own copy.
//!
//! The population is every operation this build registers, read at run time, so
//! a new composition or primitive witness with this shape turns the suite red
//! until it either indexes by a grid-varying id or confines itself to workgroup
//! zero.
//!
//! What this does not prove: that a program naming a grid-varying index
//! partitions its output correctly. It rejects the shape that cannot be correct
//! under any grid.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::visit::walk_exprs;

fn lanes(program: &Program) -> u64 {
    program
        .workgroup_size()
        .into_iter()
        .map(u64::from)
        .product()
}

fn max_writable_count(program: &Program) -> u64 {
    program
        .buffers()
        .iter()
        .filter(|decl| matches!(decl.access(), BufferAccess::ReadWrite) || decl.is_output())
        .map(|decl| u64::from(decl.count()))
        .max()
        .unwrap_or(1)
}

fn names_grid_varying_index(program: &Program) -> bool {
    let mut found = false;
    walk_exprs(program, |expr| {
        if matches!(expr, Expr::LogicalTileId { .. } | Expr::LogicalIndex { .. }) {
            found = true;
        }
    });
    found
}

fn store_only_program(workgroup_size: [u32; 3], entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(8)],
        workgroup_size,
        entry,
    )
}

#[test]
fn a_first_logical_tile_guard_counts_as_naming_the_grid() {
    let guarded = store_only_program(
        [1, 1, 1],
        vec![Node::if_then(
            Expr::is_first_logical_tile(),
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        )],
    );
    assert!(
        names_grid_varying_index(&guarded),
        "Fix: the walk must reach an If condition, or this contract cannot see a guard"
    );

    let unguarded = store_only_program(
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    assert!(
        !names_grid_varying_index(&unguarded),
        "Fix: a body with no logical tile or point index must not read as grid-varying"
    );
    assert!(
        max_writable_count(&unguarded) > lanes(&unguarded),
        "Fix: the control program must be one this contract would examine"
    );

    let region_wrapped = store_only_program(
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from("control"),
            source_region: None,
            body: std::sync::Arc::new(vec![Node::if_then(
                Expr::is_first_logical_tile(),
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            )]),
        }],
    );
    assert!(
        names_grid_varying_index(&region_wrapped),
        "Fix: the walk must descend into Region bodies, which every composition uses"
    );
}

#[test]
fn every_multi_workgroup_registered_program_names_a_grid_varying_index() {
    // Naming the library catalog links this crate into the test binary, and its
    // inventory registrations come with it. An integration test that touches
    // only `vyre-foundation` links no registrations at all, and the sweep then
    // examines an empty registry and proves nothing.
    let library_entries = vyre_libs::operation_catalog::all_entries().count();
    assert!(
        library_entries > 0,
        "Fix: this build registers no library operation, so the sweep has no population to read"
    );

    let mut registered = 0;
    let mut examined = Vec::new();
    let mut offenders = Vec::new();
    // Every operation this build registers, whatever tier or feature put it
    // there, not just the library view: a primitive witness is dispatched the
    // same way.
    for entry in OperationRegistry::global().iter() {
        let id = entry.id;
        let Some(program) = entry.program() else {
            continue;
        };
        registered += 1;
        if max_writable_count(&program) <= lanes(&program) {
            continue;
        }
        examined.push(id);
        if !names_grid_varying_index(&program) {
            offenders.push(id);
        }
    }

    // A narrow feature set can register nothing that writes past one workgroup,
    // and that is a fact about the build rather than a hole in the sweep: the
    // predicate is proved able to fail by the control program in
    // `a_first_workgroup_guard_counts_as_naming_the_grid`. What would make this
    // sweep vacuous is reading fewer programs than the catalog it just counted.
    assert!(
        registered >= library_entries,
        "Fix: the sweep read {registered} programs against a catalog of {library_entries}; a registration with no program body cannot be checked for this shape"
    );
    assert!(
        offenders.is_empty(),
        "{} of {} multi-workgroup compositions write global memory identically in every workgroup: {}. Fix: index the writes by a grid-varying logical identity, or wrap the body in Node::if_then(Expr::is_first_logical_tile(), ..).",
        offenders.len(),
        examined.len(),
        offenders.join(", ")
    );
}
