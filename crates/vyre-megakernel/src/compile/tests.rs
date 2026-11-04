use super::*;
use crate::ArtifactNodeId;
use vyre_foundation::ir::Program;

fn program(workgroup: [u32; 3]) -> Vec<u8> {
    Program::wrapped(Vec::new(), workgroup, Vec::new())
        .canonical_wire_bytes()
        .expect("fixture program encodes")
}

fn record(node: u32, workgroup: [u32; 3]) -> GeometryRecord {
    crate::geometry_fixtures::geometry(node, node, workgroup)
}

fn node(id: u32, workgroup: [u32; 3]) -> NodeRecord {
    NodeRecord {
        id: ArtifactNodeId(id),
        name: format!("n{id}"),
        program: program(workgroup),
    }
}

/// WHY: the workgroup a source program declares is an input to the search.
/// Emission used to rewrite it while lowering, so the program the artifact
/// authenticated and the module the device ran disagreed on the one field a
/// launch cannot recover from. Freezing happens once, here, and the artifact
/// carries the result.
#[test]
fn a_recorded_program_declares_the_workgroup_the_search_selected() {
    let frozen = frozen_nodes(
        &[node(0, [8, 1, 1]), node(1, [32, 1, 1])],
        &[record(0, [32, 1, 1]), record(1, [32, 1, 1])],
    )
    .expect("both nodes carry selected geometry");

    for record in &frozen {
        let program = Program::from_wire(&record.program).expect("a frozen program decodes");
        assert_eq!(program.workgroup_size, [32, 1, 1], "node {}", record.id.0);
    }
    assert_eq!(
        frozen[1].program,
        program([32, 1, 1]),
        "a program already at the selected shape is carried through unchanged"
    );
}

/// WHY: a node with no selected geometry has no shape to be frozen at, and
/// re-encoding it at its declared shape would reintroduce exactly the
/// disagreement freezing exists to end.
#[test]
fn a_node_without_selected_geometry_is_refused() {
    let error = frozen_nodes(&[node(0, [8, 1, 1])], &[record(1, [32, 1, 1])])
        .expect_err("a node with no geometry cannot be frozen");
    assert_eq!(
        error
            .diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.as_deref()),
        Some("planner.geometry[0]")
    );
}
