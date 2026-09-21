//! Tests for pass registry completeness and deletion evidence invariants (Section 188.5).

use structure_gate::workspace_root;

#[test]
fn transform_mod_contains_classifications_for_every_transform_file() {
    let root = workspace_root();
    let transform_dir = root.join("vyre-foundation/src/transform");
    let mod_rs_path = transform_dir.join("mod.rs");
    let mod_source = std::fs::read_to_string(&mod_rs_path)
        .expect("vyre-foundation/src/transform/mod.rs must be readable");

    assert!(
        mod_source.contains("FOUNDATION_TRANSFORM_CLASSIFICATIONS"),
        "vyre-foundation/src/transform/mod.rs must define FOUNDATION_TRANSFORM_CLASSIFICATIONS"
    );

    let entries = std::fs::read_dir(&transform_dir).expect("transform directory must be readable");

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if name == "mod" || name.is_empty() {
            continue;
        }
        let expected_needle = format!("name: \"{name}\"");
        assert!(
            mod_source.contains(&expected_needle),
            "transform module `{name}` found on disk but not registered in FOUNDATION_TRANSFORM_CLASSIFICATIONS in mod.rs"
        );
    }
}
