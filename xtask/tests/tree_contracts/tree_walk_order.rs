//! The shared tree walk yields one order for one tree.

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;
use xtask::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

/// How many siblings each directory of the fixture holds.
///
/// Wide enough that a filesystem returning its own order cannot be mistaken for
/// an ordered walk: ext4 and tmpfs both return hash order, and forty-eight names
/// landing in name order by chance is not a case worth ruling in.
const SIBLINGS: usize = 16;

/// Build a fixture tree, creating every entry in `order`.
///
/// The same names in both fixtures, so the only difference between the two trees
/// is the sequence the directory entries were written in.
fn fixture(order: impl Iterator<Item = usize> + Clone) -> TempDir {
    let temp = TempDir::new().expect("create a fixture directory");
    let root = temp.path();
    for index in order.clone() {
        let directory = root.join(format!("dir{index:02}"));
        fs::create_dir(&directory).expect("create a fixture subdirectory");
        for leaf in order.clone() {
            fs::write(directory.join(format!("leaf{leaf:02}.rs")), "\n").expect("write a leaf");
        }
    }
    for index in order {
        fs::write(root.join(format!("root{index:02}.rs")), "\n").expect("write a root file");
    }
    temp
}

/// Every entry the walk yields, relative to `root`.
fn walked(root: &Path) -> Vec<PathBuf> {
    tree_walk::pruned(root, BUILD_OUTPUT_AND_VCS)
        .map(|entry| {
            entry
                .expect("read a fixture entry")
                .path()
                .strip_prefix(root)
                .expect("an entry of the walked root")
                .to_path_buf()
        })
        .collect()
}

/// The walk yields names in order, whatever order the entries were created in.
///
/// WHY: `hygiene-matrix.json` renders one row per finding in walk order, so a
/// walk that passed readdir order through made a committed artifact a property
/// of the filesystem rather than of the source. Regenerating it on another
/// machine produced the same 329 rows in a different sequence, and the artifact
/// comparison reported a stale artifact for a tree nobody had changed. This does
/// not catch a renderer that reorders rows after the walk; it fixes the one
/// answer every gate reads the tree through.
#[test]
fn the_shared_walk_yields_names_in_order() {
    let ascending = fixture(0..SIBLINGS);
    let descending = fixture((0..SIBLINGS).rev());

    let walk = walked(ascending.path());
    let mut ordered = walk.clone();
    ordered.sort();
    assert_eq!(
        walk, ordered,
        "Fix: the shared tree walk must yield entries in name order; a gate that renders walk \
         order into a committed artifact otherwise writes different bytes per filesystem"
    );

    assert_eq!(
        walk,
        walked(descending.path()),
        "Fix: the shared tree walk must not depend on the order the entries were created in"
    );

    let leaves = walk
        .iter()
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .count();
    assert_eq!(
        leaves,
        SIBLINGS * SIBLINGS + SIBLINGS,
        "Fix: the fixture must present every name to the walk, or the ordering assertion above \
         proves nothing about a wide directory"
    );
}

/// A pruned directory name is pruned wherever it appears, and its subtree with it.
///
/// WHY: the ordering above is a property of the same iterator the prune rule
/// lives in. A walk rebuilt to sort its entries could drop the prune and pass
/// every assertion in the test above while reading whatever the last build left
/// in `target`.
#[test]
fn the_shared_walk_prunes_build_output_at_every_depth() {
    let temp = fixture(0..2);
    let root = temp.path();
    for parent in ["", "dir00"] {
        let generated = root.join(parent).join("target");
        fs::create_dir(&generated).expect("create a build output directory");
        fs::write(generated.join("generated.rs"), "\n").expect("write a generated file");
    }

    let walk = walked(root);
    assert!(
        walk.iter().all(|path| !path
            .components()
            .any(|component| component.as_os_str() == "target")),
        "Fix: the shared tree walk must prune build output at every depth; walked: {walk:?}"
    );
}
