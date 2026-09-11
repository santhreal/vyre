use std::path::{Component, Path, PathBuf};

/// Absolute root of the checkout being linted, when the environment names one.
///
/// This crate is published, so the root must not be baked in at compile time.
/// A path recorded on the machine that built the binary means nothing on the
/// machine that runs it, and reading it with `env!` would make the crate
/// uncompilable anywhere that does not carry this repository's cargo config.
///
/// There is also no need to bake it. `vyre-lints` takes `--workspace-root`, and
/// every root it scans is derived from that argument, so the root already has
/// one owner and this function only supplies the prefix used to shorten a path
/// for display. When cargo exports `VYRE_CHECKOUT_ROOT` it is used; otherwise
/// `workspace_relative` falls back to the crate-directory suffix, which yields
/// the same text for any path inside the workspace.
pub(crate) fn checkout_root() -> Option<PathBuf> {
    std::env::var_os("VYRE_CHECKOUT_ROOT").map(PathBuf::from)
}

pub(crate) fn workspace_relative(path: &Path) -> String {
    if let Some(root) = checkout_root() {
        if let Ok(relative) = path.strip_prefix(&root) {
            return normalized_path(relative);
        }
    }

    let mut components = path.components();
    let mut suffix = None;
    while let Some(component) = components.next() {
        let Component::Normal(name) = component else {
            continue;
        };
        let name = name.to_string_lossy();
        if name == "docs"
            || name == "conform"
            || name == "xtask"
            || name == "vyre"
            || name.starts_with("vyre-")
        {
            suffix = Some(PathBuf::from(component.as_os_str()).join(components.as_path()));
        }
    }

    normalized_path(suffix.as_deref().unwrap_or(path))
}

fn normalized_path(path: &Path) -> String {
    let path = path.to_string_lossy();
    if path.contains('\\') {
        path.replace('\\', "/")
    } else {
        path.into_owned()
    }
}
