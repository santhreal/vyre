use std::path::{Component, Path, PathBuf};

/// Absolute root of the checkout that compiled this binary.
///
/// `VYRE_CHECKOUT_ROOT` is declared in `.cargo/config.toml` as a checkout-
/// relative path, so reading it with `env!` records this checkout's absolute
/// location in the crate's dep-info. Without that input, a target directory
/// shared by several checkouts hands one checkout the lint binary another one
/// compiled, and the report then describes the wrong tree.
pub(crate) fn compiled_checkout_root() -> PathBuf {
    PathBuf::from(env!(
        "VYRE_CHECKOUT_ROOT",
        "Fix: run cargo from inside the vyre checkout so its .cargo/config.toml applies."
    ))
}

/// Absolute root of the checkout this tool was invoked in.
pub(crate) fn checkout_root() -> PathBuf {
    std::env::var_os("VYRE_CHECKOUT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(compiled_checkout_root)
}

pub(crate) fn workspace_relative(path: &Path) -> String {
    let workspace_root = checkout_root();
    if let Ok(relative) = path.strip_prefix(workspace_root) {
        return normalized_path(relative);
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
