use std::path::{Component, Path, PathBuf};

pub(crate) fn workspace_relative(path: &Path) -> String {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."));
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
