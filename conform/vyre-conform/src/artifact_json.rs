//! Bounded reads, writes, and typed field extraction for JSON proof artifacts.

pub(crate) fn write_json_artifact(
    out: &str,
    json: String,
    artifact_kind: &str,
) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "failed to create {artifact_kind} directory `{}`: {error}. Fix: choose a writable --out parent.",
                    parent.display()
                )
            })?;
        }
    }
    std::fs::write(out, json).map_err(|error| {
        format!(
            "failed to write {artifact_kind} `{out}`: {error}. Fix: choose a writable --out path."
        )
    })
}

const MAX_PROVE_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;

pub(crate) fn read_prove_artifact_bounded(path: &str) -> Result<String, String> {
    let mut reader = std::fs::File::open(path).map_err(|error| {
        format!(
            "failed to read certificate `{path}`: {error}. Fix: pass a readable prove artifact."
        )
    })?;
    let mut bytes = Vec::new();
    let mut total = 0u64;
    let mut chunk = [0u8; 8192];
    loop {
        let read = std::io::Read::read(&mut reader, &mut chunk).map_err(|error| {
            format!(
                "failed to read certificate `{path}`: {error}. Fix: pass a readable prove artifact."
            )
        })?;
        if read == 0 {
            return String::from_utf8(bytes).map_err(|error| {
                format!(
                    "certificate `{path}` is not UTF-8: {error}. Fix: pass a valid JSON prove artifact."
                )
            });
        }
        let read = read as u64;
        total = total.saturating_add(read);
        if total > MAX_PROVE_ARTIFACT_BYTES {
            return Err(format!(
                "certificate `{path}` exceeds {MAX_PROVE_ARTIFACT_BYTES} byte merge cap. Fix: shard or trim the prove artifact before merging."
            ));
        }
        bytes.extend_from_slice(&chunk[..read as usize]);
    }
}

pub(crate) fn value_field<'a>(
    value: &'a serde_json::Value,
    field: &str,
    path: &str,
) -> Result<&'a serde_json::Value, String> {
    value
        .get(field)
        .ok_or_else(|| format!("certificate `{path}` missing `{field}`. Fix: regenerate it."))
}

pub(crate) fn string_field<'a>(
    value: &'a serde_json::Value,
    field: &str,
    path: &str,
) -> Result<&'a str, String> {
    value_field(value, field, path)?.as_str().ok_or_else(|| {
        format!("certificate `{path}` field `{field}` must be a string. Fix: regenerate it.")
    })
}

pub(crate) fn u32_field(value: &serde_json::Value, field: &str, path: &str) -> Result<u32, String> {
    let raw = value_field(value, field, path)?.as_u64().ok_or_else(|| {
        format!(
            "certificate `{path}` field `{field}` must be an unsigned integer. Fix: regenerate it."
        )
    })?;
    u32::try_from(raw).map_err(|_| {
        format!("certificate `{path}` field `{field}` exceeds u32::MAX. Fix: regenerate it.")
    })
}

pub(crate) fn usize_field(
    value: &serde_json::Value,
    field: &str,
    path: &str,
) -> Result<usize, String> {
    let raw = value_field(value, field, path)?.as_u64().ok_or_else(|| {
        format!(
            "certificate `{path}` field `{field}` must be an unsigned integer. Fix: regenerate it."
        )
    })?;
    usize::try_from(raw).map_err(|_| {
        format!("certificate `{path}` field `{field}` exceeds usize::MAX. Fix: regenerate it.")
    })
}
