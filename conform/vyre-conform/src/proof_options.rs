//! Shard specification and the shared `plan`/`prove` command-line option parser.

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShardSpec {
    pub(crate) index: usize,
    pub(crate) count: usize,
}

#[derive(Debug)]
pub(crate) struct ProofOptions {
    pub(crate) out: Option<String>,
    pub(crate) certificates_dir: Option<String>,
    pub(crate) backend_filter: String,
    pub(crate) ops_filter: String,
    pub(crate) shard: Option<ShardSpec>,
}

pub(crate) fn parse_proof_options(
    command: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ProofOptions, String> {
    let mut out = None;
    let mut certificates_dir = None::<String>;
    let mut backend_filter = std::env::var("VYRE_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "all".to_string());
    let mut ops_filter = "all".to_string();
    let mut shard = None::<ShardSpec>;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => {
                out = Some(next_option_value(&mut it, "--out")?);
            }
            "--certificates" if command == "prove" => {
                certificates_dir = Some(next_option_value(&mut it, "--certificates")?);
            }
            "--backend" => {
                backend_filter = next_option_value(&mut it, "--backend")?;
            }
            "--ops" => {
                ops_filter = next_option_value(&mut it, "--ops")?;
            }
            "--shard" => {
                let value = next_option_value(&mut it, "--shard")?;
                shard = Some(parse_shard_spec(&value)?);
            }
            other => {
                return Err(format!(
                    "unknown flag `{other}`. Fix: use `vyre-conform {command} --out <path> [--backend <all|backend-id>] [--ops <all|op_id>] [--shard <index>/<count>]`."
                ));
            }
        }
    }
    Ok(ProofOptions {
        out,
        certificates_dir,
        backend_filter,
        ops_filter,
        shard,
    })
}

pub(crate) fn next_option_value(
    it: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, String> {
    it.next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing value for {flag}. Fix: pass a non-empty value."))
}

fn parse_shard_spec(value: &str) -> Result<ShardSpec, String> {
    let (index, count) = value.split_once('/').ok_or_else(|| {
        format!("invalid shard `{value}`. Fix: use zero-based `--shard <index>/<count>`, for example `--shard 0/8`.")
    })?;
    let index = index.parse::<usize>().map_err(|error| {
        format!("invalid shard index `{index}`: {error}. Fix: use a zero-based integer.")
    })?;
    let count = count.parse::<usize>().map_err(|error| {
        format!("invalid shard count `{count}`: {error}. Fix: use a positive integer.")
    })?;
    if count == 0 {
        return Err("invalid shard count `0`. Fix: shard count must be positive.".to_string());
    }
    if index >= count {
        return Err(format!(
            "invalid shard `{value}`. Fix: shard index must be less than shard count."
        ));
    }
    Ok(ShardSpec { index, count })
}
