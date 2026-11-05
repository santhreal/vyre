//! Parse an embedded TOML manifest, or exit with its parse error.
//!
//! The manifests these gates read are `include_str!`-ed at build time, so a
//! parse failure is a repository defect and not a runtime condition.

use serde::de::DeserializeOwned;

/// Parse embedded TOML, naming `path` in the error.
pub fn parse_embedded_toml<T>(path: &str, text: &str) -> Result<T, String>
where
    T: DeserializeOwned,
{
    toml::from_str::<T>(text).map_err(|error| format!("Fix: {path} is invalid TOML: {error}"))
}

/// Unwrap a parse result, exiting with status 2 on a malformed manifest.
pub fn data_or_exit<T>(result: &'static Result<T, String>) -> &'static T {
    match result {
        Ok(data) => data,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
