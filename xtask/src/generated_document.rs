//! Writing one generated document to the checkout.
//!
//! Both documentation writers create the directory and then write the file, and
//! both report the same two failures. Two copies of that meant a writer could
//! learn to create a missing parent while the other still failed on it.

use std::fs;
use std::path::Path;

use crate::gate::GateError;

/// Write `content` to `path`, creating the directory it lives in.
///
/// The directory is created because a generated document is the first file in
/// its directory the first time it is written, and a reader who asked for the
/// document did not ask to make a folder for it first.
pub fn write(path: &Path, content: &str) -> Result<(), GateError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            GateError::new(
                format!("could not create `{}`: {error}", parent.display()),
                "check the checkout is writable",
            )
        })?;
    }
    fs::write(path, content).map_err(|error| {
        GateError::new(
            format!("could not write `{}`: {error}", path.display()),
            "check the checkout is writable",
        )
    })
}
