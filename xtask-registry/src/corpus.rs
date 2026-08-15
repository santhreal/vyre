//! Selection of release corpus cases for the gates that work on them.

use vyre_foundation::ir::Program;
use xtask::gate::GateError;

/// The generated release corpus, narrowed to one case when `--program` named
/// one. `verb` is what the calling gate does to a case, and appears in the
/// error a caller reads when the selection is empty.
pub fn selected_cases(
    selected: Option<&str>,
    verb: &str,
) -> Result<Vec<(String, Program)>, GateError> {
    let cases: Vec<(String, Program)> =
        vyre_foundation::optimizer::corpus::generate_release_corpus()
            .into_iter()
            .filter(|case| match selected {
                Some(id) => case.id == id,
                None => true,
            })
            .map(|case| (case.id, case.program))
            .collect();
    if cases.is_empty() {
        return Err(GateError::new(
            match selected {
                Some(id) => format!("no release corpus case is named `{id}`"),
                None => format!("the release corpus generated no case, so there is nothing to {verb}"),
            },
            format!("run the gate without --program to {verb} every generated case"),
        ));
    }
    Ok(cases)
}
