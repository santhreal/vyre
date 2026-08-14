//! Shared OP_MATRIX release-backend row classification.

pub(crate) fn count_supported_release_backend_rows(rows: &[String]) -> usize {
    rows.iter()
        .filter(|row| {
            parse_release_backend_row(row)
                .is_some_and(|(_op, _backend, status)| status == "supported")
        })
        .count()
}

fn parse_release_backend_row(row: &str) -> Option<(&str, &str, &str)> {
    let (prefix, status) = row.rsplit_once(':')?;
    let (op, backend) = prefix.rsplit_once(':')?;
    Some((op, backend, status))
}
