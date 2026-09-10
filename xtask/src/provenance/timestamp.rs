//! Render a unix epoch second as an RFC 3339 UTC instant.
//!
//! A release provenance document states when the release it describes was
//! produced. A wall clock read makes that value differ on every run, which a
//! byte-compared artifact cannot survive, so the second comes from the tree
//! itself and this turns it into the format the documents declare.
//!
//! The conversion is `jiff`, with its default features off so nothing reads a
//! system time zone or bundles a zone database: an epoch second renders the
//! same text on every host. A hand-written era calculation stood here and was
//! a second copy of the one the benchmark harness carries.

use crate::provenance::ProvenanceError;

/// `seconds` since the unix epoch as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// # Errors
///
/// Returns an error when `seconds` names an instant outside the representable
/// range, which a declared `SOURCE_DATE_EPOCH` can and a commit date cannot.
pub fn rfc3339_utc(seconds: i64) -> Result<String, ProvenanceError> {
    let instant =
        jiff::Timestamp::from_second(seconds).map_err(|error| ProvenanceError::UnmeasuredFact {
            fact: "release timestamp".to_string(),
            reason: format!("{seconds} is not a representable unix second: {error}"),
        })?;
    Ok(instant.strftime("%Y-%m-%dT%H:%M:%SZ").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the format is what a CycloneDX or in-toto consumer parses, and the
    /// rendering has to survive a leap day, a century that is not a leap year,
    /// a century that is, and the epoch itself. Each case below is a known
    /// instant, so a change of conversion cannot pass by agreeing with itself.
    ///
    /// What it does not catch: leap seconds. Unix time does not carry them and
    /// neither does this.
    #[test]
    fn a_known_instant_renders_in_rfc_3339() {
        for (seconds, rendered) in [
            (0_i64, "1970-01-01T00:00:00Z"),
            (1, "1970-01-01T00:00:01Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (4_107_542_400, "2100-03-01T00:00:00Z"),
            (1_789_016_022, "2026-09-10T04:53:42Z"),
            (-1, "1969-12-31T23:59:59Z"),
        ] {
            assert_eq!(
                rfc3339_utc(seconds).expect("a representable second"),
                rendered,
                "at {seconds}"
            );
        }
    }

    /// WHY: a document rendered twice from the same tree must be byte
    /// identical, which is the whole reason the value is not read from a
    /// clock.
    #[test]
    fn the_same_second_renders_the_same_text() {
        assert_eq!(
            rfc3339_utc(1_789_016_022).expect("a representable second"),
            rfc3339_utc(1_789_016_022).expect("a representable second")
        );
    }

    /// WHY: `SOURCE_DATE_EPOCH` is declared by whoever runs the build, and a
    /// value naming no instant must be refused rather than rendered as a year
    /// the format has no room for. A silent wrong date in a provenance
    /// document is worse than a failed release.
    #[test]
    fn a_second_naming_no_instant_is_refused() {
        let error = rfc3339_utc(i64::MAX).expect_err("i64::MAX names no instant");
        assert!(
            error.to_string().contains("not a representable unix second"),
            "{error}"
        );
    }
}
