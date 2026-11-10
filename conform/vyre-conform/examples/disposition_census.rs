//! Census of the derived law disposition of every operation in the live registry.
//!
//! Prints one tab-separated row per operation: id, derived disposition,
//! recorded absence, declared laws, confirmed laws, refuted laws, source file,
//! defect count, and the evidence for the disposition.

fn main() {
    let registry = vyre_registry_link::operation::live_operation_registry();
    let mut rows = Vec::new();
    for entry in registry.iter() {
        let survey = vyre_conform::law_survey::survey_operation(&entry);
        let defects = vyre_conform::law_survey::judge(&entry, &survey);
        let recorded = entry.absence().map_or("none", |absence| absence.name());
        rows.push(format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            entry.id,
            survey.disposition.name(),
            recorded,
            entry.laws.join("+"),
            survey.confirmed.join("+"),
            survey
                .refuted
                .iter()
                .map(|(law, _)| *law)
                .collect::<Vec<_>>()
                .join("+"),
            entry.source_file,
            defects.len(),
            survey.evidence().replace('\t', " ")
        ));
    }
    rows.sort();
    for row in &rows {
        println!("{row}");
    }
    eprintln!("rows={}", rows.len());
}
