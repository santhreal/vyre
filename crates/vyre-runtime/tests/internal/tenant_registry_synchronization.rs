use super::*;

/// WHY: issued handles and generations must survive lock poisoning rather than
/// being cleared; this checks identity through registration, lookup and retirement.
#[test]
fn poisoned_maps_preserve_issued_identity() {
    let registry = TenantRegistry::new();
    let issued = registry.register("before-panic").expect("registration");
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _tenants = registry.tenants.write().expect("unpoisoned lock");
        panic!("injected tenant-map lock failure");
    }));
    assert!(poisoned.is_err());
    let found = registry
        .lookup(issued.id())
        .expect("issued handle retained");
    assert_eq!(found.generation(), issued.generation());
    registry.unregister(issued.id()).expect("retirement");
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _generations = registry.generations.lock().expect("unpoisoned lock");
        panic!("injected generation-map lock failure");
    }));
    assert!(poisoned.is_err());
    let next = registry.register("after-panic").expect("registration");
    assert_eq!(next.id(), issued.id());
    assert_eq!(next.generation(), issued.generation() + 1);
    assert_eq!(registry.active_tenants().len(), 1);
}
