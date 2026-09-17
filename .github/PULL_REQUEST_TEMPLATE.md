## Summary

<!-- What does this change and why? Link the roadmap item or issue if there is one. -->

## Mathematical contract

<!-- For numerical changes: state the formula, assumptions, and tolerances involved,
     or write "not applicable" for tooling-only changes. -->

## Validation

<!-- How was the change checked? Analytic reference, independent deterministic
     quadrature, calibration study, property test, ... -->

## Checklist

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --all-features` passes
- [ ] `cargo doc --no-deps --all-features` builds without warnings
- [ ] public items are documented, including `# Errors` sections
- [ ] `CHANGELOG.md` has an entry under *Unreleased* (when user-visible)
- [ ] `ROADMAP.md` is updated (when a milestone item changes)
