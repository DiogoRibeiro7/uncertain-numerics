# Security policy

## Supported versions

Security fixes are applied to the `main` branch and to the latest release published on crates.io. Older releases do not receive backports.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub private vulnerability reporting instead:
<https://github.com/DiogoRibeiro7/uncertain-numerics/security/advisories/new>

You can expect an acknowledgement within seven days. Once the report is confirmed, a fix and a coordinated disclosure date are agreed with the reporter before any advisory is published.

## Scope

`uncertain-numerics` is a pure-Rust library with `unsafe_code = "forbid"` and a single dependency (`nalgebra`). Its public API validates every input and returns typed errors rather than panicking.

The following are treated as security-relevant and welcome through the private channel:

- panics or unbounded resource use reachable through the public API with attacker-controlled input;
- vulnerabilities in the dependency tree that affect this crate (also caught by `cargo deny check`).

Numerical inaccuracy, miscalibrated posterior uncertainty, or convergence problems are correctness bugs, not vulnerabilities. Please report them as ordinary issues with a minimal reproduction.
