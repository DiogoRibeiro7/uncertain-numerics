# Contributing to uncertain-numerics

Thank you for considering a contribution. This project has a narrow goal: numerical methods whose uncertainty statements are explicit, tested, and honest about their assumptions. The guidelines below exist to keep that goal intact as the code grows.

## Ground rules

- Be respectful. The project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Open an issue before starting large work so the mathematical contract can be agreed first. Small fixes can go straight to a pull request.
- Every numerical claim in code or documentation must be backed by a test: an analytic reference, an independent deterministic computation, or a calibration study.

## Development setup

You need a stable Rust toolchain at or above the minimum supported version (1.85) with `rustfmt` and `clippy` installed. There are no other build dependencies.

```sh
git clone https://github.com/DiogoRibeiro7/uncertain-numerics.git
cd uncertain-numerics
cargo test --all-features
```

Optional tools used by CI:

```sh
cargo install cargo-deny    # license, advisory and duplicate-dependency checks
```

## Before you open a pull request

Run the same checks CI runs. All of them must pass:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features --all-targets
cargo test --all-features --doc
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny check            # if installed
```

On Windows PowerShell, set the environment variable first: `$env:RUSTDOCFLAGS = "-D warnings"`.

The crate enables `clippy::pedantic` and `missing_docs`. Prefer fixing a lint over silencing it. When silencing is the right call (for example an exact float comparison that is intended), scope the `allow` as narrowly as possible and say why in a comment.

## Coding standards

- **Validate at the boundary.** Constructors and public methods reject non-finite values, dimension mismatches, and invalid parameters with a typed error. Never let a `NaN` propagate silently.
- **Errors, not panics.** Public functions return `Result` with a crate error type that implements `std::error::Error` and `Display`. Document every failure mode in an `# Errors` section.
- **Keep assumptions explicit.** A zero prior mean, a fixed jitter, a symmetry tolerance: each is a named parameter or a documented constant, never an implicit default that changes results.
- **No silent regularization.** Jitter is supplied by the caller and is never escalated automatically to make a factorization succeed.
- **Cholesky, not inverses.** Solve linear systems from a reusable factorization; do not form explicit inverses.
- **Document the mathematics.** Doc comments state the formula being implemented, in plain text or LaTeX, and the conditions under which it holds.
- **`unsafe` is forbidden** crate-wide.

## Testing standards

Tests are organized in two layers:

- unit tests next to the code cover invariants, validation, and closed-form special cases;
- integration tests in `tests/` are scientific studies: analytic fixtures, independent deterministic quadrature, coverage and calibration simulations, misspecification and numerical-stability regressions.

When adding a numerical method:

1. add at least one fixture with a known analytic answer;
2. compare against an independent deterministic computation where one exists;
3. if the method reports uncertainty, add a calibration check under the assumed model and document any known miscalibration in `docs/`;
4. use explicit, justified tolerances and name them as constants.

Deterministic pseudo-random sequences are preferred over external RNG crates so that studies are reproducible without extra dependencies.

## Documentation

- Public items need doc comments; CI fails on missing documentation.
- Design notes and validation write-ups live in [`docs/`](docs/). Add one when a method has non-obvious statistical behavior worth recording.
- Keep [`ROADMAP.md`](ROADMAP.md) in sync when a milestone item is completed or re-scoped.
- Add an entry under *Unreleased* in [`CHANGELOG.md`](CHANGELOG.md) for user-visible changes.

## Branches, commits and pull requests

- Branch from `main` using a short prefix: `feat/`, `fix/`, `test/`, `docs/`, `chore/`.
- Write commit subjects in the imperative mood, at most 72 characters, for example `Add finite uniform measure`. A body explaining *why* is welcome for non-trivial changes.
- Keep pull requests focused. One method, one fix, or one refactor per PR reviews far better than a mixture.
- Fill in the pull request template, including the mathematical contract and how it was validated.

## Minimum supported Rust version

The MSRV is declared in `Cargo.toml` and checked in CI. Raising it is a deliberate change that must be mentioned in the changelog, and it will not happen in a patch release.

## Licensing of contributions

By submitting a contribution you agree that it is licensed under the same terms as the project, MIT OR Apache-2.0, without any additional terms or conditions.
