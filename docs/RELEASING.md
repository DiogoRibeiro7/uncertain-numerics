# Releasing

Releases are driven by git tags. Pushing a tag of the form `vX.Y.Z` runs the [release workflow](../.github/workflows/release.yml), which:

1. checks that the tag matches the version in `Cargo.toml`;
2. runs the test suite and a `cargo publish --dry-run`;
3. publishes the crate to crates.io through trusted publishing;
4. creates a GitHub release whose notes are the matching `CHANGELOG.md` section.

## Release checklist

1. Make sure `main` is green in CI.
2. On a branch, bump `version` in `Cargo.toml` following semantic versioning.
3. In `CHANGELOG.md`, rename the *Unreleased* section to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty *Unreleased* section above it, and update the link references at the bottom.
4. Review `ROADMAP.md` and tick or re-scope items as needed.
5. Open a pull request titled `Release X.Y.Z`, get it merged.
6. Tag the merge commit and push the tag:

   ```sh
   git checkout main
   git pull
   git tag -a vX.Y.Z -m "uncertain-numerics X.Y.Z"
   git push origin vX.Y.Z
   ```

7. Watch the release workflow. If the version is already on crates.io, for example after a manual first publication, the publish job is skipped and the GitHub release is still created. If it fails before publishing, fix the problem, delete the tag locally and remotely, and tag again. If it fails after publishing, do not re-tag: crates.io versions are immutable, so bump to the next patch version instead.

## One-time setup

### First publication

Trusted publishing can only be configured for a crate that already exists on crates.io, so the very first version must be published manually by a crate owner:

```sh
cargo login            # paste an API token from https://crates.io/settings/tokens
cargo publish --dry-run
cargo publish
```

Immediately before the first publish, confirm the name is still free at <https://crates.io/crates/uncertain-numerics>. Then tag the published commit and push the tag as usual: the workflow notices the version already exists, skips publishing, and creates the GitHub release.

### Trusted publishing

After the first publication, on the crate settings page on crates.io add a trusted publisher with:

| Field | Value |
| --- | --- |
| Repository owner | `DiogoRibeiro7` |
| Repository name | `uncertain-numerics` |
| Workflow filename | `release.yml` |
| Environment | `release` |

Then in the GitHub repository settings create an environment named `release`. Restricting it to protected tags is recommended so only maintainers can trigger a publish.

If trusted publishing is not wanted, replace the authentication step in the workflow with a `CARGO_REGISTRY_TOKEN` repository secret.

## Versioning policy

- Pre-1.0: minor versions may contain breaking API changes and are called out in the changelog; patch versions never do.
- Raising the minimum supported Rust version is at least a minor bump.
- Changes to numerical behavior that alter results within documented tolerances are patch-level; changes that alter contracts (for example what an error variant means) are breaking.
