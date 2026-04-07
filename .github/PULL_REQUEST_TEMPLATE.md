## Summary

<!-- What does this PR do? 1-3 bullet points. -->

## Test plan

<!-- How was this tested? -->

- [ ] `cargo test --workspace`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo fmt --all -- --check`

## Checklist

- [ ] Tests added/updated for new behavior
- [ ] Documentation updated (if user-facing)
- [ ] No new `unwrap()` on mutex/rwlock acquisitions
