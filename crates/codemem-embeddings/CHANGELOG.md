# Changelog

## [0.8.1](https://github.com/cogniplex/codemem/compare/v0.8.0...v0.8.1) (2026-03-08)


### Bug Fixes

* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))


### Refactoring

* remove dead cache_stats() from EmbeddingService ([8612df0](https://github.com/cogniplex/codemem/commit/8612df0fdb14b38a99dbcf4b582722922c15d2dc))
* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* remove double LRU cache from EmbeddingService, make batch_size configurable ([6a9aec3](https://github.com/cogniplex/codemem/commit/6a9aec3c85698b0415528e268ed4925b15f36340))


### Tests

* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
* add embedding provider and CLI lifecycle test coverage ([1793bdf](https://github.com/cogniplex/codemem/commit/1793bdf218e21320d763881bab9e1c2eedac33f2))
