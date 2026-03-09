# Changelog

## [0.10.2](https://github.com/cogniplex/codemem/compare/v0.10.1...v0.10.2) (2026-03-09)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-embeddings bumped from 0.9.1 to 0.10.0

## [0.10.1](https://github.com/cogniplex/codemem/compare/v0.10.0...v0.10.1) (2026-03-09)


### Refactoring

* fix bugs, unify flows, add safe accessors, define constants ([#16](https://github.com/cogniplex/codemem/issues/16)) ([1554c3f](https://github.com/cogniplex/codemem/commit/1554c3f64f0c2e456cbe5f17548a813f62a9a4f4))
* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))
* tier 2 quick wins — constructors, dedup helpers, storage ergonomics ([#14](https://github.com/cogniplex/codemem/issues/14)) ([e7243b7](https://github.com/cogniplex/codemem/commit/e7243b7f79af93662fcf644b31ea091dd863d0b4))
* tier 3 — domain logic to engine, drop binary storage/embeddings deps ([#15](https://github.com/cogniplex/codemem/issues/15)) ([a92b846](https://github.com/cogniplex/codemem/commit/a92b8463e2660b0318c01fa03f18f9ac864ddc39))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.0 to 0.10.1
    * codemem-storage bumped from 0.10.0 to 0.10.1
    * codemem-embeddings bumped from 0.9.0 to 0.9.1

## [0.10.0](https://github.com/cogniplex/codemem/compare/v0.9.0...v0.10.0) (2026-03-08)


### Features

* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))


### Bug Fixes

* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))


### Documentation

* update stale docs — remove volatile numbers, fix counts, rewrite CONTRIBUTING ([#11](https://github.com/cogniplex/codemem/issues/11)) ([c481a12](https://github.com/cogniplex/codemem/commit/c481a12d833d02fe12ac86e6de2d09bba7e99158))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.9.0 to 0.10.0
    * codemem-storage bumped from 0.9.0 to 0.10.0
    * codemem-embeddings bumped from 0.8.1 to 0.9.0

## [0.9.0](https://github.com/cogniplex/codemem/compare/v0.8.0...v0.9.0) (2026-03-08)


### Features

* **engine:** respect configured vector dimensions ([9a4d18d](https://github.com/cogniplex/codemem/commit/9a4d18decd256d3549d86cd289bda7c9453043b8))
* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))
* **recall:** entity expansion surfaces structurally connected memories ([fce2acb](https://github.com/cogniplex/codemem/commit/fce2acb7850c2fd4ab7de44d44ac1f5009cbf6ff))
* semantic-aware chunking with boundary splitting and signature injection ([f261fef](https://github.com/cogniplex/codemem/commit/f261fefc04d1022773a95c98b48ffa22c2590fee))


### Bug Fixes

* auto-create ~/.codemem directory on engine startup ([71154fb](https://github.com/cogniplex/codemem/commit/71154fbe79c8aa308015ea9048898a52480d1b6d))
* blame enrichment now stores ownership insights for all qualifying files ([93a6571](https://github.com/cogniplex/codemem/commit/93a6571008d116893eef4ce1f9627584ef8284f0))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))
* use relative paths for graph node IDs ([2c9e6a2](https://github.com/cogniplex/codemem/commit/2c9e6a270a79670840547e229f6391ccdd83c4d8))


### Performance

* batch graph node/edge inserts and embedding storage in persistence pipeline ([3b518dd](https://github.com/cogniplex/codemem/commit/3b518dda2230af1c668b75d4cc9af083cc057d9c))


### Refactoring

* consolidate enrichment dispatch into run_enrichments() ([77b5921](https://github.com/cogniplex/codemem/commit/77b592168f3dacd54fec38f33a3c37437bbed69e))
* encapsulate CodememEngine fields behind accessor methods ([5d64df7](https://github.com/cogniplex/codemem/commit/5d64df7335bf76aa4d919b673159f9cf6aae52f7))
* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* split enrichment.rs into module directory (15 files) ([c39f6e2](https://github.com/cogniplex/codemem/commit/c39f6e2a0be2779011900a16d1102b61b0a78388))
* split large test files into focused modules ([faf8ff4](https://github.com/cogniplex/codemem/commit/faf8ff43ee23a93e4db9359961f0fd6593b69cbc))
* split monolithic engine files into focused modules ([be79b09](https://github.com/cogniplex/codemem/commit/be79b097cf36fa443f7c6dd1961417187422e83a))


### Tests

* add 201 new tests across engine, API, CLI, and MCP layers ([b5e26a8](https://github.com/cogniplex/codemem/commit/b5e26a82f12d0135c10141f22e7aeb9b24882d78))
* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
* add coverage for relative path normalization ([b6bff33](https://github.com/cogniplex/codemem/commit/b6bff3394ff6e68ef7117f69c765b147a62d4641))


### Miscellaneous

* add debug-level timing for embed/sqlite/vector persistence phases ([f46f82a](https://github.com/cogniplex/codemem/commit/f46f82a8bfc6766f7c6702577d92d0a49d6e91ea))
