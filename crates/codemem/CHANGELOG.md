# Changelog

## [0.10.2](https://github.com/cogniplex/codemem/compare/v0.10.1...v0.10.2) (2026-03-09)


### Bug Fixes

* cargo install codemem broken on crates.io ([#17](https://github.com/cogniplex/codemem/issues/17)) ([13e5e78](https://github.com/cogniplex/codemem/commit/13e5e781ead4298ec0d2078071ac3f03f487a48e))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-engine bumped from 0.10.1 to 0.10.2

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
    * codemem-engine bumped from 0.10.0 to 0.10.1

## [0.10.0](https://github.com/cogniplex/codemem/compare/v0.9.0...v0.10.0) (2026-03-08)


### Features

* session continuity and persistence pipeline improvements ([#10](https://github.com/cogniplex/codemem/issues/10)) ([970b6f8](https://github.com/cogniplex/codemem/commit/970b6f899dccee81e84143ad6a8ac96f8965307c))


### Bug Fixes

* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.9.0 to 0.10.0
    * codemem-storage bumped from 0.9.0 to 0.10.0
    * codemem-embeddings bumped from 0.8.1 to 0.9.0
    * codemem-engine bumped from 0.9.0 to 0.10.0

## [0.9.0](https://github.com/cogniplex/codemem/compare/v0.8.0...v0.9.0) (2026-03-08)


### Features

* **engine:** respect configured vector dimensions ([9a4d18d](https://github.com/cogniplex/codemem/commit/9a4d18decd256d3549d86cd289bda7c9453043b8))
* **graph:** tag-based auto-linking and memory-neighbor scoring ([1403797](https://github.com/cogniplex/codemem/commit/1403797ebca46847e617e8d9a2faa9da31565e5a))
* track per-session metadata in MCP HTTP transport ([95c5f06](https://github.com/cogniplex/codemem/commit/95c5f062ff1c1d90b2cc18494533928d48e95339))
* **ui:** replace Sigma.js with react-force-graph-2d, overhaul PointCloud ([56ba13a](https://github.com/cogniplex/codemem/commit/56ba13afb044a9ef4a1cd27d4261e00695d3700a))
* wire up all 14 enrichment types in enrich_codebase tool ([c389e3f](https://github.com/cogniplex/codemem/commit/c389e3f11d16b4e1bf5054be6768ed4157a1f4f4))


### Bug Fixes

* **api:** route memory CRUD through engine pipeline ([e21a4a7](https://github.com/cogniplex/codemem/commit/e21a4a7ea94071e6f8795f8da174bbd0aaadb8bc))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* create ~/.codemem directory in serve command before opening DB ([1353b75](https://github.com/cogniplex/codemem/commit/1353b75d00f235ca235888f0336660aefd6eb3a2))
* implement MCP SSE server-push for GET /mcp endpoint ([85a8c3f](https://github.com/cogniplex/codemem/commit/85a8c3f9bd024ec1504d0502fd6fa53ddd4c33f0))
* propagate config changes to running engine in API routes ([a1c2f86](https://github.com/cogniplex/codemem/commit/a1c2f8632699a3a95852422b5e35b61a05c1e4f4))
* use full persist pipeline in store_pattern_memory, remove dead code ([a262533](https://github.com/cogniplex/codemem/commit/a2625331cb876c869cd2b921621c8d38b4c77f81))


### Refactoring

* consolidate enrichment dispatch into run_enrichments() ([77b5921](https://github.com/cogniplex/codemem/commit/77b592168f3dacd54fec38f33a3c37437bbed69e))
* encapsulate CodememEngine fields behind accessor methods ([5d64df7](https://github.com/cogniplex/codemem/commit/5d64df7335bf76aa4d919b673159f9cf6aae52f7))
* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* remove legacy MCP tool aliases and update to canonical names ([f03809d](https://github.com/cogniplex/codemem/commit/f03809dc5b611469282ef6f72941d842d43e5e62))
* rewrite code-mapper as multi-agent team ([fa09a04](https://github.com/cogniplex/codemem/commit/fa09a04ba2cdb84719d6a8d64627a1c9e04fe9f6))
* split large test files into focused modules ([faf8ff4](https://github.com/cogniplex/codemem/commit/faf8ff43ee23a93e4db9359961f0fd6593b69cbc))
* split monolithic engine files into focused modules ([be79b09](https://github.com/cogniplex/codemem/commit/be79b097cf36fa443f7c6dd1961417187422e83a))


### Tests

* add 201 new tests across engine, API, CLI, and MCP layers ([b5e26a8](https://github.com/cogniplex/codemem/commit/b5e26a82f12d0135c10141f22e7aeb9b24882d78))
* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
* add embedding provider and CLI lifecycle test coverage ([1793bdf](https://github.com/cogniplex/codemem/commit/1793bdf218e21320d763881bab9e1c2eedac33f2))
* **cli:** expand CLI command test coverage with extracted testable functions ([938ee04](https://github.com/cogniplex/codemem/commit/938ee04c3590a119ec37677c490b3720e3b1e9d1))


### Miscellaneous

* fix .gitignore to exclude ui-dist build artifacts ([a78ac22](https://github.com/cogniplex/codemem/commit/a78ac22f09a0e99375eefcca46ac61f70ec97a3c))
