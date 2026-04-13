# Changelog

## [0.16.0](https://github.com/cogniplex/codemem/compare/v0.15.0...v0.16.0) (2026-04-13)


### Features

* JinaBERT support, F16 default, configurable embedding model ([#68](https://github.com/cogniplex/codemem/issues/68)) ([48f423d](https://github.com/cogniplex/codemem/commit/48f423d4f85ee0174276491ea1ace35bac37d214))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.17.0 to 0.18.0

## [0.15.0](https://github.com/cogniplex/codemem/compare/v0.14.0...v0.15.0) (2026-04-07)


### Features

* JinaBERT support, F16 default, configurable embedding model ([#68](https://github.com/cogniplex/codemem/issues/68)) ([48f423d](https://github.com/cogniplex/codemem/commit/48f423d4f85ee0174276491ea1ace35bac37d214))


### Bug Fixes

* post-review fixes for temporal graph, Gemini provider, and migration safety ([#56](https://github.com/cogniplex/codemem/issues/56)) ([f345dc6](https://github.com/cogniplex/codemem/commit/f345dc65550d2281d204856b1757a901f0a1f338))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.16.0 to 0.17.0

## [0.14.0](https://github.com/cogniplex/codemem/compare/v0.13.1...v0.14.0) (2026-03-27)


### Features

* add Google Gemini embedding provider ([#50](https://github.com/cogniplex/codemem/issues/50)) ([007f0d4](https://github.com/cogniplex/codemem/commit/007f0d45bbc6fc84e629ae0c668840e26206d515))
* configurable embedding model, dtype, and batch size ([#31](https://github.com/cogniplex/codemem/issues/31)) ([6dfbfce](https://github.com/cogniplex/codemem/commit/6dfbfce0377b5e46c0a2de8907ac11a53f19e490))
* v0.4.0 production hardening — zero unwraps, safe concurrency, config persistence, schema migrations ([7a81665](https://github.com/cogniplex/codemem/commit/7a816651580ff7b891a40a9bc41322373fcddb15))
* v0.6.0 — 8 new language parsers, operational metrics, CLI commands, test coverage ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Bug Fixes

* **ci:** resolve all formatting, clippy, and eslint failures ([80cd400](https://github.com/cogniplex/codemem/commit/80cd400d02904c15de8937604b53abe733b05386))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))
* embedding memory leaks, silent batch drops, HNSW ghost compaction, BM25 persistence ([7a41a36](https://github.com/cogniplex/codemem/commit/7a41a367d7031ac423743f6a36bb317fca0fb054))
* post-review fixes for temporal graph, Gemini provider, and migration safety ([#56](https://github.com/cogniplex/codemem/issues/56)) ([f345dc6](https://github.com/cogniplex/codemem/commit/f345dc65550d2281d204856b1757a901f0a1f338))
* restore batch size 32 and remove redundant device.synchronize() ([b0469d3](https://github.com/cogniplex/codemem/commit/b0469d351e40a248dd04d59d2e3da4545ddbc6ba))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.15.0 to 0.16.0

## [0.13.1](https://github.com/cogniplex/codemem/compare/v0.13.0...v0.13.1) (2026-03-18)


### Bug Fixes

* post-review fixes for temporal graph, Gemini provider, and migration safety ([#56](https://github.com/cogniplex/codemem/issues/56)) ([f345dc6](https://github.com/cogniplex/codemem/commit/f345dc65550d2281d204856b1757a901f0a1f338))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.14.0 to 0.15.0

## [0.13.0](https://github.com/cogniplex/codemem/compare/v0.12.0...v0.13.0) (2026-03-18)


### Features

* add Google Gemini embedding provider ([#50](https://github.com/cogniplex/codemem/issues/50)) ([007f0d4](https://github.com/cogniplex/codemem/commit/007f0d45bbc6fc84e629ae0c668840e26206d515))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.13.0 to 0.14.0

## [0.12.0](https://github.com/cogniplex/codemem/compare/v0.11.1...v0.12.0) (2026-03-16)


### Features

* add Google Gemini embedding provider ([#50](https://github.com/cogniplex/codemem/issues/50)) ([007f0d4](https://github.com/cogniplex/codemem/commit/007f0d45bbc6fc84e629ae0c668840e26206d515))
* configurable embedding model, dtype, and batch size ([#31](https://github.com/cogniplex/codemem/issues/31)) ([6dfbfce](https://github.com/cogniplex/codemem/commit/6dfbfce0377b5e46c0a2de8907ac11a53f19e490))
* v0.4.0 production hardening — zero unwraps, safe concurrency, config persistence, schema migrations ([7a81665](https://github.com/cogniplex/codemem/commit/7a816651580ff7b891a40a9bc41322373fcddb15))
* v0.6.0 — 8 new language parsers, operational metrics, CLI commands, test coverage ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Bug Fixes

* **ci:** resolve all formatting, clippy, and eslint failures ([80cd400](https://github.com/cogniplex/codemem/commit/80cd400d02904c15de8937604b53abe733b05386))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))
* embedding memory leaks, silent batch drops, HNSW ghost compaction, BM25 persistence ([7a41a36](https://github.com/cogniplex/codemem/commit/7a41a367d7031ac423743f6a36bb317fca0fb054))
* restore batch size 32 and remove redundant device.synchronize() ([b0469d3](https://github.com/cogniplex/codemem/commit/b0469d351e40a248dd04d59d2e3da4545ddbc6ba))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.12.0 to 0.13.0

## [0.11.1](https://github.com/cogniplex/codemem/compare/v0.11.0...v0.11.1) (2026-03-13)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.11.0 to 0.12.0

## [0.11.0](https://github.com/cogniplex/codemem/compare/v0.10.1...v0.11.0) (2026-03-11)


### Features

* configurable embedding model, dtype, and batch size ([#31](https://github.com/cogniplex/codemem/issues/31)) ([6dfbfce](https://github.com/cogniplex/codemem/commit/6dfbfce0377b5e46c0a2de8907ac11a53f19e490))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.1 to 0.11.0

## [0.10.1](https://github.com/cogniplex/codemem/compare/v0.10.0...v0.10.1) (2026-03-09)


### Refactoring

* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))


### Miscellaneous

* release main ([#12](https://github.com/cogniplex/codemem/issues/12)) ([95e6fc4](https://github.com/cogniplex/codemem/commit/95e6fc4b100950257c6955bea6db3d2138b4ab52))
* release main ([#18](https://github.com/cogniplex/codemem/issues/18)) ([4fba8b2](https://github.com/cogniplex/codemem/commit/4fba8b206ee4b1a6ca89429e4cbb9dc4a6ad33cb))

## [0.10.0](https://github.com/cogniplex/codemem/compare/v0.9.1...v0.10.0) (2026-03-09)


### Features

* v0.4.0 production hardening — zero unwraps, safe concurrency, config persistence, schema migrations ([7a81665](https://github.com/cogniplex/codemem/commit/7a816651580ff7b891a40a9bc41322373fcddb15))
* v0.6.0 — 8 new language parsers, operational metrics, CLI commands, test coverage ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Bug Fixes

* **ci:** resolve all formatting, clippy, and eslint failures ([80cd400](https://github.com/cogniplex/codemem/commit/80cd400d02904c15de8937604b53abe733b05386))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))
* embedding memory leaks, silent batch drops, HNSW ghost compaction, BM25 persistence ([7a41a36](https://github.com/cogniplex/codemem/commit/7a41a367d7031ac423743f6a36bb317fca0fb054))
* restore batch size 32 and remove redundant device.synchronize() ([b0469d3](https://github.com/cogniplex/codemem/commit/b0469d351e40a248dd04d59d2e3da4545ddbc6ba))


### Refactoring

* create codemem-engine, merge MCP+API+CLI into unified codemem crate ([ab0089b](https://github.com/cogniplex/codemem/commit/ab0089ba6b17d9f3027ac8dfc0923c264b5f71b9))
* remove dead cache_stats() from EmbeddingService ([8612df0](https://github.com/cogniplex/codemem/commit/8612df0fdb14b38a99dbcf4b582722922c15d2dc))
* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* remove double LRU cache from EmbeddingService, make batch_size configurable ([6a9aec3](https://github.com/cogniplex/codemem/commit/6a9aec3c85698b0415528e268ed4925b15f36340))
* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))


### Tests

* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
* add embedding provider and CLI lifecycle test coverage ([1793bdf](https://github.com/cogniplex/codemem/commit/1793bdf218e21320d763881bab9e1c2eedac33f2))
* **embeddings,graph,vector,watch:** add comprehensive unit tests ([0c26645](https://github.com/cogniplex/codemem/commit/0c26645ff811feeb82afee9183adc203dd58848a))


### Documentation

* rewrite README, add crate READMEs, installer, brew tap workflow, and code-mapper agent ([a4e7676](https://github.com/cogniplex/codemem/commit/a4e76763460160e94f0e5150c6a33c0d6cf0aca7))
* update architecture, CLI reference, MCP tools, and comparison docs. ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Miscellaneous

* release main ([#12](https://github.com/cogniplex/codemem/issues/12)) ([95e6fc4](https://github.com/cogniplex/codemem/commit/95e6fc4b100950257c6955bea6db3d2138b4ab52))
* release main ([#7](https://github.com/cogniplex/codemem/issues/7)) ([ec89b7a](https://github.com/cogniplex/codemem/commit/ec89b7a6f30fc07d1c41b236a9c4e8eb6cacfc49))
* release main ([#8](https://github.com/cogniplex/codemem/issues/8)) ([c4aad88](https://github.com/cogniplex/codemem/commit/c4aad88782e6c8a53353cd919f7f249331d0d8ed))

## [0.9.1](https://github.com/cogniplex/codemem/compare/v0.9.0...v0.9.1) (2026-03-09)


### Bug Fixes

* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))


### Refactoring

* tier 1 quick wins — dead code, wiring, visibility, dedup ([#13](https://github.com/cogniplex/codemem/issues/13)) ([56a469a](https://github.com/cogniplex/codemem/commit/56a469a25ad57a00cfeee6714b76e8c582f33ace))


### Miscellaneous

* release main ([#8](https://github.com/cogniplex/codemem/issues/8)) ([c4aad88](https://github.com/cogniplex/codemem/commit/c4aad88782e6c8a53353cd919f7f249331d0d8ed))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.10.0 to 0.10.1

## [0.9.0](https://github.com/cogniplex/codemem/compare/v0.8.1...v0.9.0) (2026-03-08)


### Features

* v0.4.0 production hardening — zero unwraps, safe concurrency, config persistence, schema migrations ([7a81665](https://github.com/cogniplex/codemem/commit/7a816651580ff7b891a40a9bc41322373fcddb15))
* v0.6.0 — 8 new language parsers, operational metrics, CLI commands, test coverage ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Bug Fixes

* **ci:** resolve all formatting, clippy, and eslint failures ([80cd400](https://github.com/cogniplex/codemem/commit/80cd400d02904c15de8937604b53abe733b05386))
* **ci:** use explicit crate versions for release-please compatibility ([cc54698](https://github.com/cogniplex/codemem/commit/cc54698870e3a2d69904859ff032fbd1ccc224a2))
* **ci:** use explicit path+version for internal deps instead of workspace inheritance ([cc3f43c](https://github.com/cogniplex/codemem/commit/cc3f43c82b7eb8593e69b5f78baf5cf6fe5201bb))
* embedding memory leaks, silent batch drops, HNSW ghost compaction, BM25 persistence ([7a41a36](https://github.com/cogniplex/codemem/commit/7a41a367d7031ac423743f6a36bb317fca0fb054))
* restore batch size 32 and remove redundant device.synchronize() ([b0469d3](https://github.com/cogniplex/codemem/commit/b0469d351e40a248dd04d59d2e3da4545ddbc6ba))


### Refactoring

* create codemem-engine, merge MCP+API+CLI into unified codemem crate ([ab0089b](https://github.com/cogniplex/codemem/commit/ab0089ba6b17d9f3027ac8dfc0923c264b5f71b9))
* remove dead cache_stats() from EmbeddingService ([8612df0](https://github.com/cogniplex/codemem/commit/8612df0fdb14b38a99dbcf4b582722922c15d2dc))
* remove dead code, consolidate utilities, wire config to backends ([f22efcc](https://github.com/cogniplex/codemem/commit/f22efcccd63005e08ce5b82e35f14b0a6cc7a984))
* remove double LRU cache from EmbeddingService, make batch_size configurable ([6a9aec3](https://github.com/cogniplex/codemem/commit/6a9aec3c85698b0415528e268ed4925b15f36340))


### Tests

* add comprehensive test coverage across all crates (~300 tests) ([e758f05](https://github.com/cogniplex/codemem/commit/e758f0585d31ada1b599b5db950e711c02552116))
* add embedding provider and CLI lifecycle test coverage ([1793bdf](https://github.com/cogniplex/codemem/commit/1793bdf218e21320d763881bab9e1c2eedac33f2))
* **embeddings,graph,vector,watch:** add comprehensive unit tests ([0c26645](https://github.com/cogniplex/codemem/commit/0c26645ff811feeb82afee9183adc203dd58848a))


### Documentation

* rewrite README, add crate READMEs, installer, brew tap workflow, and code-mapper agent ([a4e7676](https://github.com/cogniplex/codemem/commit/a4e76763460160e94f0e5150c6a33c0d6cf0aca7))
* update architecture, CLI reference, MCP tools, and comparison docs. ([917d940](https://github.com/cogniplex/codemem/commit/917d940da2770b052e6f65b618df5bbd95ca75db))


### Miscellaneous

* release main ([#7](https://github.com/cogniplex/codemem/issues/7)) ([ec89b7a](https://github.com/cogniplex/codemem/commit/ec89b7a6f30fc07d1c41b236a9c4e8eb6cacfc49))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * codemem-core bumped from 0.9.0 to 0.10.0

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
