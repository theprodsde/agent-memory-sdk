# Competitive Benchmark RFC

## Status

Draft. This RFC proposes a reproducible comparison suite for local and hosted
agent-memory systems. It does not claim that Agent Memory is better than any
other project.

## Goals

- Compare retrieval quality, end-to-end answer quality, decision safety, and
  local resource use as separate questions.
- Make every reported result reproducible from a dataset hash, container image,
  configuration file, and raw result artifact.
- Invite maintainers of compared projects to review their adapter and recommend
  canonical self-hosted settings before results are presented as comparisons.

## Non-goals

- Produce a single "best memory tool" ranking.
- Compare local process latency with a hosted API's network-inclusive latency.
- Treat a retrieval metric as an end-to-end QA metric.
- Compare different dataset releases, retrieval units, reader models, or top-k
  budgets as if they were head-to-head results.

## Tracks

| Track | Primary metrics | Required controls |
|---|---|---|
| Retrieval | Recall@5/10, evidence coverage@5/10, MRR | Same dataset release, index unit, top-k, and corpus per system |
| End-to-end QA | Official benchmark accuracy, answer latency, total cost | Same reader model, prompt, temperature, context budget, and judge |
| Decision safety | Wrong replay, stale reuse, false abstention, verification rate | Public repeated-query, contradiction, TTL, and update cases |
| Local resources | Ingest time, query p50/p90/p95/p99, CPU, RSS | Same machine, OS, corpus, persistence mode, and warm/cold declaration |

## Dataset plan

| Dataset | Track | Rules |
|---|---|---|
| LongMemEval original release | Retrieval and end-to-end QA | Use one release and one retrieval unit for every system. Publish its SHA256. |
| LongMemEval cleaned release | Retrieval and end-to-end QA | Report separately from the original release; never combine baseline rankings. |
| LoCoMo | End-to-end QA | Use the official evaluator and identical reader-model budget. |
| Agent Memory Decision Suite | Decision safety | Versioned public cases for repeats, paraphrases, TTL expiry, contradictory updates, and abstention. |
| Persistent-store stress suite | Local resources | Fixed corpus sizes, no LRU cache unless the cache track is explicitly selected. |

Synthetic scaling data is a supplement, not a headline-quality claim.

## Deployment rules

- A local SDK is compared with each project's self-hosted or OSS deployment.
- Hosted results, if included, are reported in a separate table with network
  latency and API cost.
- Archived, retired, or unsupported implementations are ineligible. Adapters
  must target each project's current maintainers-supported codebase; for Letta,
  that means `letta-ai/letta-code`, not the archived `letta-ai/letta` server.
- Each adapter declares model providers, embedding models, external services,
  cache state, and whether LLM calls are used during ingestion or retrieval.
- Maintainers may submit an adapter or a configuration correction. The suite
  records the reviewer and configuration revision in the result artifact.

## Reproducibility contract

Each result must conform to
[`benchmarks/competitive/result.schema.json`](../benchmarks/competitive/result.schema.json)
and include:

- System version or commit SHA and adapter commit SHA.
- Dataset URL, release name, and SHA256.
- Hardware, OS, Python version, container image, and command line.
- Retrieval unit, top-k, prompt, reader model, judge, seeds, and cache state.
- Aggregate metrics and raw per-case outputs.
- Failures, timeouts, unsupported capabilities, and cost inputs.

Run each configuration at least five times for latency/resource measurements.
Report a median and spread, not only the best run.

## Claim rubric

| Evidence | Allowed wording |
|---|---|
| Agent Memory-only measurement | "Measured on this hardware/workload." |
| Matched retrieval run | "Higher retrieval recall than these named configurations." |
| Matched end-to-end run | "Higher benchmark QA accuracy than these named configurations." |
| Matched decision suite | "Lower wrong-replay rate than these named configurations." |
| Unmatched release, index unit, deployment, or reader model | "Not directly comparable." |

No result supports "best memory tool" without a published aggregate metric and
predeclared weighting. Capability differences should remain visible instead.

## Maintainer review request

Before publishing any comparative result, open an issue with the compared
project containing the exact adapter, dependency versions, deployment command,
and preliminary result. Ask maintainers to confirm:

1. The OSS/self-hosted configuration is representative.
2. The selected retrieval unit and embedding model are appropriate.
3. The result schema captures meaningful latency, resource, and cost fields.
4. Any project-specific failure modes or capabilities should be added.
