# DINOv3 Resolution-7 Recovery Design

## Scope and safety

Recover only Amsterdam, Gombe, Kampala, Kozhikode, Malegaon, Sitapur, and
Vijayawada. Existing embedding shards are immutable inputs: Amsterdam resumes by
image name into its existing city directory, while all H3, pairwise, B5c, audit,
and final outputs are written below a timestamped recovery root until validation
passes. Missing comparisons are never imputed and zero is never used as a
sentinel.

## Architecture

`dinov3_res7_recovery.py` is the testable control-plane CLI. Its audit command
walks every boundary—city metadata, possible image-index aliases, indexed image
paths, embedding shards, current pano metadata, recomputed resolution-7 H3,
current Stage-3 membership, H3 summaries, pairwise shards, and B5c outputs—and
writes JSON plus CSV evidence. Alias selection is conservative: exact and
metadata-derived aliases may be selected automatically only when unique;
ambiguous candidates are reported and stop recovery for that city.

The same CLI builds an affected-only pair manifest and validates the final
parquet contract. The manifest includes a pair only when at least one endpoint is
a recovered city and both endpoints have usable resolution-7 summaries. The
validator streams parquet metadata/data, enforces the five-column schema,
direct H3-to-city membership, canonical uniqueness, non-null finite similarity,
and reports city/H3/row/pair coverage without inventing absent pairs.

`pipeline/run_dinov3_res7_recovery.bash` is the data-plane orchestrator. It runs
a preflight audit, submits embedding jobs only for cities with uniquely resolved
source imagery, waits using `squeue` plus `sacct`, rejects failed/cancelled jobs,
re-audits, rebuilds affected H3 summaries, creates the affected manifest,
submits bounded pairwise arrays into the recovery root, waits, runs B5c there,
and validates before writing a READY marker. Job IDs and exact commands are
appended to a machine-readable run log.

## Data flow and failure handling

The source index remains authoritative for embedding completeness, while current
GSV pano/path metadata is authoritative for image-to-H3 assignment. Stage-3
membership is an independent comparison target, never a source of fabricated
vectors. For Sitapur the audit distinguishes: embedded panoids absent from
current metadata; coordinates mapping outside the current grid; current core
cells with source panoids but no embedding; and core cells with no source
imagery.

All phases are idempotent. Nonempty pairwise shards are skipped only after their
schema and resolution are checked. Slurm terminal states other than COMPLETED
stop the run with job/task evidence. A city with no unique source index or no
existing indexed image is marked `source_imagery_absent` only after checked
locations and counts are serialized; otherwise it remains `unresolved`, not
silently absent.

## Testing

Fixture tests cover Unicode/suffix/known aliases, ambiguous aliases, affected
pair generation, resume-safe embedding commands, Sitapur-style stale membership,
orientation repair, exact schema, null/zero/duplicate rejection, and explicit
missing-city coverage. Shell tests exercise dry-run command generation and Slurm
state parsing without submitting jobs.
