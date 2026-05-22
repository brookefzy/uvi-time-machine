# B7 Default Optimized City Root Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `B7_similarity_by_landuse.py` default to the pre-aggregated `c_city_similarity_optimized_res=<res>` source when `--pairwise-root` is omitted.

**Architecture:** Change the default root constant to the curated parent directory, add a `resolve_pairwise_root(...)` helper that derives the resolution-specific optimized-city folder, and remove the implicit fallback from the old raw pairwise root. Explicit `--pairwise-root` values should still work for temp-shard inputs.

**Tech Stack:** Python, unittest
