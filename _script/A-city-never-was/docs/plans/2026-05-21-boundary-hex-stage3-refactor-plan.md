# Boundary Hex Stage3 Refactor Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the misplaced `similarity_hex_ids` prerequisite with boundary-derived city H3 hex lists so stage3 can retain zero-POI hexes and remain architecturally upstream of B7 landuse similarity analysis.

**Architecture:** Add a standalone boundary-to-H3 generator in the external landuse scripts that writes per-city parquet files under `city_boundary_hex_ids/res=<resolution>/`. Update stage3 to read those boundary hex lists, left-join POI aggregates onto the full hex set, and expose resolution as a CLI parameter defaulting to `8`.

**Tech Stack:** Python, geopandas, h3, pandas, unittest
