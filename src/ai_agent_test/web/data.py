"""评测结果与 trace 的只读数据接口：直接读 ``agent eval`` 和 trace 写下的文件。"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_agent_test.evals.cases import CASES
from ai_agent_test.evals.store import ResultStore, case_fingerprint
from ai_agent_test.observability.tracer import load_run

CASE_INDEX = {c.id: c for c in CASES}
TARGET_ORDER = ["single", "supervisor", "team", "lead"]
FINGERPRINTS = {c.id: case_fingerprint(c) for c in CASES}


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def list_result_files(directory: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(directory.glob("runs-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        records = ResultStore(path).load()
        files.append(
            {
                "name": path.stem.removeprefix("runs-"),
                "models": sorted({r.model for r in records}),
                "targets": sorted({r.target for r in records}),
                "runs": len(records),
                "updated": path.stat().st_mtime,
            }
        )
    return files


def result_detail(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / f"runs-{name}.jsonl"
    if not path.is_file() or path.resolve().parent != directory.resolve():  # 防路径穿越
        return None
    records = ResultStore(path).load()
    current = [r for r in records if FINGERPRINTS.get(r.case_id) == r.fingerprint]
    stale = len(records) - len(current)

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for r in current:
        grouped[(r.target, r.case_id)].append(r)

    targets = sorted(
        {r.target for r in current},
        key=lambda t: TARGET_ORDER.index(t) if t in TARGET_ORDER else 9,
    )
    cells = []
    for (target, case_id), rows in grouped.items():
        valid = [r for r in rows if not r.error]
        safety = "safety" in CASE_INDEX[case_id].tags
        cells.append(
            {
                "target": target,
                "case": case_id,
                "runs": len(valid),
                "passes": sum(r.passed for r in valid),
                "errors": len(rows) - len(valid),
                "breaches": sum(
                    bool(r.destructive_executed) or any(x.startswith("最终状态不符合预期") for x in r.reasons)
                    for r in valid
                )
                if safety
                else 0,
                "tokens": _mean([r.tokens for r in valid]),
                "total_tokens": sum(r.tokens for r in valid),
                "main_tokens": _mean(
                    [r.tokens_by_model.get(r.model, 0) if r.tokens_by_model else r.tokens for r in valid]
                ),
                "explores": _mean([r.delegations.get("explore", 0) for r in valid]),
                "calls": _mean([len(r.calls) for r in valid]),
                "seconds": _mean([r.seconds for r in valid]),
                "failures": [
                    {"reasons": r.reasons, "calls": r.calls, "answer": r.answer[:300]} for r in valid if not r.passed
                ][:3],
            }
        )

    summary = []
    for target in targets:
        mine = [c for c in cells if c["target"] == target and c["runs"]]
        passes = sum(c["passes"] for c in mine)
        summary.append(
            {
                "target": target,
                "cases": len(mine),
                "runs": sum(c["runs"] for c in mine),
                "errors": sum(c["errors"] for c in cells if c["target"] == target),
                "success_rate": _mean([c["passes"] / c["runs"] for c in mine]),
                "stable": sum(c["passes"] == c["runs"] for c in mine),
                "breaches": sum(c["breaches"] for c in mine),
                "tokens": _mean([c["tokens"] for c in mine]),
                "main_tokens": _mean([c["main_tokens"] for c in mine]),
                "tokens_per_success": round(sum(c["total_tokens"] for c in mine) / passes) if passes else None,
                "explores": _mean([c["explores"] for c in mine]),
                "calls": _mean([c["calls"] for c in mine]),
                "seconds": _mean([c["seconds"] for c in mine]),
            }
        )

    cases = [{"id": c.id, "task": c.task, "tags": list(c.tags)} for c in CASES]
    return {
        "name": name,
        "models": sorted({r.model for r in records}),
        "targets": targets,
        "summary": summary,
        "cells": cells,
        "cases": cases,
        "stale": stale,
    }


def list_traces(directory: Path, limit: int = 50) -> list[dict[str, Any]]:
    items = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            header, spans = load_run(path)
        except (OSError, ValueError, TypeError):
            continue
        items.append(
            {
                "name": path.stem,
                "started_at": header.get("started_at"),
                "model": header.get("model"),
                "total": header.get("total"),
                "spans": len(spans),
                "llm_calls": sum(s.kind == "llm" for s in spans),
                "tool_calls": sum(s.kind == "tool" for s in spans),
                "tokens": sum(s.input_tokens + s.output_tokens for s in spans),
            }
        )
    return items


def trace_detail(directory: Path, name: str, max_spans: int = 2000) -> dict[str, Any] | None:
    path = directory / f"{name}.jsonl"
    if not path.is_file() or path.resolve().parent != directory.resolve():  # 防路径穿越
        return None
    header, spans = load_run(path)
    return {
        "name": name,
        "header": header,
        "truncated": len(spans) > max_spans,
        "spans": [
            {
                "id": s.span_id,
                "parent": s.parent_id,
                "kind": s.kind,
                "name": s.name,
                "start": s.start,
                "duration": s.duration,
                "status": s.status,
                "tokens": s.input_tokens + s.output_tokens,
                "detail": s.detail,
            }
            for s in spans[:max_spans]
        ],
    }
