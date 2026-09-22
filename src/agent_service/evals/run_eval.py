"""Run the labelled dataset through the graph and report accuracy, outcomes and latency.

    agent-eval --backend rules
    agent-eval --backend openai --out evals/results

The same dataset runs against any backend, so a model change can be judged with numbers instead of
opinions. Results are written as JSON so runs can be compared in CI or a notebook.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from agent_service.backends import get_backend
from agent_service.config import Settings
from agent_service.graph import build_graph, initial_state
from agent_service.knowledge import KnowledgeBase


def load_dataset(path: str | None) -> list[dict]:
    if path:
        raw = Path(path).read_text("utf-8")
    else:
        raw = resources.files("agent_service.evals").joinpath("dataset.jsonl").read_text("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def run(settings: Settings, dataset: list[dict]) -> dict:
    kb = KnowledgeBase.load_default()
    backend = get_backend(settings)
    graph = build_graph(backend, kb, clarify_threshold=settings.clarify_threshold,
                        max_snippets=settings.max_snippets)
    rows: list[dict] = []
    latencies: list[float] = []
    for item in dataset:
        started = time.perf_counter()
        result = graph.invoke(initial_state(item["text"]))
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        cls = result.get("classification")
        rows.append({
            "id": item["id"],
            "expected_category": item["category"],
            "predicted_category": cls.category if cls else None,
            "expected_urgency": item["urgency"],
            "predicted_urgency": cls.urgency if cls else None,
            "confidence": cls.confidence if cls else None,
            "outcome": result["outcome"],
            "citations": result.get("citations", []),
            "guard_issues": result["guard"].issues if result.get("guard") else [],
            "latency_ms": round(latency_ms, 2),
        })
    scored = [r for r in rows if r["outcome"] != "clarify"]
    cat_acc = sum(r["predicted_category"] == r["expected_category"] for r in scored) / max(len(scored), 1)
    urg_acc = sum(r["predicted_urgency"] == r["expected_urgency"] for r in scored) / max(len(scored), 1)
    cited = sum(1 for r in scored if r["outcome"] == "answered" and r["citations"])
    answered = sum(1 for r in rows if r["outcome"] == "answered")
    summary = {
        "backend": backend.name,
        "model": settings.openai_model if backend.name == "openai" else None,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": len(rows),
        "category_accuracy": round(cat_acc, 3),
        "urgency_accuracy": round(urg_acc, 3),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "answered_with_citations": f"{cited}/{answered}",
        "guard_rejections": sum(1 for r in rows if r["guard_issues"]),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "mean": round(statistics.fmean(latencies), 1),
        },
        "misclassified": [r["id"] for r in scored if r["predicted_category"] != r["expected_category"]],
    }
    return {"summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["rules", "openai"], default="rules")
    parser.add_argument("--dataset", default=None, help="Path to a JSONL file (default: bundled)")
    parser.add_argument("--out", default="evals/results", help="Directory for the JSON report")
    args = parser.parse_args(argv)

    settings = Settings(backend=args.backend)
    report = run(settings, load_dataset(args.dataset))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{args.backend}-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), "utf-8")
    s = report["summary"]
    print(f"backend={s['backend']} n={s['n']} category_acc={s['category_accuracy']:.3f} "
          f"urgency_acc={s['urgency_accuracy']:.3f} outcomes={s['outcomes']} "
          f"citations={s['answered_with_citations']} guard_rejections={s['guard_rejections']} "
          f"latency_ms p50={s['latency_ms']['p50']} p95={s['latency_ms']['p95']}")
    if s["misclassified"]:
        print("misclassified:", ", ".join(s["misclassified"]))
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
