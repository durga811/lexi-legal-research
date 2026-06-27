"""Reproducible live smoke for the agent (requires Gemini + Pinecone).

Verifies the simple path, the deep precedent path, and — to prove the agent is
NOT a hard-coded Lakshmi pipeline — a different deep query (contributory negligence).

Run:  uv run python -m src.agent.agent_smoke
"""

from __future__ import annotations

from src.agent.graph import run_agent


def _deep_checks(label: str, s: dict, failures: list[str]) -> None:
    if s.get("query_complexity") != "deep":
        failures.append(f"{label}: did not route deep")
    if not (s.get("final_answer") or "").strip():
        failures.append(f"{label}: empty answer")
    if not s.get("selected_evidence", {}).get("adverse"):
        failures.append(f"{label}: no adverse precedents surfaced (eval dim 4)")
    vr = s.get("validation_report", {})
    if vr.get("uncited_or_invented_docs"):
        failures.append(f"{label}: invented/uncited docs {vr['uncited_or_invented_docs']}")
    docs = {c["doc_id"] for c in s.get("selected_evidence", {}).get("selected", [])}
    if len(docs) < 3:
        failures.append(f"{label}: evidence not diverse ({len(docs)} docs)")


def main() -> int:
    failures: list[str] = []

    print("[1/3] simple lookup ...")
    s1 = run_agent("Which judgments involve commercial vehicles?")
    if s1.get("query_complexity") != "simple":
        failures.append("simple: did not route simple")
    if not (s1.get("final_answer") or "").strip():
        failures.append("simple: empty answer")

    print("[2/3] deep — Lakshmi-style licence/insurer dispute ...")
    s2 = run_agent("Find supporting and adverse precedents and a strategy for a claimant widow whose "
                   "husband was killed by an unlicensed commercial truck driver, where the insurer "
                   "denies liability claiming the policy is void.")
    _deep_checks("deep-licence", s2, failures)

    print("[3/3] deep — different topic (contributory negligence) to prove non-hard-coding ...")
    s3 = run_agent("Find precedents that support our argument on contributory negligence in a "
                   "motorcycle accident, and the strongest adverse cases.")
    _deep_checks("deep-negligence", s3, failures)

    print("\nsummary:")
    for tag, s in (("simple", s1), ("deep-licence", s2), ("deep-negligence", s3)):
        sel = s.get("selected_evidence", {}) or {}
        print(f"  {tag:14} complexity={s.get('query_complexity'):7} "
              f"support={[c['doc_id'] for c in sel.get('supporting', [])]} "
              f"adverse={[c['doc_id'] for c in sel.get('adverse', [])]}")

    if failures:
        print("\nAGENT SMOKE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAGENT SMOKE PASSED — simple + deep paths, adverse coverage, grounded citations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
