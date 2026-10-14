"""Chapter 7.8 (optional) — score the instruction-tuned model's answers with a
larger local LLM as judge, exactly as the book does with Ollama.

Prerequisites: install Ollama (https://ollama.com), run `ollama pull llama3.1`,
and keep `ollama serve` running. Then:

    python scripts/ch07_ollama_judge.py --judge llama3.1

Reads results/ch07_responses.json, writes results/ch07_judge.json.
LLM-as-judge scores are noisy and biased toward fluent text; treat them as a
rough quality signal next to the objective sentiment accuracy, not as proof.
"""

import argparse
import statistics

import requests

from _common import rel
from finllm.paths import RESULTS_DIR
from finllm.utils import load_json, run_metadata, save_json


def ask(prompt: str, model: str, url: str) -> str:
    r = requests.post(url, json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                 "stream": False, "options": {"seed": 123, "temperature": 0}}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge", default="llama3.1")
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    args = p.parse_args()
    data = load_json(RESULTS_DIR / "ch07_responses.json")
    scored = []
    for e in data["responses"]:
        q = e["instruction"] + (f"\n\n{e['input']}" if e["input"] else "")
        prompt = (f"Given the input `{q}` and correct output `{e['reference']}`, score the model response "
                  f"`{e['model_response']}` on a scale from 0 to 100, where 100 is the best score. "
                  "Respond with the integer number only.")
        raw = ask(prompt, args.judge, args.url)
        try:
            score = int("".join(ch for ch in raw if ch.isdigit())[:3])
        except ValueError:
            score = None
        scored.append({**e, "judge_score": score})
        print(score, "|", e["instruction"][:70])
    vals = [s["judge_score"] for s in scored if s["judge_score"] is not None]
    out = {"judge": args.judge, "n": len(scored), "n_scored": len(vals),
           "mean_score": statistics.mean(vals) if vals else None,
           "median_score": statistics.median(vals) if vals else None,
           "items": scored, "meta": run_metadata()}
    print(f"mean judge score {out['mean_score']}")
    print("saved", rel(save_json(out, RESULTS_DIR / "ch07_judge.json")))


if __name__ == "__main__":
    main()
