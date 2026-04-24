#!/usr/bin/env python3
"""
Plot validation-loss curves for the gmm_patch vs baseline SQuAD runs.

Reads validation.jsonl from each run's results/ subdir, overlays curves, and
prints the max absolute delta at matching steps (the correctness sanity check
— a functional no-op patch should match the baseline within step-to-step noise).

Usage:
    # after running both run_squad_* scripts (writes results/squad_{gmm,baseline}/):
    python eval/plot_squad_val_curves.py

    # or with pre-shipped JSONLs (if you just want to regenerate the plot):
    python eval/plot_squad_val_curves.py \\
        --gmm-jsonl     results/squad_gmm_val.jsonl \\
        --baseline-jsonl results/squad_baseline_val.jsonl
"""
import argparse
import json
import pathlib
import sys


def load_val_jsonl(path: pathlib.Path) -> list[tuple[int, float]]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found - did the run finish?")
    pts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "val_loss" in rec and "step" in rec:
                pts.append((int(rec["step"]), float(rec["val_loss"])))
    pts.sort(key=lambda x: x[0])
    return pts


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parent.parent
    default_gmm_dir = repo / "results" / "squad_gmm"
    default_base_dir = repo / "results" / "squad_baseline"
    default_out = repo / "results" / "squad_val_loss_comparison.png"

    ap = argparse.ArgumentParser()
    ap.add_argument("--gmm-dir", default=str(default_gmm_dir))
    ap.add_argument("--baseline-dir", default=str(default_base_dir))
    ap.add_argument("--gmm-jsonl", default=None,
                    help="override path to patched validation.jsonl (else <gmm-dir>/validation.jsonl)")
    ap.add_argument("--baseline-jsonl", default=None,
                    help="override path to baseline validation.jsonl")
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()

    gmm_path = pathlib.Path(args.gmm_jsonl) if args.gmm_jsonl else pathlib.Path(args.gmm_dir) / "validation.jsonl"
    base_path = pathlib.Path(args.baseline_jsonl) if args.baseline_jsonl else pathlib.Path(args.baseline_dir) / "validation.jsonl"
    gmm = load_val_jsonl(gmm_path)
    base = load_val_jsonl(base_path)

    print(f"gmm_patch:  {len(gmm)} val points, steps {[s for s, _ in gmm]}")
    print(f"baseline:   {len(base)} val points, steps {[s for s, _ in base]}")

    shared = dict(gmm).keys() & dict(base).keys()
    if shared:
        deltas = [abs(dict(gmm)[s] - dict(base)[s]) for s in shared]
        max_delta = max(deltas)
        mean_gmm = sum(v for _, v in gmm) / len(gmm)
        print(f"max |delta val_loss| at shared steps: {max_delta:.6f}")
        print(f"relative (max delta / mean val_loss): {max_delta / mean_gmm:.4%}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([s for s, _ in base], [v for _, v in base], marker="o", label="baseline (stock HF for-loop)")
    ax.plot([s for s, _ in gmm], [v for _, v in gmm], marker="s", label="gmm_patch (grouped GEMM)")
    ax.set_xlabel("step")
    ax.set_ylabel("val_loss")
    ax.set_title("Nemotron-3-Nano-30B-A3B-BF16 / SQuAD / 100 steps / gmm_patch vs baseline")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
