# Results

**Note (July 2026):** earlier versions of this page reported single-run
results (including a noise-sweep table) that were superseded by the
multi-seed falsification campaign. They have been removed.

The current, verified results are:

- **Controller (Unified-LoRA vs AdaLoRA):** falsified — AdaLoRA is superior
  on quality and stability in every tested regime. See the summary table in
  the main [README](../README.md).
- **φ observability signal:** preliminary validation on Qwen3-8B (Tinker),
  multi-seed, pre-registered thresholds. Full technical note, raw traces,
  and reproducible scripts in [`validation/`](../validation/).
