# Episode plans

- `train/`: links to current immutable training plans.
- `validation/`: separate plans for model selection, with distinct seeds.
- `benchmark/`: copied frozen evaluation suites, organized by task and tier.
- `legacy`: a link to earlier source-project plans for reproducibility.

Keep existing plans immutable. Directory grouping does not change scenario schema:
both validation and benchmark plans use `split: eval`. Existing plans preserve their
episode IDs, scene seeds, and fingerprints. Use fresh output folders for generation.
