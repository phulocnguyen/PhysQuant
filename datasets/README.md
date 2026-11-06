# Dataset locations

- `native_hdf5/`: links to the existing Object Understanding and Relational
  Understanding 7,000-episode collections, plus earlier local recordings.
- `lerobot/`: link to the existing Object Understanding export.
- `rollouts/`: destination for new closed-loop evaluation results.

No recordings or exports were moved or rewritten. Links can point to active runs.
Dataset contents are ignored by Git and excluded from evaluation bundles. Pass an
explicit output path to collect or export onto the large external storage volume.
