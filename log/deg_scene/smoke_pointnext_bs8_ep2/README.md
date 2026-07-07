# Release Checkpoint Directory

This directory contains the default PointNeXt smoke checkpoint used by the
release README examples.

Expected layout:

```text
log/deg_scene/smoke_pointnext_bs8_ep2/
  checkpoints/
    best_model.pth
  eval_test_metrics.json
  eval_val_results.txt
  train.log
```

`checkpoints/best_model.pth` is about 52 MB in this release and can be stored in
normal GitHub Git history. If you replace it with a checkpoint above GitHub's
single-file limit, keep the same relative path in local runs and publish the
weight through Git LFS or a GitHub Release asset instead.
