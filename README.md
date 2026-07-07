# Scene-Conditioned LiDAR Degeneracy Prediction

PointNeXt-based LiDAR degeneracy prediction for single-frame point clouds with
geometric statistical label generation.

The release model predicts:

- `0`: `tunnel_like`
- `1`: `open_like`
- `2`: `nondeg_or_other`

For `tunnel_like` samples, the horizontal degeneracy direction is represented as
an unoriented axis angle in `[0, 180)`. The default direction discretization uses
`num_dir_bins=12`, or 15 degrees per bin.

## Release Scope

This repository is a PointNeXt-only release. Datasets and large generated
experiment outputs are not included. The default released checkpoint is expected
under:

```text
log/deg_scene/smoke_pointnext_bs8_ep2/checkpoints/best_model.pth
```

The included smoke checkpoint is small enough for normal GitHub storage. If you
replace it with a larger checkpoint, prefer Git LFS or a GitHub Release asset
and keep the same directory layout.

## Repository Layout

```text
data_utils/
  DegSceneDataLoader.py
models/
  deg_scene_model.py
  pointcloud_ops.py
  pointnext_backbone.py
utils/
  deg_losses.py
  deg_metrics.py
tools/
  augment_deg_dataset_rotate_z.py
  bag_degeneracy_labeler.py
  build_scene_direction_pseudo_labels.py
  check_dir_bin_mapping.py
  plot_deg_dataset_figs.py
  smoke_test_deg_scene.py
generate_deg_scene_dataset.py
train_deg_scene.py
test_deg_scene.py
infer_deg_scene_pcd.py
view_npy_pointcloud.py
```

## Installation

Install PyTorch for your CUDA/runtime environment from the official PyTorch
instructions, then install the remaining Python dependencies:

```shell
pip install -r requirements.txt
```

## Data Organization

Point clouds are stored as one frame per file. The loader supports `.npy`,
`.npz`, `.txt`, `.csv`, `.pts`, `.xyz`, `.bin`, `.pkl`, and `.pickle`; the first
three columns must be xyz.

The recommended label format is CSV. Required columns are a point-cloud path
(`file_path`, `path`, `points`, or `point_path`) and a class label
(`scene_type` or `class_gt`). Common optional columns include:

```text
split, dir_x, dir_y, dir_z, dir_xy_valid, dir_exist_gt,
dir_bin_gt, dir_bin_valid, angle_deg, rz_gt, sample_weight
```

Example:

```csv
file_path,split,scene_type,dir_x,dir_y,dir_z,dir_xy_valid,dir_bin_gt,dir_bin_valid
points/frame_000001.npy,train,tunnel_like,1.0,0.0,0.0,1,0,1
points/frame_000002.npy,val,open_like,0.0,0.0,0.0,0,-1,0
points/frame_000003.npy,test,nondeg_or_other,0.0,0.0,0.0,0,-1,0
```

The dataset is not distributed with this repository. Put your local data under a
separate directory and pass it with `--data_root` and `--label_path`.

## Minimal Checks

Run the CPU smoke test:

```shell
python tools/smoke_test_deg_scene.py
```

Run evaluation on a prepared split:

```shell
python test_deg_scene.py \
  --label_path path/to/labels/deg_scene_labels.csv \
  --data_root path/to/dataset_root \
  --checkpoint log/deg_scene/smoke_pointnext_bs8_ep2/checkpoints/best_model.pth \
  --split test \
  --num_point 2048 \
  --input_channel 3 \
  --use_uniform_sample \
  --use_cpu
```

Run single `.pcd` inference:

```shell
python infer_deg_scene_pcd.py \
  --pcd path/to/frame.pcd \
  --checkpoint log/deg_scene/smoke_pointnext_bs8_ep2/checkpoints/best_model.pth \
  --num_point 2048 \
  --use_cpu
```

## Training

Synthetic data can be generated for a quick local workflow:

```shell
python generate_deg_scene_dataset.py --out_dir ./deg_scene_synth --num_each 100
```

Train a PointNeXt model:

```shell
python train_deg_scene.py \
  --label_path ./deg_scene_synth/labels/deg_scene_labels.csv \
  --data_root ./deg_scene_synth \
  --num_point 2048 \
  --batch_size 8 \
  --epoch 100 \
  --input_channel 3 \
  --backbone pointnext \
  --use_uniform_sample \
  --num_dir_bins 12 \
  --learning_rate 0.0003 \
  --lambda_cls 0.5 \
  --lambda_mag 0.2 \
  --lambda_tun 2.0 \
  --lambda_rz 0.2 \
  --lambda_lock 0.1 \
  --dir_bin_smoothing 0.2 \
  --log_dir pointnext_run
```

Training logs and checkpoints are written under `log/deg_scene/<log_dir>/`.
Only `log/deg_scene/smoke_pointnext_bs8_ep2/` is included in this release tree.

## Tools

The `tools/` directory is limited to label generation, label inspection, and
minimal model checks:

- `bag_degeneracy_labeler.py`: generate geometric-statistical labels from LiDAR
  bag / PointCloud2 data.
- `build_scene_direction_pseudo_labels.py`: build direction pseudo labels for
  scene-level degeneracy training.
- `augment_deg_dataset_rotate_z.py`: rotate point-cloud datasets around the z
  axis and update tunnel direction labels.
- `check_dir_bin_mapping.py`: verify direction-bin mapping and label quality.
- `plot_deg_dataset_figs.py`: inspect label statistics and diagnostic plots.
- `smoke_test_deg_scene.py`: run a random-data model/loss/metric smoke test.

## Citation

```bibtex
@misc{deg_scene_lidar_2026,
  title  = {Scene-Conditioned LiDAR Degeneracy Prediction With Geometric Statistical Label Generation},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Code release}
}
```
