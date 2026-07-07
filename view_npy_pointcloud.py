import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", help="Path to .npy point cloud file")
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size, e.g. 0.05")
    args = parser.parse_args()

    points = np.load(args.npy_path)

    print("Loaded:", args.npy_path)
    print("Shape:", points.shape)
    print("Dtype:", points.dtype)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected shape [N, C>=3], got {points.shape}")

    xyz = points[:, :3].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()