'''
This script is used to preview the 3D point cloud data in .ply format.
'''

import open3d as o3d
import argparse
import os
import numpy as np

def preview_ply(ply_file, comparison_file=None):
    # Check if the file exists
    if not os.path.exists(ply_file):
        print(f"File {ply_file} does not exist.")
        return
    
    # Check if the comparison file exists
    if comparison_file is not None and not os.path.exists(comparison_file):
        print(f"File {comparison_file} does not exist.")
        comparison_file = None

    pcd = o3d.io.read_point_cloud(ply_file)
    
    if comparison_file is not None:
        comparison_pcd = o3d.io.read_point_cloud(comparison_file)
        # pcd.paint_uniform_color([1, 0, 0])
        # comparison_pcd.paint_uniform_color([0, 1, 0])
        pcd += comparison_pcd

    # Set point size 
    point_size = 0.05
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = point_size

    vis.add_geometry(pcd)
    vis.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ply_file", default="/media/cc/Elements/KITTI-360/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000001270_0000001549.ply", help="Path to the .ply file")
    parser.add_argument("--ply_file", default="/media/cc/Elements/KITTI-360/data_3d_semantics/test/2013_05_28_drive_0008_sync/static/0000001277_0000001491.ply", help="Path to the .ply file")
    parser.add_argument("--comparison_file", default="/home/cc/chg_ws/ros_ws/semantic_map_coda_ws/src/dsp_global_mapping/data/result.ply", help="Path to the .ply file")

    args = parser.parse_args()

    preview_ply(args.ply_file, args.comparison_file)
