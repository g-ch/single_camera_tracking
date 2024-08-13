import cv2
import numpy as np
import argparse
import os

def read_disparity_and_save_depth(disparity_image_path, output_npy_path, focal_length, baseline):
    """
    Reads a disparity image, converts it to real depth, and saves the depth image as a .npy file.

    Parameters:
    disparity_image_path (str): Path to the disparity image file.
    output_npy_path (str): Path to save the depth .npy file.
    focal_length (float): Focal length of the camera in pixels.
    baseline (float): Baseline distance between the two cameras in meters.
    """
    # Step 1: Load the disparity image
    disparity_image = cv2.imread(disparity_image_path, cv2.IMREAD_GRAYSCALE)
    
    if disparity_image is None:
        raise ValueError(f"Cannot load the disparity image from {disparity_image_path}")
    
    disparity_image = disparity_image.astype(np.float32)

    # Step 2: Convert disparity to depth
    # Avoid division by zero by setting a minimum disparity
    min_disparity = 1e-5
    disparity_image[disparity_image == 0] = min_disparity

    depth_image = (focal_length * baseline) / disparity_image

    # Step 3: Save the depth image as a .npy file
    np.save(output_npy_path, depth_image)

# Example usage:
# read_disparity_and_save_depth('disparity_image.png', 'depth_image.npy', focal_length=0.8, baseline=0.5)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--disparity_directory", default="/media/cc/Elements/KITTI-360/depth/sgm/2013_05_28_drive_0000_sync/disparities", help="Path to the disparity image directory")
    parser.add_argument("--output_directory", default="/media/cc/Elements/KITTI-360/depth/sgm/2013_05_28_drive_0000_sync/sequences/0", help="Path to save the depth .npy file")
    parser.add_argument("--focal_length", type=float, default=552.554261, help="Focal length of the camera in pixels")
    parser.add_argument("--baseline", type=float, default=0.6, help="Baseline distance between the two cameras in meters")

    args = parser.parse_args()

    # Check if the output directory exists, if not, create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    # Iterate through all disparity images in the directory
    for file in os.listdir(args.disparity_directory):
        if file.endswith(".png"):

            disparity_image_path = os.path.join(args.disparity_directory, file)
            output_npy_path = os.path.join(args.output_directory, file.replace(".png", ".npy"))
            
            read_disparity_and_save_depth(disparity_image_path, output_npy_path, args.focal_length, args.baseline)
            print(f"Saved depth image to {output_npy_path}")
