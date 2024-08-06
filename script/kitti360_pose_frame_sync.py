'''
This script is to generate a new text file with the same lines as the input text file, but with the lines sorted according to the ten-digit numbers extracted from the image filenames in the specified folder.
The script is to format the test pose data from the KITTI-360 dataset.
'''

import os
import argparse

def process_files(input_txt, image_folder, output_txt):
    # Read lines from the input text file
    with open(input_txt, 'r') as file:
        lines = file.readlines()
    
    # Get the list of image filenames from the folder
    image_filenames = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Extract ten-digit numbers from image filenames and sort them
    numbers = sorted([int(f.split('.')[0]) for f in image_filenames])
    
    # Create a dictionary mapping each number to its corresponding line from the input text file
    number_to_line = {numbers[i]: lines[i].strip() for i in range(len(numbers))}

    # Write the new lines to the output text file
    with open(output_txt, 'w') as file:
        for number in numbers:
            file.write(f"{number} {number_to_line[number]}\n")

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, default='/media/cc/Elements/KITTI-360_Scripts/kitti360Scripts/kitti360scripts/evaluation/semantic_slam/test_data/orbslam/test_poses_3.txt', help='Path to the input text file')
    parser.add_argument('--image_folder', type=str, default='/media/cc/Elements/KITTI-360/data_2d_test_slam/test_3/2013_05_28_drive_0002_sync/image_00/data_rect', help='Path to the folder containing the images')
    parser.add_argument('--output_txt', type=str, default='output.txt', help='Path to the output text file')
    args = parser.parse_args()

    process_files(args.input_txt, args.image_folder, args.output_txt)
