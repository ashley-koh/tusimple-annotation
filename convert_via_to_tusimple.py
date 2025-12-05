#!/usr/bin/env python3
"""
Convert VIA (VGG Image Annotator) format lane annotations to TuSimple format.

This script reads a JSON file from VIA and converts the lane annotations to
TuSimple format using cubic spline interpolation.

Usage:
    python convert_via_to_tusimple.py [input_json] [output_json] [--raw-file-prefix <prefix>] [--image-height <height>]

Example:
    python convert_via_to_tusimple.py via_annotations.json labels_lvlane.json --raw-file-prefix "clips/LVLane_train_sunny/"
    python convert_via_to_tusimple.py --image-height 120  # For 320x120 images
    python convert_via_to_tusimple.py  # Uses defaults: via_annotations.json -> labels_tusimple.json (720p)
"""

import argparse
import json

import numpy as np
from scipy.interpolate import CubicSpline


def convert_via_to_tusimple(
    input_json_path, output_json_path, raw_file_prefix="", image_height=720
):
    """
    Convert VIA format annotations to TuSimple format.

    Args:
        input_json_path: Path to input VIA JSON file
        output_json_path: Path to output TuSimple JSON file
        raw_file_prefix: Prefix to add to raw_file paths (e.g., "clips/LVLane_train_sunny/")
        image_height: Height of the images in pixels (default: 720 for 1280x720 images)
    """
    # Create h_samples array based on image height
    # For 720p: y from 160 to 710 in steps of 10
    # For other resolutions, scale proportionally
    if image_height == 720:
        h_samples = list(range(160, 720, 10))
    elif image_height == 120:
        # For 120p: y from 27 to 119 in steps of 2 (approximately same coverage)
        h_samples = list(range(27, 120, 2))
    else:
        # General case: sample from ~22% to ~99% of image height
        start_y = int(image_height * 0.22)
        end_y = int(image_height * 0.99)
        step = max(1, int((end_y - start_y) / 55))  # Try to get ~56 samples
        h_samples = list(range(start_y, end_y, step))

    # Load VIA JSON file
    with open(input_json_path, "r") as f:
        via_data = json.load(f)

    # Extract image metadata from VIA format
    # VIA JSON has structure: {"_via_img_metadata": {image_key: {filename, regions, ...}}}
    if "_via_img_metadata" in via_data:
        data = via_data["_via_img_metadata"]
    else:
        # Fallback for older VIA format or direct metadata
        data = via_data

    # Process each image
    with open(output_json_path, "w") as outfile:
        for each_image in data:
            label = data[each_image]
            filename = label["filename"]
            lanes = np.array([], dtype=int)
            count_lanes = 0

            # Process each lane in the image
            for each_lane in label["regions"]:
                label_coordinates = each_lane["shape_attributes"]
                x = label_coordinates["all_points_x"]
                y = label_coordinates["all_points_y"]

                # Sort points by y-coordinate
                data_samples = [(slp, cls) for slp, cls in zip(x, y)]
                data_samples.sort(key=lambda x: x[1])

                x_cor = []
                y_cor = []
                for sample in data_samples:
                    x_cor.append(sample[0])
                    y_cor.append(sample[1])

                # Create cubic spline interpolation
                cs = CubicSpline(y_cor, x_cor)

                # Generate TuSimple format coordinates for all h_samples
                # Initialize with -2 (missing value)
                tusimple_x = np.ones(len(h_samples), dtype=int) * (-2)

                # Fill in x-coordinates where the lane exists
                y_min = min(y_cor)
                y_max = max(y_cor)

                for idx, h in enumerate(h_samples):
                    # Only interpolate if h is within the lane's y-range
                    if y_min <= h <= y_max:
                        tusimple_x[idx] = int(cs(h))

                # Convert to 2D array for stacking
                tusimple_x = np.array([tusimple_x])

                # Stack lanes
                count_lanes += 1
                if count_lanes > 1:
                    lanes = np.vstack([lanes, tusimple_x])
                else:
                    lanes = tusimple_x

            # Create TuSimple format dictionary
            dictionary = {
                "lanes": lanes.tolist(),
                "h_samples": h_samples,
                "raw_file": raw_file_prefix + filename,
            }

            # Write to output file (one JSON object per line)
            json_object = json.dumps(dictionary)
            outfile.write(json_object)
            outfile.write("\n")

    print(f"Conversion complete! Output saved to: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VIA format lane annotations to TuSimple format"
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default="racetrack_via.json",
        help="Path to input VIA JSON file (default: via_annotations.json)",
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        default="labels_tusimple.json",
        help="Path to output TuSimple JSON file (default: labels_tusimple.json)",
    )
    parser.add_argument(
        "--raw-file-prefix",
        default="",
        help="Prefix to add to raw_file paths (e.g., 'clips/LVLane_train_sunny/')",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=720,
        help="Height of the images in pixels (default: 720 for 1280x720, use 120 for 320x120)",
    )

    args = parser.parse_args()

    convert_via_to_tusimple(
        args.input_json, args.output_json, args.raw_file_prefix, args.image_height
    )


if __name__ == "__main__":
    main()
