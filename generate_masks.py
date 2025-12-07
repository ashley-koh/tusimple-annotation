import json
import os
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def parse_tusimple_data(json_path):
    """Parse TuSimple JSON file and return list of entries."""
    entries = []
    with open(json_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def coordinates_to_points(lane, h_samples):
    """Convert lane coordinates to (x,y) points, filtering invalid values."""
    points = []
    for x, y in zip(lane, h_samples):
        if x >= 0:  # Valid coordinate
            points.append([int(x), int(y)])
    return np.array(points, dtype=np.int32)


def create_binary_mask(lanes, h_samples, img_shape=(180, 320), lane_width=8):
    """Create binary mask with all lanes combined."""
    mask = np.zeros(img_shape, dtype=np.uint8)

    for lane in lanes:
        points = coordinates_to_points(lane, h_samples)
        if len(points) > 1:
            # Draw thick polyline
            cv2.polylines(mask, [points], False, (255,), lane_width, cv2.LINE_AA)
            # Fill small gaps with circles at each point
            for point in points:
                cv2.circle(mask, tuple(point), lane_width // 2, (255,), -1)

    return mask


def create_instance_mask(lanes, h_samples, img_shape=(180, 320), lane_width=8):
    """Create a single instance mask with lanes in different intensities, brighter from left to right."""
    mask = np.zeros(img_shape, dtype=np.uint8)

    # Filter lanes with at least one valid point
    valid_lanes = [lane for lane in lanes if any(x >= 0 for x in lane)]

    if not valid_lanes:
        return mask

    # Sort lanes by leftmost x coordinate
    valid_lanes.sort(key=lambda lane: min(x for x in lane if x >= 0))

    # Assign intensities from dark gray to white
    num_lanes = len(valid_lanes)
    intensities = np.linspace(64, 255, num_lanes, dtype=np.uint8)

    for lane, intensity in zip(valid_lanes, intensities):
        points = coordinates_to_points(lane, h_samples)

        if len(points) > 1:
            # Draw thick polyline for this instance
            cv2.polylines(
                mask, [points], False, (int(intensity),), lane_width, cv2.LINE_AA
            )
            # Fill small gaps
            for point in points:
                cv2.circle(mask, tuple(point), lane_width // 2, (int(intensity),), -1)

    return mask


def setup_directories(base_dir):
    """Create the required directory structure."""
    dirs = [
        base_dir,
        os.path.join(base_dir, "gt_image_binary"),
        os.path.join(base_dir, "gt_image_instance"),
        os.path.join(base_dir, "image"),
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def copy_images(entries, target_image_dir):
    """Copy source images to target directory with sequential naming."""
    for i, entry in enumerate(entries):
        src_path = entry["raw_file"]
        if os.path.exists(src_path):
            filename = f"{i:04d}.png"
            dst_path = os.path.join(target_image_dir, filename)
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Source image not found: {src_path}")


def generate_dataset(
    json_path, output_dir="racetrack_training_data", test_size=0.2, random_state=42
):
    """Main function to generate the complete dataset."""

    # Setup directories
    setup_directories(output_dir)

    # Parse data
    entries = parse_tusimple_data(json_path)
    print(f"Found {len(entries)} entries in JSON file")

    # Copy images
    image_dir = os.path.join(output_dir, "image")
    copy_images(entries, image_dir)

    # Generate masks
    binary_dir = os.path.join(output_dir, "gt_image_binary")
    instance_dir = os.path.join(output_dir, "gt_image_instance")

    for i, entry in enumerate(entries):
        # Generate masks
        binary_mask = create_binary_mask(entry["lanes"], entry["h_samples"])
        instance_mask = create_instance_mask(entry["lanes"], entry["h_samples"])

        # Save binary mask
        filename = f"{i:04d}.png"
        binary_path = os.path.join(binary_dir, filename)
        cv2.imwrite(binary_path, binary_mask)

        # Save instance mask
        instance_path = os.path.join(instance_dir, filename)
        cv2.imwrite(instance_path, instance_mask)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(entries)} entries")

    # Create train/val splits
    filenames = [f"{i:04d}" for i in range(len(entries))]
    train_files, val_files = train_test_split(
        filenames, test_size=test_size, random_state=random_state
    )

    # Write split files with full paths
    base_path = "./data/racetrack_training_data"
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        lines = [
            f"{base_path}/image/{fn}.png {base_path}/gt_image_binary/{fn}.png {base_path}/gt_image_instance/{fn}.png"
            for fn in train_files
        ]
        f.write("\n".join(lines))

    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        lines = [
            f"{base_path}/image/{fn}.png {base_path}/gt_image_binary/{fn}.png {base_path}/gt_image_instance/{fn}.png"
            for fn in val_files
        ]
        f.write("\n".join(lines))

    print("Dataset generated successfully!")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")


if __name__ == "__main__":
    # Usage example
    generate_dataset("labels_tusimple.json", "racetrack_training_data")
