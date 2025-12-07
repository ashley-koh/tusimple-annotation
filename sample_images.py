import argparse
import os
import shutil


def linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def main():
    parser = argparse.ArgumentParser(
        description="Sample images from input directory to get exactly num_images images in alphanumerical order."
    )
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument(
        "--output_dir",
        default="racetrack_training_data/image",
        help="Output directory for sampled images (default: racetrack_training_data/image)",
    )
    parser.add_argument(
        "--num_images", type=int, required=True, help="Number of images to sample"
    )
    args = parser.parse_args()

    # List and sort files alphanumerically
    files = os.listdir(args.input_dir)
    files.sort()

    total_files = len(files)
    if total_files == 0:
        print("No images found in input directory.")
        return

    num_images = args.num_images
    if num_images > total_files:
        print(
            f"Requested {num_images} images but only {total_files} available. Taking all."
        )
        num_images = total_files

    # Calculate evenly spaced indices
    indices = [int(x) for x in linspace(0, total_files - 1, num_images)]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy and rename sampled images
    for count, idx in enumerate(indices):
        src = os.path.join(args.input_dir, files[idx])
        ext = os.path.splitext(files[idx])[1]
        new_name = f"{count:04d}{ext}"
        dst = os.path.join(args.output_dir, new_name)
        shutil.copy(src, dst)
        print(f"Copied {files[idx]} as {new_name} to {args.output_dir}")


if __name__ == "__main__":
    main()
