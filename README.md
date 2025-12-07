# TuSimple Annotation Workflow

This repository is a fork of the original [tusimple-annotation](https://github.com/zillur-av/tusimple-annotation) by Zillur Rahman and Brendan Tran Morris. It provides a complete workflow to create a lane detection dataset in TuSimple format, from raw images to final dataset.

## Overview

The workflow consists of the following steps:
1. **Sample Images**: Sample a subset of raw images for annotation.
2. **Annotate with VIA**: Use the VGG Image Annotator (VIA) to label lanes.
3. **Convert Annotations**: Convert VIA JSON to TuSimple JSON format.
4. **Generate Dataset**: Create binary and instance masks, along with train/val splits.

## Step 1: Sample Images

Prepare an input directory containing your raw images, named in alphanumerical order (e.g., `0001.png`, `0002.png`, etc.). Images must be in a format readable by VIA (e.g., PNG, JPG).

Run the sampling script to select a subset of images evenly spaced across the dataset:

```bash
python sample_images.py <input_dir> --num_images <desired_number>
```

This creates `racetrack_training_data/image/` with the sampled images, ready for annotation.

## Step 2: Annotate with VIA

Use the [VIA annotation tool](https://www.robots.ox.ac.uk/~vgg/software/via/) to annotate lanes.

1. Download the VIA zip file and open `via.html` in a browser.
2. Click `Add Files` and select your images.
3. Select `polyline` from the `Region Shape` section.
4. Draw polylines on each lane, starting from the bottom (highest y-value) to the top (lowest y-value). Use at least 6-7 points per lane for smooth curves.
   - Ensure x-coordinates are in order (increasing or decreasing consistently).
   - Example valid: [239, 250, 270, 320, 380, ... 570]
   - Invalid: [239, 250, 245, 270, 320, 380, ... 570]
5. Save the project periodically.
6. Export annotations as JSON (e.g., `racetrack_via.json`).

![Demo annotation](sample-annotation.png)

## Step 3: Convert VIA Annotations to TuSimple Format

Run the conversion script to transform VIA JSON into TuSimple-compatible JSON.

```bash
python convert_via_to_tusimple.py
```

This generates `labels_tusimple.json` with the required format for lane detection models.

## Step 4: Generate the Dataset

Run the dataset generation script to create masks and splits.

```bash
python generate_masks.py
```

This creates:
- `training_data/image/`: Copied images.
- `training_data/gt_image_binary/`: Binary masks (all lanes combined).
- `training_data/gt_image_instance/`: Instance masks (lanes with progressive brightness from left to right).
- `training_data/train.txt` and `training_data/val.txt`: Train/val splits with full paths.

The instance masks have lanes sorted by leftmost position, with intensities from dark gray (64) to white (255).

**Note**: The generated dataset is formatted for compatibility with the [lanenet-lane-detection-pytorch](https://github.com/IrohXu/lanenet-lane-detection-pytorch) repository on GitHub.

## Citation

If you use this tool, please cite the original paper by Rahman and Morris, as well as the [VIA annotation tool](https://www.robots.ox.ac.uk/~vgg/software/via/):

```
@article{rahman2023lvlane,
  title={LVLane: Deep Learning for Lane Detection and Classification in Challenging Conditions},
  author={Rahman, Zillur and Morris, Brendan Tran},
  journal={2023 IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year={2023}
}
```
