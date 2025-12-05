import json
import os
from argparse import ArgumentParser as ArgParse

import cv2
import numpy as np


# Color palette for lane visualization
def getcolor(code):
    if code == 1:
        return (0, 255, 255)
    if code == 2:
        return (0, 255, 0)
    if code == 3:
        return (255, 255, 0)
    if code == 4:
        return (255, 0, 0)
    if code == 5:
        return (0, 0, 255)
    if code == 6:
        return (45, 88, 200)
    if code == 7:
        return (200, 22, 100)


def process(root_dir, json_file):
    # Image visualizationz
    # cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    count = 1

    # Open lane and class ground truth files
    with open(root_dir + json_file, "r") as file:
        json_lines = file.readlines()
        line_index = 0

        # Iterate over each image
        while line_index < len(json_lines):
            json_line = json_lines[line_index]
            i = 0
            sample = json.loads(json_line)
            lanes = sample["lanes"]
            raw_file = root_dir + sample["raw_file"]
            im = cv2.imread(raw_file)

            # Display image and draw lane
            while i < len(lanes):
                polyline = []
                for v in range(0, len(sample["h_samples"]) - 1):
                    point_h_begin = sample["h_samples"][v]
                    point_h_end = sample["h_samples"][v + 1]
                    point_w_begin = lanes[i][v]
                    point_w_end = lanes[i][v + 1]

                    if point_w_begin != -2 and point_w_end != -2:
                        cv2.circle(
                            im, (point_w_begin, point_h_begin), 3, getcolor(i + 1)
                        )
                i += 1

            cv2.imshow("image", im)

            code = cv2.waitKey(4000)
            # Code for navigation
            # If z is pressed exit from the program
            if code == ord("z"):
                # closeall(output, file)
                return
            if code == ord("l"):
                line_index += 100
                count += 100
                continue
            if code == ord("k"):
                line_index += 50
                count += 50
                continue
            if code == ord("j"):
                line_index += 20
                count += 20
                continue
            if code == ord("h"):
                line_index += 10
                count += 10
                continue
            if code == ord("r"):
                line_index -= 1
                count -= 1
                continue
            count += 1
            line_index += 1


if __name__ == "__main__":
    ap = ArgParse()
    ap.add_argument("--root", type=str, default="")
    ap.add_argument("--labels", type=str, default="labels_tusimple.json")

    args = ap.parse_args()

    process(args.root, args.labels)
