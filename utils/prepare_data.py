import os
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from numpy.typing import NDArray

def parse_args() :
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type = str, required = True)
    parser.add_argument("--model", type = str, default = "facebook/sam2.1-hiera-large")

    parser.add_argument("--xmin", type = int)
    parser.add_argument("--ymin", type = int)
    parser.add_argument("--xmax", type = int)
    parser.add_argument("--ymax", type = int)

    parser.add_argument("--output_mask_path", type = str)

    args = parser.parse_args()
    return args


@torch.inference_mode()
def prepare_mask(
    image: NDArray, 
    predictor: SAM2ImagePredictor, 
    input_box: NDArray) -> Image.Image :
    
    """
    Converts a given bounding-box to a segmentation mask.

    Args:
        image (NDArray): RGB image.
        predictor (SAM2ImagePredictor): Initialized instance.
        input_box (NDArray): Bounding box in [x_min, y_min, x_max, y_max] format.

    Returns:
        Image.Image: Segmentation mask with the same dimensions as `input_image`.
    """

    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords = None,
        point_labels = None,
        box = input_box[None, :],
        multimask_output = False,
    )

    mask = masks[0].astype(np.uint8) * 255
    mask = Image.fromarray(mask)

    return mask


if __name__ == "__main__" :
    args = parse_args()

    if os.path.isdir(args.path) :
        raise NotImplementedError("Please use with a single image.")
    
    image = Image.open(args.path).convert("RGB")
    image = np.array(image)

    predictor = SAM2ImagePredictor.from_pretrained(args.model)
    
    input_box = np.array([args.x_min, args.y_min, args.x_max, args.y_max])

    mask = prepare_mask(image, predictor, input_box)
    
    plt.imshow(mask)
    if args.output_mask_path is not None :
        mask.save("segmentation_mask.png", format = 'PNG')
