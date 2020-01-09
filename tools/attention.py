#!/usr/bin/env python3

import sys
import os
import argparse
import logging

from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DRDNet
from cnn_visualize.misc_functions import (preprocess_image,
                                          convert_to_grayscale,
                                          apply_colormap_on_image,
                                          format_np_output)
from cnn_visualize.gradcam import GradCam
from cnn_visualize.guided_backprop import GuidedBackprop
from cnn_visualize.guided_gradcam import guided_grad_cam


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format=('%(asctime)s %(levelname)8s %(name)10s %(lineno)3d -- %(message)s'),
                        datefmt="%H:%M:%S")

    this_path = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    default_model = os.path.join(this_path, "model.pth")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help="Path for output images", metavar=this_path, default=this_path)
    parser.add_argument('-s', '--state', help="Model state file to load", metavar=default_model, default=default_model)
    parser.add_argument('-l', '--layer', type=int, default=-1, help="For which layer to generate a gradcam. Defaults to all.")
    # `class` is a reserved keyword
    parser.add_argument('-c', '--class', type=int, default=None, help="Class of input image. If left blank, the most probable class from the predictions will be used.", dest="target_class")
    parser.add_argument('-g', '--gif', action="store_true", default=False, help="Generate a GIF from all images instead of single images.")
    parser.add_argument('input', help="Input image")

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise RuntimeError("Output is not a directory: %s" % args.path)

    if not os.path.isfile(args.state):
        raise RuntimeError("Model state file does not exist: %s" % args.state)

    if not os.path.isfile(args.input):
        raise RuntimeError("Input file doesn't exist: %s" % args.input)

    input_name = ".".join(os.path.basename(args.input).split(".")[:-1])

    logger.info("Generating activation map of %s in %s" % (args.input, args.path))

    model = DRDNet()
    # always keep this in CPU, because for single images this is fast enough
    model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))

    orig_image = Image.open(args.input).convert("RGB")
    prep_img = preprocess_image(orig_image)

    mods = model.features._modules

    # take all layers
    layer_cams = (
        (idx, mod, GradCam(model, target_layer=idx)) for idx, mod in enumerate(mods)
    )

    if 0 <= args.layer <= len(mods):
        # remove all layers not requested. slow, but ¯\_(ツ)_/¯
        layer_cams = filter(lambda el: args.layer == el[0], layer_cams)

    outputs = []
    for i, name, gcv2 in layer_cams:
        try:
            # Generate cam mask
            activation_map = gcv2.generate_cam(prep_img, args.target_class)

            # TODO: P: am i discarding negative values?
            heatmap, heatmap_img = apply_colormap_on_image(orig_image, activation_map, 'hsv')

            if args.gif:
                outputs.append(heatmap_img)
                logger.info("Added layer %d (%s) to GIF" % (i, name))
            else:
                output = os.path.join(args.path, "gradcam_%s__%02d_%s.png" % (input_name, i, name))
                # TODO: remove alpha channel if present?
                heatmap_img.save(output)
                logger.info("GradCam saved in %s" % output)
        except Exception as e:
            logger.error("An error occured calculating the gradcam in layer %d: %s" % (i, e))
            logger.exception(e)

    if args.gif and len(outputs) > 1:
        name = os.path.join(args.path, "gradcam_%s.gif" % input_name)
        logger.info("Saving GIF in %s" % name)
        outputs[0].save(name,
                        save_all=True,
                        append_images=outputs[1:],
                        duration=1000,
                        loop=0)
