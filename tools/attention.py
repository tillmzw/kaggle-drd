#!/usr/bin/env python3

import sys
import os
import argparse
import logging

from PIL import Image
import torch
import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DRDNet

""" LOAD CNN VISUALIZATIONS LIB FROM DIR """
VIS_DIR = '%s/pytorch-cnn-visualizations/src' % os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(VIS_DIR):
    raise RuntimeError("Visualization project not found: %s" % VIS_DIR)
else:
    print(VIS_DIR)
    sys.path.insert(0, VIS_DIR)

from misc_functions import (get_example_params,
                            preprocess_image,
                            convert_to_grayscale,
                            save_gradient_images, 
                            format_np_output)
from gradcam import GradCam
from guided_backprop import GuidedBackprop
from guided_gradcam import guided_grad_cam


""" END IMPORTS """

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format=('%(asctime)s %(levelname)8s %(name)10s %(lineno)3d -- %(message)s'),
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help="Output image name", default=None)
    parser.add_argument('-s', '--state', help="Model state file to load")
    parser.add_argument('-l', '--layer', type=int, default=-1)
    # `class` is a reserved keyword
    parser.add_argument('-c', '--class', type=int, default=None, help="Class of input image", dest="target_class")
    parser.add_argument('input', help="Input image")

    args = parser.parse_args()

    if not args.output:
        args.output = "%s/%s_gradcam.%s" % (
                os.path.dirname(args.input),
                os.path.basename(args.input),
                args.input.split(".")[-1])

    if not os.path.isfile(args.state):
        raise RuntimeError("Model state file does not exist: %s" % args.state)

    if not os.path.isfile(args.input):
        raise RuntimeError("Input file doesn't exist: %s" % args.input)

    logger.info("Generating activation map of %s: %s" % (args.input, args.output))

    model = DRDNet()
    model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))

    orig_image = Image.open(args.input).convert("RGB")
    prep_img = preprocess_image(orig_image)

    if args.layer == -1:
        cams = (GradCam(model, target_layer=i) for i in range(len(model.features._modules)))
    else:
        cams = (GradCam(model, target_layer=args.layer), )

    for i, gcv2 in enumerate(cams):
        try:
            # Generate cam mask
            cam = gcv2.generate_cam(prep_img, args.target_class)

            # Guided backprop
            GBP = GuidedBackprop(model)
            # Get gradients
            guided_grads = GBP.generate_gradients(prep_img, args.target_class)

            # Guided Grad cam
            cam_gb = guided_grad_cam(cam.T, guided_grads)
            #save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
            gradient = convert_to_grayscale(cam_gb)

            gradient = gradient - gradient.min()
            gradient /= gradient.max()
            im = format_np_output(gradient)
            im = Image.fromarray(im)

            if args.layer == -1:
                e = args.output.split(".")
                output = "%s_%02d.%s" % (".".join(e[:-1]), i, e[-1])
            else:
                output = args.output 

            im.save(output)
            logger.info("GradCam for layer %d saved in %s" % (i, output))
        except Exception as e:
            logger.error("An error occured calculating the gradcam in layer %d: %s" % (i, e))
