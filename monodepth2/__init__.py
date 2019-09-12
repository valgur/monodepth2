# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn.functional
from torchvision import transforms

from monodepth2 import layers
from monodepth2 import networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist, model_names


class Monodepth2:
    def __init__(self, model_name, models_dir="models", device=None):
        assert model_name in model_names
        self.model_name = model_name
        self.models_dir = models_dir

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        self.loaded = False
        self.feed_shape = None
        self.encoder = None
        self.depth_decoder = None

    def load(self):
        download_model_if_doesnt_exist(self.model_name, self.models_dir)

        model_path = os.path.join(self.models_dir, self.model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        print("   Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_shape = (loaded_dict_enc['height'], loaded_dict_enc['width'])
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        self.loaded = True

    def resize_input(self, image_tensor):
        return torch.nn.functional.interpolate(
            image_tensor, self.feed_shape, mode="bicubic", align_corners=True)

    @staticmethod
    def resize_output(image_tensor, original_shape):
        return torch.nn.functional.interpolate(
            image_tensor, original_shape, mode="bilinear", align_corners=True)

    @staticmethod
    def input_image_to_tensor(image):
        return transforms.ToTensor()(image).unsqueeze(0)

    def predict(self, input_image_tensor):
        assert isinstance(input_image_tensor, torch.Tensor)
        assert self.loaded

        input_image_tensor = input_image_tensor.to(self.device)
        features = self.encoder(input_image_tensor)
        outputs = self.depth_decoder(features)

        sigmoidal_disp = outputs[("disp", 0)]
        disp, depth = disp_to_depth(sigmoidal_disp, 0.1, 100)

        return disp, depth
