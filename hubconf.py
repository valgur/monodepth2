# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import torch

import networks
from utils import get_state_file as _get_state_file, download_model as _download_model

dependencies = ['torch', 'torchvision']


def ResnetEncoder(pretrained_model=None, kind="depth", map_location=None, num_layers=18, num_input_images=None):
    num_input_images = num_input_images or 1 if kind == "depth" else 2
    if pretrained_model == "imagenet":
        encoder = networks.ResnetEncoder(num_layers, True, num_input_images)
        encoder.eval()
        return encoder
    encoder = networks.ResnetEncoder(num_layers, False, num_input_images)
    if pretrained_model:
        if kind == "depth":
            encoder_path = _get_state_file(pretrained_model, "encoder.pth")
        elif kind == "pose":
            encoder_path = _get_state_file(pretrained_model, "pose_encoder.pth")
        else:
            raise ValueError("kind must be either 'depth' or 'pose'")
        if not os.path.exists(encoder_path):
            _download_model(pretrained_model)
        print("-> Loading pretrained encoder " + encoder_path)
        loaded_dict_enc = torch.load(encoder_path, map_location=map_location)
        encoder.feed_height = loaded_dict_enc['height']
        encoder.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()
    return encoder


def DepthDecoder(pretrained_model=None, map_location=None, num_ch_enc=(64, 64, 128, 256, 512), scales=tuple(range(4)),
                 **kwargs):
    decoder = networks.DepthDecoder(num_ch_enc=num_ch_enc, scales=scales, **kwargs)
    if pretrained_model:
        decoder_path = _get_state_file(pretrained_model, "depth.pth")
        if not os.path.exists(decoder_path):
            _download_model(pretrained_model)
        print("-> Loading pretrained depth decoder " + decoder_path)
        loaded_dict = torch.load(decoder_path, map_location=map_location)
        decoder.load_state_dict(loaded_dict)
    decoder.eval()
    return decoder


def PoseDecoder(pretrained_model=None, map_location=None, num_ch_enc=(64, 64, 128, 256, 512), num_input_features=1,
                num_frames_to_predict_for=None, stride=1):
    decoder = networks.PoseDecoder(num_ch_enc, num_input_features, num_frames_to_predict_for, stride)
    if pretrained_model:
        decoder_path = _get_state_file(pretrained_model, "pose.pth")
        if not os.path.exists(decoder_path):
            _download_model(pretrained_model)
        print("-> Loading pretrained pose decoder " + decoder_path)
        loaded_dict = torch.load(decoder_path, map_location=map_location)
        decoder.load_state_dict(loaded_dict)
    decoder.eval()
    return decoder
