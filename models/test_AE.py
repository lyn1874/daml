#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20.02.20 at 12:02
This function tests the proposed architecture in terms of input dimension, output dimension
@author: li
"""
import tensorflow as tf
import models.AE as AE
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Training anomaly detection model using end2end')
parser.add_argument('-ds', '--data_set', type=str, default='ucsd1', metavar='DATA_SET',
                    help='the used dataset can be ucsd1, ucsd2, avenue, shanghaitech, kth_walk')
parser.add_argument('-bs', '--batch_size', type=int, default=64, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-ne', '--num_encode_layer', type=int, default=4, metavar='NUM_ENCODER_LAYER',
                    help='the number of encoder layers')
parser.add_argument('-nd', '--num_decode_layer', type=int, default=4, metavar='NUM_DECODER_LAYER',
                    help='the number of decoder layers')
parser.add_argument('-od', '--output_dim', type=int, default=4, metavar='OUTPUT_DIM',
                    help='the last dimension of the input')
parser.add_argument('-regu', '--regu_par', type=float, default=0.001, metavar='L2_REGU_PAR',
                    help='L2 regularization parameter')
parser.add_argument('-zmr', '--z_mse_ratio', type=float, default=0.001, metavar='Z_MSE_RATIO',
                    help='weight for latent space mse loss')
parser.add_argument('-io', '--input_option', type=str, default="original", metavar='INPUT_OPTION',
                    help='whether the input for the model is cropped or not')
parser.add_argument('-ao', '--augment_option', type=str, default="none", metavar='AUGMENT_OPTION',
                    help='whether to apply the augmention on the training data')
parser.add_argument('-dv', '--darker_value', type=float, default=0.4, metavar='DARKER_VALUE',
                    help='the degree of brightness, 0.0 means using original frames')
parser.add_argument('-mt', '--model_type', type=str, default='2d_2d_pure_unet', metavar='MODEL_TYPE',
                    help='the selected base model')
parser.add_argument('-mm', '--motion_model', type=str, default='conv3d', metavar='MOTION_MODEL',
                    help='the selected motion method')
parser.add_argument('-lo', '--learn_opt', type=str, default='learn_fore', metavar='LEARN_OPT',
                    help='Whether the background frame needs to be subtracted from the original frame')

args = parser.parse_args()


def check_model(model_type, motion_method, data_set):
    tf.reset_default_graph()
    args.data_set = data_set
    args.model_type = model_type
    args.motion_model = motion_method
    args.time_step = 6
    args.batch_size = 10
    imh, imw, ch = 128, 192, 3
    x_input_placeholder = tf.placeholder(shape=[args.time_step+1, args.batch_size, imh, imw, ch],
                                         dtype=tf.float32)
    args.num_encoder_layer = 4
    args.num_decoder_layer = 4
    model_use = AE.DAML(args)
    p_x_recons, p_x_pred, latent_space_gt, latent_space_pred = model_use.forward(x_input_placeholder)

    print("Reconstruction------------------", p_x_recons)
    print("Prediction----------------------", p_x_pred)
    print("Latent code gt------------------", latent_space_gt)
    print("Latent code pred----------------", latent_space_pred)

    var_tr = tf.trainable_variables()
    print("-----------------Printing trainable variables-------------------")
    print("---------ENCODER-------------")
    [print(v) for v in var_tr if 'encoder' in v.name and 'kernel' in v.name]
    print("---------Motion Model--------")
    [print(v) for v in var_tr if 'motion' in v.name and 'kernel' in v.name]
    print("--------Decoder--------------")
    [print(v) for v in var_tr if 'decoder' in v.name and 'kernel' in v.name]


    