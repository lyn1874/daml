#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:48:32 2019
This script includes the implementation of the base model and motion model
1. The high-level feature (z_input and z_gt_for_predict) for each input frame 
    is extracted using the encoder of the base model (AE_2d_2d_unet)
    The number of input is (input_frame + num_prediction)
2. Then the latent space z_input are send to the motion model to predict the 
    latent space for next time step \hat{z}_for_predict
3. At the same time, the latent space z_input are send to the decoder 
    of the base model (AE_2d_2d_unet) for reconstruction
@author: li
"""
import tensorflow as tf
import numpy as np


class MotionModel(object):
    """This function consider to only remember the motion information in the latent space
    """

    def __init__(self, args):
        super(MotionModel, self).__init__()
        self.data_set = args.data_set
        num_frame = args.time_step
        iterr = int(np.ceil(np.log2(num_frame)))
        iterr = [4 if iterr >= 4 else iterr][0]
        self.num_motion_layer = iterr
        self.motion_model = args.motion_method
        self.model_type = args.model_type 

    def build_conv3d_latent(self, latent_space):
        """this function is for predicting the latent space using Conv3D"""
        if isinstance(latent_space, list) is True:
            latent_space_array = tf.concat([latent_space], axis=0)
        else:
            latent_space_array = latent_space
        num_frame, batch_size, fh, fw, ch = latent_space_array.get_shape().as_list()
        latent_space_array = tf.transpose(latent_space_array, perm=(1, 0, 2, 3, 4))
        for i in range(self.num_motion_layer):
            latent_space_array = tf.keras.layers.Conv3D(filters=ch, kernel_size=(2, 3, 3), strides=(2, 1, 1),
                                                        padding='same', name='latent_%d' % i)(latent_space_array)
            latent_space_array = tf.keras.layers.BatchNormalization(name="latent_%d_bn" % i)(latent_space_array)
            if i != self.num_motion_layer - 1:
                latent_space_array = tf.keras.layers.LeakyReLU(name="latent_%d_lr" % i)(latent_space_array)
            else:
                if self.data_set == "avenue" or "moving_mnist" in self.data_set or self.model_type == "many_to_one":
                    latent_space_array = tf.nn.tanh(latent_space_array)
                else:
                    latent_space_array = tf.keras.layers.LeakyReLU(name="latent_%d_lr" % i)(latent_space_array)
            if i == self.num_motion_layer - 1:
                print(i, latent_space_array)
        left_dim = latent_space_array.get_shape().as_list()[1]
        if left_dim != 1:
            latent_space_array = tf.keras.layers.AveragePooling3D(pool_size=(left_dim, 1, 1))(latent_space_array)
        latent_space_array = tf.squeeze(latent_space_array, axis=1)
        return latent_space_array

    def build_convlstm2d(self, latent_space):
        """This function predicts the latent code for the future using convlstm2d
        Args:
            latent_space: [num_frame, batch_size, fh, fw, ch]
        """
        if isinstance(latent_space, list) is True:
            latent_space_array = tf.concat([latent_space], axis=0)
        else:
            latent_space_array = latent_space
        num_frame, batch_size, fh, fw, ch = latent_space_array.get_shape().as_list()
        latent_space_array = tf.transpose(latent_space_array, perm=(1, 0, 2, 3, 4))
        for i in range(self.num_motion_layer):
            return_seqence = [True if i < self.num_motion_layer-1 else False][0]
            latent_space_array = tf.keras.layers.ConvLSTM2D(filters=ch, kernel_size=3, strides=1, padding='same',
                                                            return_sequences=return_seqence)(latent_space_array)
            if i < self.num_motion_layer - 1:
                latent_space_array = tf.keras.layers.BatchNormalization(name="latent_%d_bn" % i)(latent_space_array)
        return latent_space_array

    def build_motion_latent(self, latent_space):
        with tf.variable_scope("build_motion_latent"):
            if self.motion_model == "conv3d":
                latent_space_update = self.build_conv3d_latent(latent_space)
            elif self.motion_model == "convlstm":
                latent_space_update = self.build_convlstm2d(latent_space)
        return latent_space_update



def create_enc_layers(num_encoder_layer, feature_root, max_dim):
    enc_conv1, enc_conv2, enc_bn_1, enc_bn_2 = [], [], [], []

    for i in range(num_encoder_layer):
        output_dim = feature_root * (2 ** i)
        if output_dim >= max_dim:
            output_dim = max_dim
        conv0 = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1, padding='same',
                                        name='enc_block_%d_conv_0' % i)
        conv1 = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=3, strides=1, padding='same',
                                        name='enc_block_%d_conv_1' % i)

        batchnorm0 = tf.keras.layers.BatchNormalization(name='enc_block_%d_batchnorm_0' % i)
        batchnorm1 = tf.keras.layers.BatchNormalization(name='enc_block_%d_batchnorm_1' % i)

        for single_empty, single_value in zip([enc_conv1, enc_conv2, enc_bn_1, enc_bn_2], [conv0, conv1, batchnorm0, batchnorm1]):
            single_empty.append(single_value)
    return enc_conv1, enc_conv2, enc_bn_1, enc_bn_2


def create_dec_layers(num_decoder_layer, feature_root, shortcut):
    dec_conv1, dec_conv2, dec_bn_1, dec_bn_2, dec_convtranspose = [], [], [], [], []
    if shortcut == True:
        num_act_dec_layer = num_decoder_layer - 2
    else:
        num_act_dec_layer = num_decoder_layer - 1
    for i in range(num_act_dec_layer + 1):
        out_dim = 2 ** (num_act_dec_layer - i) * feature_root // 2
        deconv_convtranspose = tf.keras.layers.Conv2DTranspose(filters=out_dim, kernel_size=(2, 2),
                                                               strides=(2, 2), padding='same',
                                                               name='dec_block_%d_deconv_convtranspose' % i)
        deconv_conv0 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, strides=1,
                                              padding='same', name='dec_block_%d_deconv_conv_0' % i)
        deconv_batch0 = tf.keras.layers.BatchNormalization(name='dec_block_%d_bn_0' % i)

        deconv_conv1 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, strides=1,
                                              padding='same', name='dec_block_%d_deconv_conv_1' % i)
        deconv_batch1 = tf.keras.layers.BatchNormalization(name='dec_block_%d_bn_1' % i)

        for single_empty, single_value in zip([dec_conv1, dec_conv2, dec_bn_1, dec_bn_2, dec_convtranspose],
                                              [deconv_conv0, deconv_conv1, deconv_batch0, deconv_batch1, deconv_convtranspose]):
            single_empty.append(single_value)

    return dec_conv1, dec_conv2, dec_bn_1, dec_bn_2, dec_convtranspose


def build_encoder(x_input, layer_group, model_type, motion_model, data_set, shortcut):
    num_frame = x_input.get_shape().as_list()[0]
    enc_conv1, enc_conv2, enc_batch_norm1, enc_batch_norm2 = layer_group

    num_encoder_layer = np.shape(enc_conv1)[0]
    feat_space_tot = []
    latent_space = []
    with tf.variable_scope('build_common_encoder'):
        for i in range(num_frame):
            single_x = x_input[i]
            feat_space_per_frame = []
            for j in range(num_encoder_layer):
                single_x = enc_conv1[j](single_x)
                single_x = enc_batch_norm1[j](single_x)
                single_x = tf.keras.layers.LeakyReLU()(single_x)
                single_x = enc_conv2[j](single_x)
                single_x = enc_batch_norm2[j](single_x)
                if j != num_encoder_layer - 1:
                    single_x = tf.keras.layers.LeakyReLU()(single_x)
                else:
                    if motion_model == "convlstm" or data_set == "avenue" or \
                                "moving_mnist" in data_set or model_type == "many_to_one":
                        single_x = tf.nn.tanh(single_x)
                    else:
                        single_x = tf.keras.layers.LeakyReLU()(single_x)
                feat_space_per_frame.append(single_x)

                if j < num_encoder_layer - 1:
                    single_x, single_x_max_pool_index = tf.nn.max_pool_with_argmax(single_x,
                                                                                   ksize=(1, 2, 2, 1),
                                                                                   strides=(1, 2, 2, 1),
                                                                                   padding='SAME',
                                                                                   name="block_%d_maxpool" % j)
                else:
                    if shortcut == False:
                        single_x, single_x_max_pool_index = tf.nn.max_pool_with_argmax(single_x,
                                                                                       ksize=(1, 2, 2, 1),
                                                                                       strides=(1, 2, 2, 1),
                                                                                       padding='SAME',
                                                                                       name="block_%d_maxpool" % j)    

            latent_space.append(single_x)
            feat_space_tot.append(feat_space_per_frame)
    return latent_space, feat_space_tot


def build_decoder_many_frames(latent_codes, feature_space, layer_group, output_layer, shortcut):
    dec_conv1, dec_conv2, dec_batchnorm1, dec_batchnorm2, dec_conv2dtranspose = layer_group
    num_dec_layer = np.shape(dec_conv1)[0]
    num_frame = np.shape(latent_codes)[0]
    with tf.variable_scope("build_common_decoder"):
        output_tot = []
        for j in range(num_frame):
            single_latent = latent_codes[j]
            if shortcut == True:
                single_feat_map = feature_space[j]
            for i in range(num_dec_layer):
                single_latent = dec_conv2dtranspose[i](single_latent)
                if shortcut == True:
                    single_latent = tf.concat([single_feat_map[num_dec_layer - 1 - i], single_latent], axis=-1)
                single_latent = dec_conv1[i](single_latent)
                single_latent = dec_batchnorm1[i](single_latent)
                single_latent = tf.keras.layers.LeakyReLU()(single_latent)
                single_latent = dec_conv2[i](single_latent)
                single_latent = dec_batchnorm2[i](single_latent)
                single_latent = tf.keras.layers.LeakyReLU()(single_latent)
            pred = output_layer(single_latent)
            output_tot.append(pred)
        output_tot = tf.concat([output_tot], axis=0)
    return output_tot


def build_decoder_single_frame(latent_codes, layer_group, output_layer):
    single_latent = latent_codes
    dec_conv1, dec_conv2, dec_batchnorm1, dec_batchnorm2, dec_conv2dtranspose = layer_group
    num_dec_layer = np.shape(dec_conv1)[0]
    with tf.variable_scope("build_common_decoder"):
        for i in range(num_dec_layer):
            single_latent = dec_conv2dtranspose[i](single_latent)

            single_latent = dec_conv1[i](single_latent)
            single_latent = dec_batchnorm1[i](single_latent)
            single_latent = tf.keras.layers.LeakyReLU()(single_latent)
            
            single_latent = dec_conv2[i](single_latent)
            single_latent = dec_batchnorm2[i](single_latent)
            single_latent = tf.keras.layers.LeakyReLU()(single_latent)
        pred = output_layer(single_latent)
    return pred


class DAML(object):
    def __init__(self, args):
        """This contains the AE-Unet, PureAE, and Seq2Seq model
        """
        self.num_encoder_layer = args.num_encoder_layer
        self.num_decoder_layer = args.num_decoder_layer
        self.batch_size = args.batch_size
        self.data_set = args.data_set
        self.motion_model = args.motion_method 
        self.model_type = args.model_type
        self.time_step = args.time_step 

        if self.model_type == "2d_2d_pure_unet":
            shortcut = True
        elif self.model_type == "2d_2d_unet_no_shortcut":
            shortcut = False
        elif self.model_type == "many_to_one":
            shortcut = False

        self.shortcut = shortcut

        feature_root = 64
        max_dim = 1024
        if "moving_mnist" in self.data_set:
            max_dim = 128
        self.enc_layer_group = create_enc_layers(self.num_encoder_layer, feature_root, max_dim)
        self.dec_layer_group = create_dec_layers(self.num_decoder_layer, feature_root, self.shortcut)

        self.learn_motion_arch = MotionModel(args)

        if "moving_mnist" in self.data_set:
            self.output_layer = tf.keras.layers.Conv2D(filters=args.output_dim, kernel_size=3, strides=1,
                                                       padding='same', name='final_output_layer',
                                                       activation=tf.math.sigmoid)
        else:
            self.output_layer = tf.keras.layers.Conv2D(filters=args.output_dim, kernel_size=3, strides=1,
                                                       padding='same', name='final_output_layer')
            
    def build_unet_fps(self, x_input):
        latent_space, _ = build_encoder(x_input, self.enc_layer_group, self.model_type, 
                                        self.motion_model, self.data_set, self.shortcut)
        fh, fw, f_ch = latent_space[0].get_shape().as_list()[1:]
        latent_space_for_motion_placeholder = tf.placeholder(tf.float32,
                                                             shape = [self.time_step, self.batch_size, fh, fw, f_ch],
                                                             name="latent_space_for_motion")
        #-----build motion model-------#
        latent_space_pred = self.learn_motion_arch.build_motion_latent(latent_space_for_motion_placeholder)
        latent_space_gt = tf.placeholder(tf.float32, shape = [self.batch_size, fh, fw, f_ch],
                                         name = "gt_latent_space")
        return latent_space, latent_space_for_motion_placeholder, latent_space_pred, latent_space_gt        

    def build_pure_unet(self, x_input):
        latent_space, feature_maps = build_encoder(x_input, self.enc_layer_group, self.model_type, self.motion_model, self.data_set, 
                                                   self.shortcut)

        latent_space_for_dec, feature_for_decoder = [], []
        for i in range(self.time_step):
            latent_space_for_dec.append(latent_space[i+1])
            feature_for_decoder.append(feature_maps[i])

        latent_space_for_motion = latent_space[:self.time_step]
        latent_space_gt = latent_space[-1]
        latent_space_pred = self.learn_motion_arch.build_motion_latent(latent_space_for_motion)
        latent_space_for_dec[-1] = latent_space_pred


        decoder_output = build_decoder_many_frames(latent_space_for_dec, feature_for_decoder, self.dec_layer_group, 
                                                   self.output_layer, self.shortcut)
        p_x_recons = decoder_output[:self.time_step - 1]
        p_x_pred = decoder_output[-1:]

        return p_x_recons, p_x_pred, latent_space_gt, latent_space_pred

    def build_pure_ae(self, x_input):
        latent_space, feature_maps = build_encoder(x_input, self.enc_layer_group, self.model_type, self.motion_model, self.data_set, 
                                                   self.shortcut)

        latent_space_for_dec = []
        for i in range(self.time_step + 1):
            latent_space_for_dec.append(latent_space[i])

        latent_space_for_motion = latent_space[:self.time_step]
        latent_space_gt = latent_space[-1]
        latent_space_pred = self.learn_motion_arch.build_motion_latent(latent_space_for_motion)
        latent_space_for_dec[-1] = latent_space_pred

        decoder_output = build_decoder_many_frames(latent_space_for_dec, [], self.dec_layer_group, 
                                                   self.output_layer, self.shortcut)

        p_x_recons = decoder_output[:self.time_step]
        p_x_pred = decoder_output[-1:]

        return p_x_recons, p_x_pred, latent_space_gt, latent_space_pred

    def build_seq2seq(self, x_input):
        latent_space, feature_maps = build_encoder(x_input, self.enc_layer_group, self.model_type, self.motion_model, self.data_set, 
                                                   self.shortcut)
        latent_space_for_motion = latent_space[:self.time_step]

        latent_space_pred = self.learn_motion_arch.build_motion_latent(latent_space_for_motion)

        p_x_pred = build_decoder_single_frame(latent_space_pred, self.dec_layer_group, self.output_layer)
        p_x_pred = tf.expand_dims(p_x_pred, axis=0)

        return [], p_x_pred, [], latent_space_pred

    def forward(self, x_input):
        if self.model_type == "2d_2d_pure_unet":
            return self.build_pure_unet(x_input)
        elif self.model_type == "2d_2d_unet_no_shortcut":
            return self.build_pure_ae(x_input)
        elif self.model_type == "many_to_one":
            return self.build_seq2seq(x_input)

        