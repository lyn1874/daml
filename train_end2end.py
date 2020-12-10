#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:23:08 2019
This scrip is for training the experiement end2end
@author: li
"""
import tensorflow as tf
import models.AE as AE
import optimization.loss_tf as loss_tf
from data import read_frame_temporal as rft
import numpy as np
import os
import math
import cv2
import shutil
import const 


def train_end2end(args, data_set, model_type, motion_method, version=0, bg_ind=None, augment_opt="none"):
    model_mom_for_load_data = args.datadir
    path_mom = args.expdir
    if data_set == "ucsd1":
        stat = [8,6,2,5]
        train_ucsd1(stat, model_type, motion_method, version)
    elif data_set == "ucsd2":
        stat = [8,6,2,4]
        train_ucsd2(stat, model_type, motion_method, version)
    elif data_set == "avenue":
        stat = [6,6,2,4]
        train_avenue(stat, model_type, augment_opt, version)
    elif data_set == "shanghaitech_allinone":
        stat = [6,6,2,4]
        train_shanghaitech_allinone(stat, model_type, version)
    elif data_set == "shanghaitech_multiple":
        stat = [6,6,2,4]
        train_shanghaitech_multiple(stat, model_type, motion_method,
                                    version, bg_ind)
    # elif data_set is "moving_mnist":
    #     # 6, 6, 1, 4
    #     train_moving_mnist(model_mom_for_load_data, path_mom, stat, model_type, version)


def train_fps(model_mom_for_load_data, path_mom):
    # 31,32,33,34
    version = 0
    interval_group = np.arange(11)[1:] * 2
    learn_opt = "learn_fore"
    data_set = "ucsd2"
    motion_method = "conv3d"
    model_type = "2d_2d_pure_unet"
    time_step = 6
    args.z_mse_ratio = 0.001
    for single_interval in interval_group:
        delta = single_interval
        train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type, motion_method,
                    single_interval, version,
                    None, 4, learn_opt)


def train_ucsd1_group():
    stat = [8, 6, 2, 5]
    model_type = "2d_2d_pure_unet"
    motion_method = "convlstm"
    version = [0, 1, 2, 3]
    for single_version in version:
        train_ucsd1(stat, model_type, motion_method, single_version)


def train_ucsd1(stat, model_type, motion_method, version):
    data_set = "ucsd1"
    time_step, delta, interval, num_enc_layer = stat
    train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type, motion_method,
                interval, version,
                None,
                num_enc_layer, "learn_fore")


def train_ucsd2_group():
    stat = [8, 6, 2, 4]
    model_type = "2d_2d_pure_unet"
    motion_method = "convlstm"
    for single_version in [2, 3]:
        train_ucsd2(stat, model_type, motion_method, single_version)


def train_ucsd2(stat, model_type, motion_method, version):
    data_set = "ucsd2"
    time_step, delta, interval, num_enc_layer = stat
    train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type, motion_method, interval,
                version, None, num_enc_layer, "learn_fore")

    
def train_avenue_group():
    data_dir = args.datadir
    model_dir = args.expdir
    stat = [6, 6, 2, 4]
    motion_method = "conv3d"
    augment_opt = "none"
    for single_version in [2, 3]:
        train_avenue(data_dir, model_dir, stat, "2d_2d_pure_unet", motion_method,
                     augment_opt, single_version)
        

def train_avenue(stat, model_type, motion_method, augment_opt, version):
    data_set = "avenue"
    args.augment_option = augment_opt
    if augment_opt == "add_dark_auto":
        learn_opt = "learn_full"
    else:
        learn_opt = "learn_fore"
    time_step, delta, interval, num_enc_layer = stat
    train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type,  motion_method,
                interval, version,
                None,
                num_enc_layer, learn_opt)


def train_shanghaitech_allinone(stat, model_type, version):
    motion_method = "conv3d"
    time_step, delta, interval, num_enc_layer = stat
    data_set = "shanghaitech"
    train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type,
                motion_method, interval, version, None, num_enc_layer, "learn_fore")


def train_shanghaitech_multiple(stat, model_type, motion_method, version,
                                bg_ind=None):
    if bg_ind[0] == 0:
        bg_ind = [2, 3, 7, 9, 11]
    for single_bg_ind in bg_ind:
        train_shanghaitech_for_per_bg(args.datadir, args.expdir, stat, model_type, motion_method,
                                      single_bg_ind, version)


def train_shanghaitech_for_per_bg(model_mom_for_load_data, path_mom, stat, model_type, motion_method, bg_ind, version):
    time_step, delta, interval, num_enc_layer = stat
    data_set = "shanghaitech"
    train_model(model_mom_for_load_data, path_mom, data_set, time_step, delta, model_type,
                motion_method, interval, version, None, num_enc_layer, "learn_fore",
                bg_index_pool=[bg_ind])


def train_moving_mnist():
    motion_method = "conv3d"
    data_set = "moving_mnist"
    version = 2
    model_type = "2d_2d_unet_no_shortcut"
    z_mse_ratio = 0.001
    args.z_mse_ratio = z_mse_ratio
    num_layer = [5]
    stat_group = [[6, 2, 1]]
    for single_layer in num_layer:
        for single_stat in stat_group:
            time_step, delta, interval = single_stat
            num_enc_layer = single_layer
            train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type,
                        motion_method, interval, version, None, num_enc_layer, "learn_full")


def train_moving_mnist_single_digit(model_group):
    """This function train a pure autoencoder for moving mnist single digit dataset
    The goal of this type of experiments is to hope the latent can show some pattern between
    anomalies and normal"""
    motion_method = "conv3d"
    data_set = "moving_mnist_single_digit"
    version = 1  # version 1 means the activation layer in the last convolutional block is changed from
    # learky-relu to tanh 
    args.z_mse_ratio = 0.001
    num_layer = [5, 4]
    stat = [6, 2, 1]
    for model_type in model_group:
        for single_layer in num_layer:
            time_step, delta, interval = stat
            num_enc_layer = single_layer
            train_model(args.datadir, args.expdir, data_set, time_step, delta, model_type,
                        motion_method, interval, version, None, num_enc_layer, "learn_full")


def train_seq2seq(version):
    data_set = "ucsd2"
    motion_method = "conv3d"
    model_type = "many_to_one"
    for time_step in [4, 6, 8]:
        stat = [time_step, 2, 2, 4]
        train_model(args.datadir, args.expdir, data_set, stat[0], stat[1], model_type, 
            motion_method, stat[2], version, None, stat[-1], "learn_fore", None)



def train_model(model_mom_for_load_data, path_mom, data_set, time_step, delta, model_type, motion_method,
                single_interval, version, ckpt_dir, num_enc_layer, learn_opt, bg_index_pool=None):
    print("-------------------Start to train the model------------------------------")
    args.data_set = data_set
    interval_input = np.array([single_interval])
    bg_index = None
    args.num_encoder_layer = num_enc_layer
    args.num_decoder_layer = num_enc_layer
    args.time_step = time_step
    args.single_interval = single_interval
    args.delta = delta 
    args.learn_opt = learn_opt
    args.bg_index_pool = bg_index_pool
    model_dir = path_mom + "ano_%s_motion_end2end/" % args.data_set
    if not bg_index_pool:
        model_dir = model_dir + "time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_version_%d" % (time_step,
            delta, single_interval, model_type, motion_method, learn_opt, num_enc_layer, version)
    else:
        model_dir = model_dir + "time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_bg_%d_version_%d" % (
            time_step,
            delta, single_interval, model_type, motion_method,
            learn_opt, num_enc_layer, bg_index_pool[0], version)

    tmf = TrainMainFunc(args, model_mom_for_load_data, model_dir, ckpt_dir, time_step, interval_input, delta,
                        train_index=bg_index,
                        bg_index_pool=bg_index_pool)
    tmf.build_running()


def read_data(model_mom, data_set, concat_option, time_step, interval_input, delta, bg_index_pool=None):
    if data_set != "shanghaitech":
        train_im, test_im, imshape, targ_shape = rft.get_video_data(model_mom, data_set).forward()
        train_im_interval, in_shape, out_shape = rft.read_frame_interval_by_dataset(data_set, train_im,
                                                                                    time_step, concat_option,
                                                                                    interval_input, delta)
    else:
        train_im_group = []
        if not bg_index_pool:
            bg_index_pool = np.arange(13)[1:]

        for single_bg_index in bg_index_pool:
            if single_bg_index < 10:
                bg_index = "bg_index_0%d" % single_bg_index
            else:
                bg_index = "bg_index_%d" % single_bg_index
            print("--------loading data from bg %s---------------" % bg_index)
            test_im, test_la, imshape, targ_shape = rft.get_video_data(model_mom, args.data_set).forward(bg_index)

            test_im_interval, in_shape, out_shape = rft.read_frame_interval_by_dataset(data_set, test_im,
                                                                                       time_step, concat_option,
                                                                                       interval=interval_input,
                                                                                       delta=delta)
            train_im_group.append(test_im_interval)
        train_im_interval = np.array([v for j in train_im_group for v in j])
    return train_im_interval, imshape, targ_shape, in_shape, out_shape


class TrainMainFunc(object):
    def __init__(self, args, model_mom, model_dir, ckpt_dir, time_step, interval_input=np.array([1]), delta=None,
                 train_index=None, bg_index_pool=None):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        concat_option = "conc_tr"
        train_im_interval, imshape, targ_shape, in_shape, out_shape = read_data(model_mom, args.data_set,
                                                                                concat_option, time_step,
                                                                                interval_input, delta,
                                                                                bg_index_pool=bg_index_pool)
        args.output_dim = targ_shape[-1]
        if concat_option == "conc_tr":
            args.num_prediction = 1
        else:
            args.num_prediction = out_shape[0]

        self.args = args

        self.model_mom = model_mom
        self.model_dir = model_dir
        self.ckpt_dir = ckpt_dir

        self.data_set = args.data_set
        self.train_index = train_index
        self.temp_shape = [in_shape, out_shape]
        self.targ_shape = targ_shape
        self.imshape = imshape
        self.output_dim = args.output_dim
        self.concat = "conc_tr"
        self.time_step = time_step
        self.delta = delta
        self.interval = interval_input[0]
        self.test_im = train_im_interval
        self.input_option = args.input_option
        self.augment_option = args.augment_option
        self.darker_value = args.darker_value

        self.learn_opt = args.learn_opt
        self.model_type = args.model_type
        self.z_mse_ratio = args.z_mse_ratio

        [lrate_g_step, lrate_g], [lrate_z_step, lrate_z], [epoch, batch_size] = const.give_learning_rate_for_init_exp(self.args)

        self.lrate_g_decay_step = lrate_g_step
        self.lrate_g_init = lrate_g
        self.lrate_z_decay_step = lrate_z_step
        self.lrate_z_init = lrate_z
        self.batch_size = batch_size
        self.max_epoch = epoch

        print(args)

    def read_tensor(self):
        imh, imw, ch = self.targ_shape
        placeholder_shape = [None, 2, self.temp_shape[0][0]]
        shuffle_option = True
        if "/project/" in self.model_dir:
            repeat = 20
        else:
            repeat = 1
        images_in = tf.placeholder(tf.string, shape=placeholder_shape, name='tr_im_path')
        image_queue = rft.dataset_input(self.model_mom, self.data_set, images_in, self.learn_opt,
                                        self.temp_shape, self.imshape, self.targ_shape[:2], self.batch_size,
                                        augment_option=self.augment_option,
                                        darker_value=self.darker_value,
                                        conc_option=self.concat, shuffle=shuffle_option,
                                        train_index=None,
                                        epoch_size=repeat)
        image_init = image_queue.make_initializable_iterator()
        image_batch = image_init.get_next()
        x_input = image_batch[0]  # [batch_size, num_input_channel, imh, imw, ch]
        x_output = image_batch[1]  # [batch_size, self.output_dim, imh, imw, ch]
        im_background = image_batch[-1]
        print("=========================================")
        print("The input of the model", x_input)
        print("The output of the model", x_output)
        print("The background of the data", im_background)
        print("=========================================")

        x_input = tf.concat([x_input, x_output], axis=1)  # th==already subtract the background.

        if self.learn_opt == "learn_fore":
            x_real_input = x_input + im_background
        else:
            x_real_input = x_input
        self.x_real_input = tf.transpose(x_real_input, perm=(1, 0, 2, 3, 4))
        x_input = tf.transpose(x_input, perm=(1, 0, 2, 3, 4))  # num_frame, batch_size, imh, imw, ch
        # the last input of x_input is for prediction
        im_background = tf.transpose(im_background, perm=(1, 0, 2, 3, 4))  # num_frame, batch_size, imh, imw, ch
        if "crop" in self.input_option:
            x_input = tf.reshape(x_input, shape=[(self.time_step + 1) * self.batch_size, imh, imw, ch])
            crop_size = self.input_option.strip().split("crop_")[1]
            crop_h, crop_w = crop_size.strip().split("_")
            crop_h, crop_w = int(crop_h), int(crop_w)
            x_input_crop, stride_size, crop_box_h_w = rft.get_crop_image(x_input, crop_h, crop_w)
            x_input_crop = tf.concat([x_input_crop],
                                     axis=0)  # [num_regions, (num_time+1)*batch_size, crop_height, crop_weight,ch]
            num_box = x_input_crop.get_shape().as_list()[0]
            x_input_crop = tf.reshape(x_input_crop,
                                      shape=[num_box, self.time_step + 1, self.batch_size, crop_h, crop_w, ch])
            x_input_crop = tf.transpose(x_input_crop, perm=(1, 0, 2, 3, 4, 5))
            x_input_crop = tf.reshape(x_input_crop,
                                      shape=[self.time_step + 1, num_box * self.batch_size, crop_h, crop_w, ch])
            x_input = x_input_crop  # [time, num_box*batch, croph, cropw, ch]
            x_input = tf.transpose(x_input, perm=(1, 0, 2, 3, 4))  # [batch, time, c_h, c_w, ch]
            x_input = tf.random.shuffle(x_input)
            if crop_h >= 128:
                x_input = x_input[:4]  # this is for batch size
            print("The actual number of box", num_box)
            x_input = tf.transpose(x_input, perm=(1, 0, 2, 3, 4))  # [time, batch, c_h, c_w, ch]
            self.x_real_input = x_input
        return images_in, x_input, image_init, im_background

    def build_graph(self):
        num_recons_output = self.time_step
        image_placeholder, x_input, image_init, im_background = self.read_tensor()

        # --build encoder-------------#
        model_use = AE.DAML(self.args)
        p_x_recons, p_x_pred, latent_space_gt, latent_space_pred = model_use.forward(x_input)


        if "crop" not in self.input_option:
            if self.learn_opt == "learn_full":
                print("====the reconstruction is full frame=============")
            elif self.learn_opt == "learn_fore":
                print("====the reconstruction is frame - background=====")
                if self.model_type != "many_to_one":
                    p_x_recons = p_x_recons + im_background
                p_x_pred = p_x_pred + im_background

        if self.model_type == "2d_2d_pure_unet":
            x_recons_gt = self.x_real_input[1:self.time_step]  # [num_recons, batch_size, imh, imw, ch]
        elif self.model_type == "2d_2d_unet_no_shortcut":
            x_recons_gt = self.x_real_input[:self.time_step]
        else:
            x_recons_gt = []
        x_pred_gt = self.x_real_input[-1:]

        print("=============================================================")
        print("----the input for the model-----------------", x_input)
        print("----the groundtruth for reconstruction------", x_recons_gt)
        print("----the reconstructed frames----------------", p_x_recons)
        print("----the groundtruth for prediction----------", x_pred_gt)
        print("----the predicted frame---------------------", p_x_pred)
        print("----the gt latent space---------------------", latent_space_gt)
        print("----the predicted latent space--------------", latent_space_pred)
        print("=============================================================")


        if self.model_type== "2d_2d_pure_unet" or self.model_type== "2d_2d_unet_no_shortcut":
            if "moving_mnist" not in self.data_set:
                mse_pixel = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_recons_gt, p_x_recons), (-1, -2, -3)))
            else:
                mse_pixel = tf.keras.losses.binary_crossentropy(y_true=x_recons_gt, y_pred=p_x_recons,
                                                                from_logits=False)
                mse_pixel = tf.reduce_mean(tf.reduce_sum(mse_pixel, (-1, -2, -3)))
            mse_latent = tf.reduce_mean(
                tf.reduce_sum(tf.squared_difference(latent_space_gt, latent_space_pred), (-1, -2, -3)))
        elif self.model_type== "many_to_one":
            mse_pixel = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_pred_gt-im_background, p_x_pred-im_background), (-1, -2, -3)))
            mse_latent = tf.constant(0.0)

        z_mse_ratio_placeholder = tf.placeholder(tf.float32, name="ratio_for_z_mse")
        if self.model_type != "many_to_one":
            loss_tot = mse_pixel + mse_latent * z_mse_ratio_placeholder
        else:
            loss_tot = mse_pixel
        var_tot = tf.trainable_variables()

        [print(v) for v in var_tot if 'kernel' in v.name]
        #        print("==========================================")
        #        print("encoder decoder trainable variables")
        #        [print(v) for v in var_tot if 'motion_latent' not in v.name]
        #        print("==========================================")
        #        print("motion trainable variables")
        #        [print(v) for v in var_tot if 'motion_latent' in v.name]

        var_0 = var_tot
        loss_tot = tf.add_n([loss_tot, tf.add_n(
            [tf.nn.l2_loss(v) for v in var_0 if 'kernel' in v.name or 'weight' in v.name]) * args.regu_par])
        g_lrate = tf.placeholder(tf.float32, name='g_lrate')
        train_op_0 = loss_tf.train_op(loss_tot, g_lrate, var_opt=var_0, name='train_op_tot')

        z_lrate = tf.placeholder(tf.float32, name='z_lrate')
        if self.model_type != "many_to_one":
            var_motion = [v for v in var_tot if 'motion_latent' in v.name]
            loss_motion = mse_latent
            loss_motion = tf.add_n([loss_motion, tf.add_n(
                [tf.nn.l2_loss(v) for v in var_motion if 'kernel' in v.name or 'weight' in v.name]) * args.regu_par])
            train_op_z = loss_tf.train_op(loss_motion, z_lrate, var_opt=var_motion, name='train_latent_z')
            train_z_group = [z_lrate, train_op_z]
        else:
            train_z_group = [z_lrate, []]

        saver_set_all = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

        input_group = [image_init, image_placeholder, z_mse_ratio_placeholder]
        loss_group = [mse_pixel, mse_latent, loss_tot]
        
        train_group = [g_lrate, train_op_0, saver_set_all]
        if self.model_type== "2d_2d_pure_unet" or self.model_type== "2d_2d_unet_no_shortcut":
            im_stat = [p_x_recons, x_recons_gt, p_x_pred, x_pred_gt]
        else:
            im_stat = [p_x_pred, x_pred_gt]
        return input_group, loss_group, train_group, train_z_group, im_stat

    def build_train_op(self, sess, image_init, placeholder_group, 
                       x_train, single_epoch, num_epoch_for_full, loss_group, train_op_group):
        train_op_0, train_op_z = train_op_group
        image_placeholder, z_mse_placeholder, g_lrate_placeholder, z_lrate_placeholder = placeholder_group

        sess.run(image_init.initializer, feed_dict={image_placeholder: x_train})
        num_tr_iter_per_epoch = np.shape(x_train)[0] // self.batch_size
        
        lrate_g_npy = self.lrate_g_init * math.pow(0.1, math.floor(float(single_epoch) / float(self.lrate_g_decay_step)))
        lrate_z_npy = self.lrate_z_init * math.pow(0.1, math.floor(float(single_epoch - num_epoch_for_full) / float(self.lrate_z_decay_step)))
        
        loss_per_epoch = []
        if single_epoch <= num_epoch_for_full:
            fetches_tr = [train_op_0]
        else:
            fetches_tr = [train_op_z]
        fetches_tr.append(loss_group)

        for single_iter in range(num_tr_iter_per_epoch):

            _, _loss_group = sess.run(fetches=fetches_tr, feed_dict={z_mse_placeholder: self.z_mse_ratio,
                                                                     g_lrate_placeholder: lrate_g_npy,
                                                                     z_lrate_placeholder: lrate_z_npy})
            loss_per_epoch.append(_loss_group)

        return np.mean(loss_per_epoch, axis=0)

    def build_val_op(self, sess, image_init, image_placeholder, x_val, loss_group, image_stat, image_path, single_epoch):
        sess.run(image_init.initializer, feed_dict={image_placeholder: x_val})
        num_val_iter_per_epoch = np.shape(x_val)[0] // self.batch_size
        # image_stat: [p_x_recons, p_x_pred, x_recons_gt, x_pred_gt]
        # or
        # image_stat: [p_x_pred, x_pred_gt]
        loss_val_per_epoch = []
        for single_val_iter in range(num_val_iter_per_epoch):
            if single_val_iter != num_val_iter_per_epoch - 1:
                _loss_val = sess.run(fetches=loss_group)
            else:
                _loss_val, _stat_use = sess.run(fetches=[loss_group, image_stat])
                for single_input, single_path in zip(_stat_use, image_path):
                    for j in range(np.shape(single_input)[0]):
                        im_use = single_input[j, :]
                        shape_use = np.array(np.shape(im_use)[1:])
                        cv2.imwrite(os.path.join(single_path, "epoch_%d_frame_%d.jpg" % (single_epoch, j)), 
                                    (plot_canvas(im_use, shape_use)).astype('uint8')[:, :, ::-1])
            loss_val_per_epoch.append(_loss_val)
        return  np.mean(loss_val_per_epoch, axis=0)

    def build_running(self):

        im_path = os.path.join(self.model_dir, 'recons_gt')
        recons_path = os.path.join(self.model_dir, 'p_x_recons')
        im_pred_path = os.path.join(self.model_dir, 'pred_gt')
        pred_path = os.path.join(self.model_dir, 'p_x_pred')
        if self.model_type== "2d_2d_pure_unet" or self.model_type== "2d_2d_unet_no_shortcut":
            path_group = [recons_path, im_path, pred_path, im_pred_path]
        else:
            path_group = [pred_path, im_pred_path]
        for i in path_group:
            if not os.path.exists(i):
                os.makedirs(i)

        with tf.Graph().as_default():
            input_group, loss_group, train_group, train_z_group, im_stat = self.build_graph()
            image_init, image_placeholder, z_mse_ratio_placeholder = input_group
            mse_pixel_loss, mse_latent_loss, mse_tot = loss_group
            g_lrate, train_op, saver = train_group

            #z_lrate, train_z_op = train_z_group
            saver_restore = None
            tot_num_frame = np.shape(self.test_im)[0]
            test_im_shuffle = self.test_im[np.random.choice(np.arange(tot_num_frame),
                                                            tot_num_frame,
                                                            replace=False)]

            placeholder_group = [image_placeholder, z_mse_ratio_placeholder, g_lrate, train_z_group[0]]
            loss_group = [mse_pixel_loss, mse_latent_loss]
            train_group = [train_op, train_z_group[-1]]


            if "ucsd" in self.data_set:
                x_train = test_im_shuffle[:-self.batch_size * 4]
                x_val = test_im_shuffle[-self.batch_size * 4:]
            elif "avenue" in self.data_set or "shanghaitech" in self.data_set:
                x_train = test_im_shuffle[:-self.batch_size * 20]
                x_val = test_im_shuffle[-self.batch_size * 20:]
            else:
                x_train = test_im_shuffle[:-self.batch_size * 2]
                x_val = test_im_shuffle[-self.batch_size * 2:]

            if self.data_set== "ucsd1" and self.model_type != "many_to_one":
                num_epoch_for_full = 25
            else:
                num_epoch_for_full = self.lrate_g_decay_step
            checkpoint_path = self.model_dir + '/model.ckpt'
            print("====================================================================================")
            print("There are %d frames in total" % np.shape(self.test_im)[0])
            print("The shape of training and validation images", np.shape(x_train), np.shape(x_val))
            print(
                "%d input frames are loaded with %d stride for predicting furture frame at time t+%d" % (self.time_step,
                                                                                                         self.interval,
                                                                                                         self.delta))
            print("The lr for whole process start from %.4f and decay 0.1 every %d epoch" % (
                self.lrate_g_init, self.lrate_g_decay_step))
            print("The lr for motion process start from %.4f and decay 0.1 every %d epoch" % (
                self.lrate_z_init, self.lrate_z_decay_step))
            print("The ratio for the latent space mse loss== ", self.z_mse_ratio)
            print("The used background index is:", self.train_index)
            print("I am only focusing on the reconstruction for the first %d epochs" % num_epoch_for_full)
            print("====================================================================================")

            with tf.Session() as sess:
                if self.ckpt_dir== None:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                else:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    saver_restore.restore(sess, self.ckpt_dir)
                    print("restore parameter from ", self.ckpt_dir)

                loss_tr_tot = np.zeros([self.max_epoch, 2])
                loss_val_tot = []
                try:
                    for single_epoch in range(self.max_epoch):
                        loss_per_epoch = self.build_train_op(sess, image_init, placeholder_group, x_train, 
                                                             single_epoch, num_epoch_for_full, loss_group, 
                                                             train_group)
                        loss_tr_tot[single_epoch, :] = loss_per_epoch
                        print("Epoch %d with training pixel mse loss %.3f z mse %.3f" % (single_epoch,
                                                                                         loss_tr_tot[single_epoch, 0],
                                                                                         loss_tr_tot[single_epoch, 1]))
                        if single_epoch % 5 == 0 or single_epoch == self.max_epoch - 1:
                            # sess, image_init, image_placeholder, x_val, loss_group, image_stat, image_path, single_epoch)

                            loss_val_per_epoch = self.build_val_op(sess, image_init, image_placeholder, x_val, loss_group, im_stat, 
                                                                   path_group, single_epoch)
                            loss_val_tot.append(loss_val_per_epoch)
                            print("Epoch %d with validation pixel mse loss %.3f z mse %.3f" % (single_epoch,
                                                                                               loss_val_tot[-1][0],
                                                                                               loss_val_tot[-1][1]))
                        if np.isnan(loss_tr_tot[single_epoch, 0]):
                            np.save(self.model_dir + '/tr_loss', loss_tr_tot)
                            np.save(self.model_dir + '/val_loss', np.array(loss_val_tot))
                        if single_epoch % 5 == 0 and single_epoch != 0:
                            np.save(self.model_dir + '/tr_loss', loss_tr_tot)
                            np.save(self.model_dir + '/val_loss', np.array(loss_val_tot))
                            saver.save(sess, checkpoint_path, global_step=single_epoch)
                        if single_epoch == self.max_epoch - 1:
                            saver.save(sess, checkpoint_path, global_step=single_epoch)
                            np.save(self.model_dir + '/tr_loss', loss_tr_tot)
                            np.save(self.model_dir + '/val_loss', np.array(loss_val_tot))

                except tf.errors.OutOfRangeError:
                    print("---oh my god, my model again could't read the data----")
                    print("I am at step", single_iter, single_iter // num_tr_iter_per_epoch)
                    np.save(os.path.join(self.model_dir, 'tr_loss'), loss_tr_tot)
                    np.save(os.path.join(self.model_dir, 'val_loss'), np.array(loss_val_tot))
                    saver.save(sess, checkpoint_path, global_step=single_epoch)
                    pass


def plot_canvas(image, imshape, ny=8):
    if np.shape(image)[0] < ny:
        ny = np.shape(image)[0]
    nx = np.shape(image)[0] // ny
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    targ_height, targ_width = imshape[0], imshape[1]
    if np.shape(image)[-1] == 1:
        image = np.repeat(image, 3, -1)
    imshape[-1] = 3
    canvas = np.empty((targ_height * nx, targ_width * ny, 3))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            canvas[(nx - i - 1) * targ_height:(nx - i) * targ_height,
                   j * targ_width:(j + 1) * targ_width, :] = np.reshape(image[i * ny + j], imshape)
    return (canvas * 255.0).astype('uint8')


if __name__ == '__main__':
    args = const.args
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("-------------------------------------------------------------------")
    print(type(args.version), args.version)
    if args.version == 0:
        print("only running experiment once")
        train_end2end(args, args.data_set, args.model_type, 
                      args.motion_method, version=args.version, bg_ind=None, augment_opt="none")
    else:
        for s_version in range(args.version):
            print("running experiment for version %d" % s_version)
            train_end2end(args, args.data_set, args.model_type, 
                          args.motion_method, version=s_version, bg_ind=None, augment_opt="none")
            
    
