#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:23:08 2019
This scrip is for training the experiement end2end
@author: li
"""
import tensorflow as tf
import models.AE as AE
from data import read_frame_temporal as rft
import numpy as np
import os
import evaluate as ev
import cv2
import optimization.loss_tf as loss_tf
# import visualize_video as vv
import const


def get_conf_interval(data_set, stat, version, model_type="2d_2d_pure_unet", motion_method="conv3d"):
    auc_score_tot = []
    error_tot = []
    if motion_method is "convlstm":
        args.sliding_window = 163
        args.eval_method = "future"
    for iterr, single_version in enumerate(version):
        auc_score, error = test_end2end_model(data_set, model_type, motion_method, single_version,
                                              opt="eval_score")
        auc_score_tot.append(auc_score)
        error_tot.append(error)
    legend_tot = ["auc score", "model error"]
    auc_score_tot = np.array(auc_score_tot)
    error_tot = np.array(error_tot)
    for iterr, single_score in enumerate([auc_score_tot, error_tot]):
        print("-------------showing %s----------------" % legend_tot[iterr])
        avg = np.mean(single_score, axis=0)
        std = np.std(single_score, axis=0)
        conf = std * 1.95 / np.sqrt(np.shape(version)[0])
        if iterr == 0 and model_type is "2d_2d_pure_unet":
            label = ["r-mse", "z-mse", "z-cos", "z-l1", "p-mse"]
        elif iterr == 0 and model_type is "2d_2d_unet_no_shortcut":
            label = ["z-mse", "z-cos", "z-l1", "p-mse"]
        elif iterr == 1 and model_type is "2d_2d_pure_unet":
            label = ["recons", "z-mse", "z-cos", "p-mse"]
        elif iterr == 1 and model_type is "2d_2d_unet_no_shortcut":
            label = ["z-mse", "z-cos", "p-mse"]
        [print(np.round(v * 100, 4)) for v in single_score]
        print("----------------------------------")
        [print(label[i] + ' %.2f +- %.2f' % (avg[i] * 100, conf[i] * 100)) for i in range(np.shape(label)[0])]
        print("----------------------------------")
    

def test_end2end_model(args, data_set, model_type="2d_2d_pure_unet", motion_method="conv3d",
                       version=0, opt="save_score"):
    path_for_load_data = args.datadir
    model_mom = args.expdir
    args.model_type = model_type
    args.motion_model = motion_method
    if data_set == "ucsd1":
        stat = [8, 6, 2, 5]
    elif data_set == "ucsd2":
        stat = [8, 6, 2, 4]
    elif data_set == "avenue":
        stat = [6, 6, 2, 4]
    if data_set == "shanghaitech_allinone":
        stat = [6, 6, 2, 4]
        allinone = True
    elif data_set == "shanghaitech_multiple":
        stat = [6, 6, 2, 4]
        allinone = False
    if opt != "eval_score":
        if "shanghaitech" not in data_set:
            test_except_shanghaitech(path_for_load_data, model_mom, data_set, stat[0], stat[1], stat[2],
                                     model_type, motion_method, stat[3], version=version)
        else:
            test_shanghaitech(path_for_load_data, model_mom, stat[0], stat[1], stat[2], model_type, motion_method,
                              stat[3], version=version, allinone=allinone)
            #print("-----------------Giving the AUC score------------------------------")
    if "shanghaitech" in data_set:
        data_set = "shanghaitech"
    print("reporting the anomaly detection accuracy on dataset %s" % data_set)
    auc_score, error = test_model(path_for_load_data, model_mom, data_set, stat[0], stat[1], stat[2], model_type,
                                  motion_method, version, stat[3], "learn_fore", "test", opt="eval_score")


def test_except_shanghaitech(path_for_load_data, model_mom, data_set, 
                             time_step, delta, single_interval, model_type,
                             motion_method, num_layer=4, learn_opt="learn_fore", version=0, opt="save_score"):
    test_index_all, gt = ev.read_test_index(path_for_load_data, data_set)
    for single_ind in test_index_all:
        test_model(path_for_load_data, model_mom, data_set, time_step, delta, single_interval,
                   model_type, motion_method, version,
                   num_layer, learn_opt,
                   single_ind, opt)


def test_shanghaitech(path_for_load_data, model_mom, time_step, delta, single_interval, 
                      model_type, motion_method, num_layer, learn_opt="learn_fore", version=0, allinone=False):
    bg_index_all = []
    bg_ind_use = [8]
    #    bg_ind_use = [4,10]
    for single_bg in bg_ind_use:
        if single_bg < 10:
            bg_index_all.append('bg_index_0%d' % single_bg)
        else:
            bg_index_all.append('bg_index_%d' % single_bg)
    test_index_all, gt = ev.read_test_index(path_for_load_data, "shanghaitech")
    for bg_iterr, single_bg in enumerate(bg_index_all):
        test_index_subset = [v.strip().split('/')[0] for v in test_index_all if
                             single_bg.strip().split("index_")[1] + '_' in v]
        if allinone is True:
            select_bg_index = None
        else:
            select_bg_index = int(single_bg.strip().split('index_')[1])
        for single_test_index in test_index_subset:
            test_model(path_for_load_data, model_mom, "shanghaitech", time_step, delta, single_interval,
                       model_type, motion_method, version,
                       num_layer, learn_opt,
                       single_test_index, "save_score", bg_index=single_bg, select_bg_index=select_bg_index)


def test_model(model_mom_for_load_data, path_mom, data_set, time_step, delta, single_interval, 
               model_type, motion_method, version, num_enc_layer, learn_opt, 
               test_index_use, opt, z_mse_ratio=0.001, bg_index=None,
               select_bg_index=None, small_or_bg=None):
    args.data_set = data_set
    args.num_encoder_layer = num_enc_layer
    args.num_decoder_layer = num_enc_layer
    args.model_type = model_type
    args.learn_opt = learn_opt
    args.time_step = time_step
    args.motion_model = motion_method
    if not select_bg_index:
        model_dir = path_mom + "/ano_%s_motion_end2end/time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_version_%d" % (
            args.data_set, time_step,
            delta, single_interval,
            model_type, motion_method,
            learn_opt, num_enc_layer,
            version)
        tds_dir = model_dir.strip().split('/time_')[0] + '/tds/' + model_dir.strip().split('end2end/')[1]
        if small_or_bg:
            tds_dir = tds_dir + '/%s/' % small_or_bg
    else:
        path_mom = path_mom + "/cvpr_exp/ano_%s_motion_end2end/" % args.dataset
        model_dir = path_mom + "time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_bg_%d_version_%d" % (
            time_step,
            delta,
            single_interval,
            model_type, motion_method,
            learn_opt, num_enc_layer,
            select_bg_index,
            version)
        if "groups" in model_dir:
            tds_dir = "/media/data/cvpr_exp/"
            tds_dir = tds_dir + "ano_%s_motion_end2end/tds/time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_version_%d" % (
                args.data_set, time_step, delta, single_interval, model_type, motion_method, learn_opt,
                num_enc_layer, version)
        else:
            tds_dir = model_dir.strip().split('/time_')[
                          0] + '/tds/' + 'time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_version_%d' % (time_step,
                                                                                                 delta, single_interval,
                                                                                                 model_type,
                                                                                                 motion_method,
                                                                                                 learn_opt,
                                                                                                 num_enc_layer,
                                                                                                 version)
    if opt == "return_tds":
        return tds_dir

    if opt == "eval_score":
        if args.data_set == "shanghaitech":
            auc_score, error = ev.get_auc_score_end2end(tds_dir, model_mom_for_load_data,
                                                        data_set, single_interval, show=False, aug=None,
                                                        N=args.sliding_window,
                                                        method=args.eval_method)
        else:
            auc_score, error = ev.get_auc_score_end2end(tds_dir, model_mom_for_load_data,
                                                        data_set, single_interval, show=True, aug=None,
                                                        N=args.sliding_window,
                                                        method=args.eval_method)
        return auc_score, error
    else:
        if not os.path.isfile(tds_dir + "/pred_score_%s.npy" % test_index_use):
            tmf = TestMainFunc(args, model_mom_for_load_data, model_dir, tds_dir, time_step, single_interval,
                               test_index_use, delta, train_index=bg_index)
            if "save_figure" in opt:
                save_figure = True
            else:
                save_figure = False

            if "check_pred" in opt:
                tmf.check_prediction(save_figure)
            elif "check_recons" in opt:
                tmf.check_reconstruction(save_figure)
            elif "save_score" in opt and "moving_mnist" not in args.data_set and model_type is not "many_to_one":
                tmf.check_all()
            elif "save_score" in opt and model_type is "many_to_one":
                tmf.save_score_many_to_one()
#         elif "save_score" in opt and "moving_mnist" in args.data_set:
#             tmf.save_score_mnist()


class TestMainFunc(object):
    def __init__(self, args, model_mom, model_dir, tds_dir, 
                 time_step, single_interval, test_index_use, delta=None,
                 train_index=None, data_use="tt"):
        if not os.path.exists(tds_dir):
            os.makedirs(tds_dir)
        interval_input = [single_interval]
        concat_option = "conc_tr"
        train_im, test_im, \
            imshape, targ_shape = rft.get_video_data(model_mom, args.data_set).forward(train_index)

        if "moving_mnist" not in args.data_set:
            if data_use is "tt":
                test_im = [v for v in test_im if test_index_use in v]
            print(test_im[0])
            _im_int, in_shape, out_shape = rft.read_frame_interval_by_dataset(args.data_set, test_im,
                                                                              time_step, concat_option,
                                                                              interval=interval_input,
                                                                              delta=delta)
            test_im_interval = _im_int
        factor = [i for i in range(50)[1:] if np.shape(_im_int)[0] % i == 0]
        args.batch_size = [factor[-1] if factor else 1][0]

        args.output_dim = targ_shape[-1]
        args.num_prediction = 1

        self.args = args
        self.temp_shape = [in_shape, out_shape]
        self.targ_shape = targ_shape
        self.output_dim = args.output_dim

        self.data_set = args.data_set
        self.train_index = train_index
        self.test_index_use = test_index_use
        self.model_dir = model_dir
        self.model_mom = model_mom
        self.tds_dir = tds_dir
        self.test_im = test_im_interval
        self.imshape = imshape

        self.batch_size = args.batch_size
        self.interval = interval_input
        self.concat = concat_option
        self.time_step = time_step
        self.learn_opt = args.learn_opt
        self.model_type = args.model_type

        if self.model_type == "2d_2d_pure_unet":
            self.num_recons = self.time_step - 1
        else:
            self.num_recons = self.time_step 

        print(args)

    def read_tensor(self):
        placeholder_shape = [None, 2, self.temp_shape[0][0]]
        shuffle_option = False
        repeat = 1
        images_in = tf.placeholder(tf.string, shape=placeholder_shape, name='tr_im_path1')

        image_queue = rft.dataset_input(self.model_mom, self.data_set, images_in, self.learn_opt,
                                        self.temp_shape, self.imshape, self.targ_shape[:2], self.batch_size,
                                        conc_option=self.concat, shuffle=shuffle_option,
                                        train_index=self.train_index,
                                        epoch_size=repeat)
        image_init = image_queue.make_initializable_iterator()
        image_batch = image_init.get_next()

        x_input = image_batch[0]  # [batch_size, num_input_channel, imh, imw, ch]
        x_output = image_batch[1]  # [batch_size, self.output_dim, imh, imw, ch]
        x_background = image_batch[2]

        x_input = tf.concat([x_input, x_output], axis=1)
        x_background = tf.transpose(x_background, perm=(1, 0, 2, 3, 4))  # num_frame,batch
        x_input = tf.transpose(x_input, perm=(1, 0, 2, 3, 4))  # num_frame, batch_size, imh, imw, ch
        if self.learn_opt == "learn_fore":
            self.x_real_input = x_input + x_background
        else:
            self.x_real_input = x_input
        return images_in, x_input, image_init, x_background

    def build_graph(self):
        num_recons_output = self.time_step

        image_placeholder, x_input, image_init, data_mean_value = self.read_tensor()


        model_use = AE.DAML(self.args)
        p_x_recons, p_x_pred, latent_space_gt, latent_space_pred = model_use.forward(x_input)

        latent_space_gt = tf.expand_dims(latent_space_gt, axis=0)
        latent_space_pred = tf.expand_dims(latent_space_pred, axis=0)

        if self.model_type == "2d_2d_pure_unet":
            x_recons_gt = self.x_real_input[1:self.time_step]
        elif self.model_type == "2d_2d_unet_no_shortcut":
            x_recons_gt = self.x_real_input[:self.time_step]
        else:
            x_recons_gt = []

        if self.learn_opt == "learn_fore":
            print("====the reconstruction is frame - background=====")
            if self.model_type != "many_to_one":               
                p_x_recons = p_x_recons + data_mean_value
            p_x_pred = p_x_pred + data_mean_value

        x_pred_gt = self.x_real_input[self.time_step:]
        self.x_recons_gt = x_recons_gt
        self.x_pred_gt = x_pred_gt

        print("==================================================================")
        print("The input for the encoder is", x_input)
        print("The reconstruction and prediction are ", p_x_recons, p_x_pred)
        print("The gt for recons and pred", self.x_recons_gt, self.x_pred_gt)
        print("==================================================================")

        var_tot = tf.trainable_variables()
        saver = tf.train.Saver(var_tot)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        v_all = os.listdir(self.model_dir)
        v_all = [v for v in v_all if '.meta' in v]
        v_all = sorted(v_all, key=lambda s: int(s.strip().split('ckpt-')[1].strip().split('.meta')[0]))
        v_all = v_all[-1]
        model_index = int(v_all.strip().split('.meta')[0].strip().split('-')[-1])
        saver.restore(self.sess, os.path.join(self.model_dir, 'model.ckpt-%d' % model_index))
        print("restore parameter from", v_all)

        input_group = [image_init, image_placeholder]

        latent_group = [latent_space_gt, latent_space_pred]

        return input_group, latent_group, p_x_recons, p_x_pred

    def save_recons_pred(self):
        """This function is used to save both reconstruction and prediction"""
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        num_im = np.shape(self.test_im)[0]
        print("The shape of frames", num_im)
        iter_visualize = num_im // self.batch_size
        input_group, latent_group, p_x_recons, p_x_pred = self.build_graph()
        images_init, image_placeholder = input_group
        test_index_act = self.test_index_use[0].strip().split('/')
        b = test_index_act[0]
        if np.shape(test_index_act)[0] >= 1:
            test_index_act = [b + v for v in test_index_act[1:]][0]
        else:
            test_index_act = self.test_index_use
        p_x_recons_reshape = tf.reshape(p_x_recons, shape=[self.num_recons, self.batch_size, -1])
        x_recons_gt_reshape = tf.reshape(self.x_recons_gt, shape=[self.num_recons, self.batch_size, -1])
        p_x_pred = tf.squeeze(p_x_pred, axis=0)
        pred_gt = tf.squeeze(self.x_pred_gt, axis=0)
        self.sess.run(images_init.initializer, feed_dict={image_placeholder: self.test_im})
        im_pred_gt_tot = np.zeros([num_im, imh, imw, ch])
        im_pred_tot = np.zeros([num_im, imh, imw, ch])
        recons_tot = np.zeros([self.num_recons, num_im, imh * imw * ch])
        im_tot = np.zeros([self.num_recons, num_im, imh * imw * ch])
        for single_iter in range(iter_visualize):
            _im_gt, _im_pred, _im_recons_gt, _im_recons = self.sess.run(fetches=[pred_gt, p_x_pred,
                                                                                 x_recons_gt_reshape,
                                                                                 p_x_recons_reshape])
            im_pred_gt_tot[single_iter * self.batch_size:(single_iter + 1) * self.batch_size] = _im_gt
            im_pred_tot[single_iter * self.batch_size:(single_iter + 1) * self.batch_size] = _im_pred
            recons_tot[:, self.batch_size * single_iter:self.batch_size * (single_iter + 1), :] = _im_recons
            im_tot[:, self.batch_size * single_iter:(single_iter + 1) * self.batch_size, :] = _im_recons_gt
        im_actual_tot = crit_multi_prediction(im_tot, [0, self.interval[0], self.num_recons])
        recons_actual_tot = crit_multi_prediction(recons_tot, [0, self.interval[0], self.num_recons])
        im_actual_tot = np.reshape(im_actual_tot, [-1, imh, imw, ch])
        recons_actual_tot = np.reshape(recons_actual_tot, [-1, imh, imw, ch])
        diff = (im_actual_tot - recons_actual_tot) ** 2
        stat_tot = [im_actual_tot, recons_actual_tot, diff]
        create_group_canvas(self.tds_dir + '/', test_index_act, stat_tot, ch, "reconstruction")
        im_diff = (im_pred_gt_tot - im_pred_tot) ** 2
        stat_tot = [im_pred_gt_tot, im_pred_tot, im_diff]
        print("The average prediction error", np.mean(im_diff))
        create_group_canvas(self.tds_dir + '/', test_index_act, stat_tot, ch, "prediction")

    def check_prediction(self, save_figure=False):
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        num_im = np.shape(self.test_im)[0]
        print("The shape of frames", num_im)
        iter_visualize = num_im // self.batch_size
        num_im = self.batch_size * iter_visualize
        input_group, latent_group, p_x_recons, p_x_pred = self.build_graph()
        images_init, image_placeholder = input_group
        p_x_pred = tf.squeeze(p_x_pred, axis=0)
        pred_gt = tf.squeeze(self.x_pred_gt, axis=0)
        test_index_act = self.test_index_use[0].strip().split('/')
        b = test_index_act[0]
        if np.shape(test_index_act)[0] >= 1:
            test_index_act = [b + v for v in test_index_act[1:]][0]
        else:
            test_index_act = self.test_index_use
        self.sess.run(images_init.initializer, feed_dict={image_placeholder: self.test_im})
        if save_figure is True:
            im_pred_gt_tot = np.zeros([num_im, imh, imw, ch])
            im_pred_tot = np.zeros([num_im, imh, imw, ch])
            for single_iter in range(iter_visualize):
                _im_gt, _im_pred = self.sess.run(fetches=[pred_gt, p_x_pred])
                im_pred_gt_tot[single_iter * self.batch_size:(single_iter + 1) * self.batch_size] = _im_gt
                im_pred_tot[single_iter * self.batch_size:(single_iter + 1) * self.batch_size] = _im_pred
            im_diff = (im_pred_gt_tot - im_pred_tot) ** 2
            stat_tot = [im_pred_gt_tot, im_pred_tot, im_diff]
            print("The average prediction error", np.mean(im_diff))
            ca_dir = self.tds_dir + '/'
            create_group_canvas(ca_dir, test_index_act, stat_tot, ch, "prediction")
        else:
            im_diff = tf.reduce_sum(tf.math.squared_difference(p_x_pred, pred_gt), axis=-1)
            diff_tot_npy = np.zeros([num_im, imh, imw])
            pred_tot_npy = np.zeros([num_im, imh, imw, ch])
            for single_iter in range(iter_visualize):
                _pred_npy, _diff_single_npy = self.sess.run(fetches=[p_x_pred, im_diff])
                pred_tot_npy[single_iter * self.batch_size:(single_iter + 1) * self.batch_size] = _pred_npy
                diff_tot_npy[single_iter * self.batch_size:(single_iter + 1) * self.batch_size] = _diff_single_npy
            np.save(self.tds_dir + '/diff_%s' % test_index_act, diff_tot_npy)
            np.save(self.tds_dir + '/pred_%s' % test_index_act, pred_tot_npy)

    def check_reconstruction(self, save_figure=False):
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        num_im = np.shape(self.test_im)[0]
        print("The shape of frames", num_im)
        iter_visualize = num_im // self.batch_size
        num_im = self.batch_size * iter_visualize
        input_group, latent_group, p_x_recons, p_x_pred = self.build_graph()
        images_init, image_placeholder = input_group
        test_index_act = self.test_index_use[0].strip().split('/')
        b = test_index_act[0]
        if np.shape(test_index_act)[0] >= 1:
            test_index_act = [b + v for v in test_index_act[1:]][0]
        else:
            test_index_act = self.test_index_use[0]
        mse_diff = tf.reduce_sum(tf.math.squared_difference(p_x_recons, self.x_recons_gt))
        mse_max = tf.reduce_max(p_x_recons, axis=(-1, -2, -3))

        p_x_recons_reshape = tf.reshape(p_x_recons, shape=[self.num_recons, self.batch_size, -1])
        x_recons_gt_reshape = tf.reshape(self.x_recons_gt, shape=[self.num_recons, self.batch_size, -1])
        self.sess.run(images_init.initializer, feed_dict={image_placeholder: self.test_im})
        if save_figure is True:
            recons_tot = np.zeros([self.num_recons, num_im, imh * imw * ch])
            im_tot = np.zeros([self.num_recons, num_im, imh * imw * ch])
            for single_iter in range(iter_visualize):
                _im, _recons = self.sess.run(fetches=[x_recons_gt_reshape, p_x_recons_reshape])
                recons_tot[:, self.batch_size * single_iter:self.batch_size * (single_iter + 1), :] = _recons
                im_tot[:, self.batch_size * single_iter:(single_iter + 1) * self.batch_size, :] = _im
            im_actual_tot = crit_multi_prediction(im_tot, [0, self.interval[0], self.num_recons])
            recons_actual_tot = crit_multi_prediction(recons_tot, [0, self.interval[0], self.num_recons])
            im_actual_tot = np.reshape(im_actual_tot, [-1, imh, imw, ch])
            recons_actual_tot = np.reshape(recons_actual_tot, [-1, imh, imw, ch])
            diff = (im_actual_tot - recons_actual_tot) ** 2
            stat_tot = [im_actual_tot, recons_actual_tot, diff]
            create_group_canvas(self.tds_dir + '/', test_index_act, stat_tot, ch, "reconstruction")
        else:
            im_recons_score = np.zeros([self.num_recons, num_im, 2])
            for single_iter in range(iter_visualize):
                _mse_diff, _mse_max = self.sess.run(fetches=[mse_diff, mse_max])
                im_recons_score[:, self.batch_size * single_iter:self.batch_size * (single_iter + 1), 0] = _mse_diff
                im_recons_score[:, self.batch_size * single_iter:self.batch_size * (single_iter + 1), 1] = _mse_max
            im_act_score = crit_multi_prediction(im_recons_score, [0, self.interval[0], self.num_recons])
            print("The shape of reconstruction based criteria", np.shape(im_act_score))
            np.save(os.path.join(self.tds_dir, 'recons_score_%s' % test_index_act), im_act_score)

    def check_all(self):
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        num_im = np.shape(self.test_im)[0]
        print("The shape of frames", num_im)
        iter_visualize = num_im // self.batch_size
        num_im = self.batch_size * iter_visualize
        input_group, latent_group, p_x_recons, p_x_pred = self.build_graph()
        images_init, image_placeholder = input_group
        latent_space_gt, latent_space_pred = latent_group

        test_index_act = self.test_index_use

        recons_mse_diff = tf.reduce_mean(tf.math.squared_difference(p_x_recons, self.x_recons_gt))
        recons_im_max = tf.reduce_max(p_x_recons, axis=(-1, -2, -3))

        latent_space_mse = tf.reduce_mean(tf.math.squared_difference(latent_space_gt, latent_space_pred), (-1, -2, -3))
        latent_space_cos = tf.squeeze(loss_tf.calculate_cosine_dist(latent_space_pred, latent_space_gt, 
                                                                    self.batch_size), axis=-1)
        latent_space_l1norm = tf.reduce_mean(tf.abs(tf.subtract(latent_space_gt, latent_space_pred)), axis=(-1, -2, -3))
        latent_space_mse = tf.squeeze(latent_space_mse, axis=0)
        latent_space_cos = tf.squeeze(latent_space_cos, axis=0)
        latent_space_l1norm = tf.squeeze(latent_space_l1norm, axis=0)

        p_x_pred = tf.squeeze(p_x_pred, axis=0)
        pred_gt = tf.squeeze(self.x_pred_gt, axis=0)

        pred_im_mse = tf.reduce_mean(tf.math.squared_difference(p_x_pred, pred_gt), (-1, -2, -3))
        pred_im_max = tf.reduce_max(p_x_pred, axis=(-1, -2, -3))

        pred_based_score = np.zeros([num_im, 5])
        recons_based_score = np.zeros([self.num_recons, num_im, 2])
        self.sess.run(images_init.initializer, feed_dict={image_placeholder: self.test_im})
        for single_iter in range(iter_visualize):
            _recons_mse_diff, _recons_max, _latent_mse, _latent_cos, _latent_l1norm, \
            _pred_mse_diff, _pred_max = self.sess.run(fetches=[recons_mse_diff, recons_im_max,
                                                               latent_space_mse, latent_space_cos, latent_space_l1norm,
                                                               pred_im_mse, pred_im_max])
            for j, single_stat in enumerate([_latent_mse, _latent_cos, _latent_l1norm, _pred_mse_diff, _pred_max]):
                pred_based_score[single_iter * self.batch_size:(single_iter + 1) * self.batch_size, j] = single_stat
            for j, single_stat in enumerate([_recons_mse_diff, _recons_max]):
                recons_based_score[:, single_iter * self.batch_size:(single_iter + 1) * self.batch_size,
                j] = single_stat
        recons_based_score_act = crit_multi_prediction(recons_based_score, [0, self.interval[0], self.num_recons])
        np.save(os.path.join(self.tds_dir, 'recons_score_%s' % test_index_act), recons_based_score_act)
        np.save(os.path.join(self.tds_dir, 'pred_score_%s' % test_index_act), pred_based_score)


def create_group_canvas(tds_dir, test_index_use, stat_tot, ch, name):
    for iterr, single_stat in enumerate(stat_tot):
        if ch == 1:
            single_stat = np.repeat(single_stat, 3, -1)
        stat_tot[iterr] = (single_stat * 255.0).astype('uint8')
    ca_ind = 20
    num_canvas = int(np.ceil(np.shape(stat_tot[0])[0] / ca_ind))
    for i in range(num_canvas):
        ca_use = []
        for j in range(3):
            if i != num_canvas - 1:
                ca_use.append(stat_tot[j][i * ca_ind:(i + 1) * ca_ind])
            else:
                ca_use.append(stat_tot[j][i * ca_ind:])
        ca_use = np.array(ca_use)
        num_row = np.shape(ca_use)[0]
        ca_ = create_canvas(ca_use, [num_row, np.shape(ca_use)[1]])
        cv2.imwrite(tds_dir + '%s_diff_%s_%d.png' % (name, test_index_use, i), ca_.astype('uint8')[:, :, ::-1])


def crit_multi_prediction(use_stat, delta):
    """this function is for aggregating the score for multiple predictions
    use_stat: [num_prediction, num_im, num_crit_score]
    delta: [gap_between_input_and_out, gap_between_output, num_output]
    Output:
        [actual_num_im, num_crit_score]
    so the input can be actual score, the num_crit_score will be 7
    or the input can be actual image, the num_crit_score: imh*imw*ch
    or the input can be latent space, the num_crit_score: fh*fw*ch
    I need to remember to reshape the actual image and latent space 
    if I pass them into this function
    """
    num_prediction, num_im, num_crit = np.shape(use_stat)
    actual_tot_num = (delta[2] - 1) * delta[1] + num_im
    stat_new_tot = []
    for single_pred in range(num_prediction):
        stat_sub = use_stat[single_pred]
        before = single_pred * delta[1]
        end = actual_tot_num - before - num_im
        if before != 0:
            before_mat = np.zeros([before, num_crit])
            stat_sub = np.concatenate([before_mat, stat_sub], axis=0)
        if end != 0:
            end_mat = np.zeros([end, num_crit])
            stat_sub = np.concatenate([stat_sub, end_mat], axis=0)
        stat_new_tot.append(stat_sub)
    stat_new_tot = np.array(stat_new_tot)
    stat_update = np.zeros([actual_tot_num, num_crit])
    for single_im in range(actual_tot_num):
        multi_stat_for_one_im = stat_new_tot[:, single_im, :]
        not_equal_zero = np.mean(multi_stat_for_one_im, axis=-1) != 0
        left = np.mean(multi_stat_for_one_im[not_equal_zero, :], axis=0)
        stat_update[single_im, :] = left
    return stat_update


def create_canvas(image, nx_ny):
    """This function is used to create the canvas for the images
    image: [Num_im, imh, imw, 3]
    nx_ny: the number of row and columns in the canvas
    """
    nx, ny = nx_ny
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    targ_height, targ_width, num_ch = np.shape(image[0])[1:]
    canvas = np.empty((targ_height * nx, targ_width * ny, num_ch))
    for i, yi in enumerate(x_values):
        im_use_init = image[i]
        for j, xi in enumerate(y_values):
            im_use_end = im_use_init[j]
            # im_use_end[:, 0, :] = [204, 255, 229]
            # im_use_end[:, -1, :] = [204, 255, 229]
            # im_use_end[0, :, :] = [204, 255, 229]
            # im_use_end[-1, :, :] = [204, 255, 229]
            canvas[(nx - i - 1) * targ_height:(nx - i) * targ_height, j * targ_width:(j + 1) * targ_width,
            :] = im_use_end
    return canvas


def give_im_in_list(im):
    return [im[i * 64:(i + 1) * 64] for i in range(3)]


def sort_im(im_pred, im_recons, digit):
    """This function is used to show the reconstruction and prediction
    in the same image
    im_pred: [192, 64*13, 3]
    im_recons: [192, 64*17, 3]
    then sort the images in a list.
    im_recons
    """
    im_recons_new = im_recons[:, (-6*64):, :]
    im_pred_new = im_pred[:, :(6*64), :]
    
    pred_diff = im_pred_new[:64, :, :]
    pred_diff_sum = [np.sum(pred_diff[:, i*64:(i+1)*64, :]) for i in range(6)]
    if digit in ["4", "7"]:
        select_im_index = np.argmin(pred_diff_sum)
    else:
        select_im_index = np.argmax(pred_diff_sum)
    im_recons_use = im_recons_new[:, select_im_index * 64:(select_im_index + 1) * 64, :]
    im_pred_use = im_pred_new[:, select_im_index * 64:(select_im_index + 1) * 64, :]

    recons_list = give_im_in_list(im_recons_use)
    pred_list = give_im_in_list(im_pred_use)
    tot_im = [recons_list[2], recons_list[1], pred_list[1], pred_list[0]]
#    tot_im = [pred_list[0], pred_list[1], recons_list[1], recons_list[2]]
#    tot_im = [[v] for v in tot_im]
    ca = create_canvas([tot_im], [1, 4])
    return ca



if __name__ == '__main__':
    args = const.args
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("-------------------------------------------------------------------")
    test_end2end_model(args, args.data_set, model_type=args.model_type, motion_method=args.motion_method,
                       version=args.version, opt=args.opt)

    