#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:48:34 2019
This script is for producing the fps
In real life, this one should be used, since it's more efficient
@author: li
"""
import tensorflow as tf
import models.AE as ae
from data import read_frame_temporal
import numpy as np
import os
import evaluate as ev
import argparse
import optimization.loss_tf as loss_tf
import const 

# parser = argparse.ArgumentParser(description='Training anomaly detection model using end2end')
# parser.add_argument('-ds', '--data_set', type=str, default='ucsd1', metavar='DATA_SET',
#                     help='the used dataset can be ucsd1, ucsd2, avenue, shanghaitech, kth_walk')
# parser.add_argument('-bs', '--batch_size', type=int, default=64, metavar='BATCH_SIZE',
#                     help='input batch size for training (default: 100)')
# parser.add_argument('-ne', '--num_encode_layer', type=int, default=4, metavar='NUM_ENCODER_LAYER',
#                     help='the number of encoder layers')
# parser.add_argument('-nd', '--num_decode_layer', type=int, default=4, metavar='NUM_DECODER_LAYER',
#                     help='the number of decoder layers')
# parser.add_argument('-od', '--output_dim', type=int, default=4, metavar='OUTPUT_DIM',
#                     help='the last dimension of the input')
# parser.add_argument('-regu','--regu_par', type=float, default=0.001, metavar='L2_REGU_PAR',
#                     help='L2 regularization parameter')
# parser.add_argument('-zmr','--z_mse_ratio', type=float, default=0.001, metavar='Z_MSE_RATIO',
#                     help='weight for latent space mse loss')
# parser.add_argument('-io','--input_option', type=str, default="original", metavar='INPUT_OPTION',
#                     help='whether the input for the model is cropped or not')
# parser.add_argument('-dv','--darker_value', type=float, default=0.0, metavar='DARKER_VALUE',
#                     help='the degree of brightness, 0.0 means using original frames')
# parser.add_argument('-mt', '--model_type', type=str, default='2d_2d_pure_unet', metavar='MODEL_TYPE',
#                     help='the selected base model')
# parser.add_argument('-mm', '--motion_method', type=str, default='conv3d', metavar='MOTION_MODEL',
#                     help='the selected motion method')
# parser.add_argument('-lo', '--learn_opt', type=str, default='learn_fore', metavar='LEARN_OPT',
#                     help='Whether the background frame needs to be subtracted from the original frame')

# args = parser.parse_args()

import time

def test_end2end_model(args, data_set, version=0):
    path_for_load_data = args.datadir
    model_mom = args.expdir
    model_type = args.model_type
    motion_method = args.motion_method
    if data_set == "ucsd1":
        stat = [8, 6, 2, 5]
        time_tot = test_except_shanghaitech(path_for_load_data, model_mom, "ucsd1", 
                                            stat[0], stat[1], stat[2], model_type, motion_method, 
                                            stat[3], version = version)
    elif data_set == "ucsd2":
        stat = [8, 6, 2, 4]
        time_tot = test_except_shanghaitech(path_for_load_data, model_mom, "ucsd2", 
                                            stat[0], stat[1], stat[2], model_type, motion_method, 
                                            stat[3], version = version)
    elif data_set == "avenue":
        stat = [6, 6, 2, 4]
        time_tot = test_except_shanghaitech(path_for_load_data, model_mom, "avenue", 
                                            stat[0], stat[1], stat[2], model_type, motion_method, 
                                            stat[-1], version = version)
    
    print("-----------------Giving the AUC score------------------------------")
    test_model(path_for_load_data, model_mom, data_set, stat[0], stat[1], stat[2], model_type, 
               motion_method, version, stat[3], "learn_fore", "test", opt = "eval_score")
    print("Evaluation time for per frame")
    print("The averaged evaluation time for per frame %.4f and FPS %d"%(np.mean(time_tot), 1/np.mean(np.array(time_tot)[2:])))
    return time_tot


def test_except_shanghaitech(path_for_load_data, model_mom, data_set, 
                             time_step, delta, single_interval, model_type, motion_method, 
                             num_layer = 4,learn_opt="learn_fore",version=0, opt = "save_score"):
    test_index_all, gt = ev.read_test_index(path_for_load_data, data_set)
    time_tot = []
    for single_ind in test_index_all:
        time_avg = test_model(path_for_load_data, model_mom, data_set, time_step, delta,single_interval, 
                   model_type, motion_method, version, 
                   num_layer, learn_opt, 
                   single_ind, opt)
        time_tot.append(time_avg)
    return time_tot


def test_model(model_mom_for_load_data, path_mom, data_set, 
               time_step, delta, single_interval, model_type, motion_method, version, num_enc_layer, 
               learn_opt,test_index_use, opt, bg_index = None, select_bg_index = None):
    args.data_set = data_set
    args.num_encoder_layer = num_enc_layer
    args.num_decoder_layer = num_enc_layer
    args.model_type = model_type
    args.time_step = time_step
    args.motion_model = motion_method
    args.learn_opt = learn_opt
    if not select_bg_index:
        model_dir = path_mom+"ano_%s_motion_end2end/time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_version_%d"%(args.data_set, time_step, 
                                                                                                         delta, single_interval, 
                                                                                                         model_type, motion_method, 
                                                                                                         learn_opt, num_enc_layer, 
                                                                                                         version)
        tds_dir = model_dir.strip().split('/time_')[0]+'/tds_real_new/'+model_dir.strip().split('end2end/')[1]
    else:
        model_dir = path_mom+"ano_%s_motion_end2end/time_%d_delta_%d_gap_%d_%s_%s_%s_enc_%d_bg_%d_version_%d"%(args.data_set,
                                                                                                               time_step,
                                                                                                               delta, 
                                                                                                               single_interval,
                                                                                                               model_type, motion_method,
                                                                                                               learn_opt, num_enc_layer, 
                                                                                                               select_bg_index,
                                                                                                               version)
        tds_dir = model_dir.strip().split('/time_')[0]+'/tds_real_new/'+model_dir.strip().split('end2end/')[1]
    if opt == "eval_score":
        ev.get_auc_score_efficient(model_mom_for_load_data, tds_dir, data_set, [time_step, delta, single_interval])
    else:
        if not os.path.isfile(tds_dir + "/latent_cos_%s.npy" % test_index_use):
            tmf = test_main_func(args, model_mom_for_load_data, model_dir, tds_dir, time_step, single_interval, test_index_use, delta, train_index = bg_index)
            if delta == 2:
                time_avg = tmf.calc_fps()
            else:
                time_avg = tmf.calc_fps_delta_not_two()
            return time_avg
        else:
            return 1

    
class test_main_func(object):
    def __init__(self, args, model_mom, model_dir, tds_dir, time_step, single_interval, test_index_use, delta = None, train_index=None, data_use="tt"):
        if not os.path.exists(tds_dir):
            os.makedirs(tds_dir)
        interval_input = [single_interval]
        concat_option = "conc_tr"
        train_im, test_im, imshape, targ_shape = read_frame_temporal.get_video_data(model_mom, args.data_set).forward(train_index)
        if data_use == "tt":
            test_im = [v for v in test_im if test_index_use in v]
        print(test_im[0])
        args.output_dim = targ_shape[-1]
        test_im_interval, in_shape, out_shape = read_frame_temporal.read_frame_interval_by_dataset(args.data_set, 
                                                                                                   test_im,
                                                                                                   time_step, 
                                                                                                   concat_option,
                                                                                                   interval= interval_input,
                                                                                                   delta = delta)

        args.batch_size = interval_input[0]
        args.num_prediction = 1
        
        self.temp_shape = [in_shape, out_shape]
        self.targ_shape = targ_shape
        self.output_dim = args.output_dim

        self.data_set = args.data_set
        self.train_index = train_index
        self.test_index_use = test_index_use

        self.model_dir = model_dir
        self.tds_dir = tds_dir
        self.model_mom = model_mom
        self.test_im = test_im_interval

        self.batch_size = args.batch_size
        self.interval = interval_input
        self.concat = concat_option
        self.time_step = time_step
        self.delta = delta
        self.learn_opt = args.learn_opt
        self.concat = concat_option
        self.model_type = args.model_type
        self.motion_method = args.motion_method
        self.imshape = imshape
        
        self.args = args

    def read_tensor(self):
        imh, imw, ch = self.targ_shape
        if self.concat == "full_predict":
            placeholder_shape = [None, 2, np.max([self.temp_shape[0][0], self.temp_shape[1][0]])]
        else:
            placeholder_shape = [None, 2, self.temp_shape[0][0]]
        shuffle_option = False
        repeat = 1
        images_in = tf.placeholder(tf.string, shape = placeholder_shape, name = 'tr_im_path')
        image_queue = read_frame_temporal.dataset_input(self.model_mom, self.data_set, images_in, self.learn_opt,
                                                        self.temp_shape, self.imshape, self.targ_shape[:2], self.batch_size,
                                                        conc_option = self.concat, shuffle=shuffle_option,
                                                        train_index=self.train_index,
                                                        epoch_size = repeat)
        
        image_init = image_queue.make_initializable_iterator()
        image_batch = image_init.get_next()
        x_input = image_batch[0] #[batch_size, num_input_channel, imh, imw, ch]
        x_output = image_batch[1] #[batch_size, self.output_dim, imh, imw, ch]

        x_input = tf.concat([x_input, x_output], axis = 1)
        x_input = tf.transpose(x_input, perm = (1,0,2,3,4)) #num_frame, batch_size, imh, imw, ch
        #the last input of x_input is for prediction        
        x_output = tf.transpose(x_output, perm = (1,0,2,3,4)) #[1, batch_size, ]
        return images_in, image_init, x_input, x_output 
    
    def build_graph(self):
        imh, imw, ch = self.targ_shape
        num_process_im = 1
        num_recons_output = self.time_step

        image_placeholder, image_init, x_input, x_output = self.read_tensor()
        print("The number of frames that are going to be predicted", num_process_im)
        print("The number of frames that are going to be reconstruct", num_recons_output)
        print("The input to the encoder", x_input)
        x_input_placeholder = tf.placeholder(tf.float32, shape = [1, None, imh, imw, ch],
                                             name = "input_placeholder")        
        
        if self.model_type == "2d_2d_pure_unet":
            model_use = ae.DAML(self.args)
            
        latent_space, latent_space_for_motion_placeholder, \
            latent_space_pred, latent_space_gt = model_use.build_unet_fps(x_input_placeholder)
        
                
#         if self.model_type == "2d_2d_pure_unet":
#             vae_use = ae.AE_2d_2d_unet(args)
#         latent_space, _ = vae_use.build_common_encoder(x_input_placeholder)
        # this latent space only has one time step, each time I only process
        # one frame at a time, in the begining, I process process multiple frames
        # and concatenate them. 
        #-----build motion model-------#
#         motion_model = ae.learn_motion_in_latent_space(args)
#         latent_space_pred = motion_model.build_motion_latent(latent_space_for_motion_placeholder, 
#                                                              self.motion_method) #batch_size, fh,fw,ch
        #---finish build motion model ----#
        print("The predicted latent space shape", latent_space_pred.shape)
        print("The latent gt shape", latent_space_gt.shape)
        
        var_tot = tf.trainable_variables()
        saver = tf.train.Saver(var_tot)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())                    
        v_all = os.listdir(self.model_dir)
        v_all = [v for v in v_all if '.meta' in v]
        v_all = sorted(v_all, key=lambda s:int(s.strip().split('ckpt-')[1].strip().split('.meta')[0]))
        v_all = v_all[-1]
        model_index = int(v_all.strip().split('.meta')[0].strip().split('-')[-1])
        saver.restore(self.sess, os.path.join(self.model_dir, 'model.ckpt-%d'%model_index))
        print("restore parameter from", v_all)
        
        input_group = [image_placeholder, image_init, x_input, x_output, x_input_placeholder]
        latent_placeholder_group = [latent_space_for_motion_placeholder, latent_space_gt]
        latent_group = [latent_space, latent_space_gt, latent_space_pred]        
        
        return input_group, latent_placeholder_group, latent_group
    
    def calc_fps(self):
        """
        okay, let's write the procedure 
        1) For the first #interval times, I need to run the x_input 
            option, to get (time_steps+1) input. If interval is 2, and time steps 
            is 6, I can get 14 input in total. 
        2) Then these 14 input passed to the encoder to get 14 latent space. 
        3) Then I will always have #interval*(time_step+1) latent space, 
            and these latent space are selected based on the interval, i.e., 
            0,2,4,6,8,10,12 latent space. The first time steps are for 
            the placeholder in the motion model, the last one is latent space gt
        4) I am still running iter_visualize iteration. batch_size = interval
        """
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        num_im = np.shape(self.test_im)[0]
        print("The shape of frames", num_im) #batch_size need to be 1
        iter_visualize = num_im//self.batch_size
        num_im = self.batch_size*iter_visualize

        input_group, latent_placeholder_group, latent_group = self.build_graph()
        image_placeholder, image_init, x_input, x_output, x_input_placeholder = input_group
        latent_space_for_motion_placeholder, latent_space_gt_placeholder = latent_placeholder_group
        latent_space_from_encoder, latent_space_gt, latent_space_pred = latent_group
        if self.data_set != "avenue":
            latent_space_cos = tf.squeeze(loss_tf.calculate_cosine_dist(latent_space_pred, latent_space_gt), axis = (0,-1))
        elif self.data_set == "avenue":
            latent_space_cos = tf.reduce_mean(tf.squared_difference(latent_space_pred, latent_space_gt), axis = (-1,-2,-3)) 
        
        fh, fw, f_ch = latent_space_gt.get_shape().as_list()[1:]
        cos_value = np.zeros([num_im])
        
        self.sess.run(image_init.initializer, feed_dict={image_placeholder:self.test_im})
        time_tot = []
        for single_iter in range(iter_visualize):
            if single_iter == 0:
                x_input_npy = self.sess.run(fetches = x_input) #[num_Frame, batch,]
                x_input_npy = np.reshape(x_input_npy,  [1,self.interval[0]*(self.time_step+1), imh, imw, ch])
                #correct
                latent_space_value = self.sess.run(fetches = latent_space_from_encoder,
                                                              feed_dict = {x_input_placeholder:x_input_npy})
                #latent_space: [14, fh, fw, ch]
                latent_space_update = latent_space_value[0]
                latent_space_gt_npy = latent_space_update[-self.interval[0]:] 
                latent_space_to_motion = np.reshape(latent_space_update[:-self.interval[0]], 
                                                    [self.time_step,
                                                     self.interval[0],
                                                     fh, fw, f_ch])
                _cos_dist = self.sess.run(fetches = latent_space_cos,
                                          feed_dict={latent_space_for_motion_placeholder:latent_space_to_motion,
                                                     latent_space_gt_placeholder: latent_space_gt_npy})
                cos_value[single_iter*self.interval[0]:(single_iter+1)*self.interval[0]] = _cos_dist
                latent_space_old = latent_space_update
            else:
                time_init = time.time()
                x_output_npy = self.sess.run(fetches = x_output)
                latent_space_updated_gt = self.sess.run(fetches = latent_space_from_encoder, 
                                                        feed_dict = {x_input_placeholder:x_output_npy})
                latent_space_update = latent_space_old[self.interval[0]:]
                latent_space_update = np.concatenate([latent_space_update, 
                                                      latent_space_updated_gt[0]],axis =0)
                latent_space_gt_npy = latent_space_update[-self.interval[0]:]
                latent_space_to_motion = np.reshape(latent_space_update[:-self.interval[0]],
                                                    [self.time_step, 
                                                     self.interval[0],
                                                     fh, fw, f_ch])
                _cos_dist = self.sess.run(fetches = latent_space_cos, 
                                          feed_dict = {latent_space_for_motion_placeholder: latent_space_to_motion,
                                                       latent_space_gt_placeholder: latent_space_gt_npy})
                cos_value[single_iter*self.interval[0]:(single_iter+1)*self.interval[0]] = _cos_dist
                time_end = time.time()
                time_tot.append(time_end-time_init)
                latent_space_old = latent_space_update
            np.save(os.path.join(self.tds_dir, 'latent_cos_%s'%self.test_index_use),
                    cos_value)
        return np.mean(time_tot)/self.interval[0]
            #I have kind of finishing it, but I still need to test it!
            #I think the only way to test it is to pass them to decoder to get
            #reconstruction and prediction!
            
    def calc_fps_delta_not_two(self):
        """
        okay, let's write the procedure
        This is the function to calculate the fps when delta is not 2. 
        The case I have is time 8, interval 2, delta 6. 
        The actual number of iter num_frame/delta, but batch size still equal to interval
        1) For the first batch, I need to run x_input to get (time_steps+1) input. If the interval
        is 2, and time steps is 6, then I get 14 inputs
        2) The latent space is always delta value x fh x fw x ch
        3) These 14 input pass to the encoder to get 14 corresponding latent space. 
        4) Then for per iteration  (total number iteration = delta/batch_size)
        5) I pass the latent space to the motion model to get the score. 
        6) I run the input option in order to get the last 2 input. and the latent space. 
        7) I save #delta latent space. 
                
        """
        tf.reset_default_graph()
        imh, imw, ch = self.targ_shape
        num_im = np.shape(self.test_im)[0]
        print("The shape of frames", num_im) #batch_size need to be 1
        iter_visualize = num_im//self.delta
        num_im = self.delta*iter_visualize

        input_group, latent_placeholder_group, latent_group = self.build_graph()
        image_placeholder, image_init, x_input, x_output, x_input_placeholder = input_group
        latent_space_for_motion_placeholder, latent_space_gt_placeholder = latent_placeholder_group
        latent_space_from_encoder, latent_space_gt, latent_space_pred = latent_group

        if self.data_set != "avenue":
            latent_space_cos = tf.squeeze(loss_tf.calculate_cosine_dist(latent_space_pred, latent_space_gt, self.batch_size), axis = (0,-1))
        elif self.data_set == "avenue":
            latent_space_cos = tf.reduce_mean(tf.squared_difference(latent_space_pred, latent_space_gt), axis = (-1,-2,-3)) 
        
        fh, fw, f_ch = latent_space_gt.get_shape().as_list()[1:]
        cos_value = np.zeros([num_im])
        
        self.sess.run(image_init.initializer, feed_dict={image_placeholder:self.test_im})
        time_tot = []
        latent_space_for_saving = np.zeros([self.delta, fh, fw, f_ch])
        for single_iter in range(iter_visualize):
            if single_iter == 0:
                x_input_npy = self.sess.run(fetches = x_input) #[num_Frame, batch,]
                x_input_npy = np.reshape(x_input_npy,  [1,self.interval[0]*(self.time_step+1), imh, imw, ch])
                latent_space_value = self.sess.run(fetches = latent_space_from_encoder,
                                                   feed_dict = {x_input_placeholder:x_input_npy})
                latent_space_update = latent_space_value[0]
                latent_space_to_motion = latent_space_update[:-self.interval[0]]
                latent_space_gt_npy = latent_space_update[-self.interval[0]:]

                latent_space_for_saving[0:self.interval[0]] = latent_space_gt_npy
                latent_space_to_motion_reshape = np.reshape(latent_space_to_motion, 
                                                        [self.time_step,
                                                         self.interval[0],
                                                         fh, fw, f_ch])
                _cos_dist = self.sess.run(fetches = latent_space_cos,
                                          feed_dict={latent_space_for_motion_placeholder:latent_space_to_motion_reshape,
                                                     latent_space_gt_placeholder: latent_space_gt_npy})
                cos_value[0:self.interval[0]] = _cos_dist
                    
                for j in range(self.delta//self.interval[0]-1):                    
                    #latent_space: [14, fh, fw, ch]      
                    x_input_npy_new = self.sess.run(fetches = x_input[-2:])
                    x_input_npy_new = np.reshape(x_input_npy_new, [1, self.interval[0]*2, imh, imw, ch]) #12,18,13,19
                    latent_space_for_new_data = self.sess.run(fetches = latent_space_from_encoder[0], 
                                                              feed_dict = {x_input_placeholder:x_input_npy_new}) #12,13,18,19
                    latent_space_gt_npy = latent_space_for_new_data[-self.interval[0]:]
                    latent_space_to_motion = np.concatenate([latent_space_to_motion[self.interval[0]:],
                                                             latent_space_for_new_data[:self.interval[0]]],axis = 0)
                    latent_space_to_motion_reshape = np.reshape(latent_space_to_motion, [self.time_step, self.interval[0],
                                                                                         fh, fw, f_ch])
                    _cos_dist = self.sess.run(fetches = latent_space_cos, 
                                              feed_dict = {latent_space_for_motion_placeholder:latent_space_to_motion_reshape,
                                                           latent_space_gt_placeholder: latent_space_gt_npy})
                    cos_value[(single_iter*self.delta+(j+1)*self.interval[0]):(single_iter*self.delta+(j+2)*self.interval[0])] = _cos_dist

                    latent_space_for_saving[(j+1)*self.interval[0]:(j+2)*self.interval[0]] = latent_space_gt_npy
                latent_space_to_motion_old = latent_space_to_motion
            else:
                time_init = time.time()
                for j in range(self.delta//self.interval[0]):
                    x_output_npy = self.sess.run(fetches = x_output)
                    latent_space_updated_gt = self.sess.run(fetches = latent_space_from_encoder, 
                                                            feed_dict = {x_input_placeholder: x_output_npy}) #22,23
                    latent_space_to_motion = np.concatenate([latent_space_to_motion_old[self.interval[0]:],
                                                             latent_space_for_saving[:self.interval[0]]],
                                                             axis = 0) #[6,7,8,9,10,11,12,12 - 17]
                    latent_space_for_saving = np.concatenate([latent_space_for_saving[self.interval[0]:],
                                                              latent_space_updated_gt[0]], axis = 0)
                    latent_space_to_motion_reshape = np.reshape(latent_space_to_motion,
                                                                [self.time_step, self.interval[0], fh, fw, f_ch])
                    latent_space_gt_npy = latent_space_updated_gt[0]
                    _cos_dist = self.sess.run(fetches = latent_space_cos, 
                                              feed_dict = {latent_space_for_motion_placeholder: latent_space_to_motion_reshape,
                                                           latent_space_gt_placeholder: latent_space_gt_npy})
                    cos_value[(single_iter*self.delta+j*self.interval[0]):(single_iter*self.delta+(j+1)*self.interval[0])] = _cos_dist
                    
                    latent_space_to_motion_old = latent_space_to_motion                    
                time_end = time.time()
                time_tot.append(time_end-time_init)
        np.save(os.path.join(self.tds_dir, 'latent_cos_%s'%self.test_index_use),
                cos_value)
        return np.mean(time_tot)/self.delta
#         return np.sum(time_tot)/num_im



if __name__ == '__main__':
    args = const.args
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("-------------------------------------------------------------------")
    
    test_end2end_model(args, args.data_set, args.version)

    
    
        
                
                    
                    
                

                
            

        
    
        
