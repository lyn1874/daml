#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:13:36 2019
This script is used for reading video frames with temporal information
The input frames are loaded with a stride, i.e.,
if there are 6 input frames with a stride 2, then the input frames indices will be
0, 2, 4, 6, 8, 10. The output frame will be 12 if the distance between last input frame
and output frame is also 2.
Only one step prediction is considered
@author: li
"""
import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as io


class get_video_data(object):
    def __init__(self, model_mom, data_set):
        super(get_video_data, self).__init__()
        self.model_mom = model_mom
        self.data_set = data_set

    def read_ucsd_ped(self, num=1):
        path_mom = self.model_mom + "UCSDped%d/" % num
        for tr_or_tt in ["Train", "Test"]:
            path = path_mom + tr_or_tt + '_jpg'  # [ucsdped1/train]
            if not os.path.exists(path):
                all_path = []
                print("path %s doesn't exist, this is fine for testing but not fine for training" % path)
            else:
                all_path = sorted(os.listdir(path))  # [ucsdped1/train/train001]
                all_path = [path + '/' + v for v in all_path if 'jpg' in v]
            if tr_or_tt is "Train":
                tr_tot = all_path
            elif tr_or_tt is "Test":
                tt_tot = all_path
        print("There are %d training images and %d test images" % (np.shape(tr_tot)[0], np.shape(tt_tot)[0]))
        if num == 1:
            imshape = np.array([158, 238, 3])
        elif num == 2:
            imshape = np.array([240, 360, 3])
        targshape = np.array([128, 192, 1])
        return np.array(tr_tot), np.array(tt_tot), imshape, targshape

    def read_avenue_data(self):
        path_mom = self.model_mom + "Avenue/frames/"
        for tr_or_tt in ["training", "testing"]:
            path = path_mom + tr_or_tt
            all_path = os.listdir(path)
            all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
            all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
            all_path = [path + '/' + v for v in all_path if 'jpg' in v]
            if tr_or_tt is "training":
                tr_tot = all_path
            elif tr_or_tt is "testing":
                tt_tot = all_path
        print("there are %d training images and %d test images" % (np.shape(tr_tot)[0], np.shape(tt_tot)[0]))
        imshape = np.array([360, 640, 3])
        targshape = np.array([128, 224, 3])
        return np.array(tr_tot), np.array(tt_tot), imshape, targshape

    def read_shanghaitech(self, bg_name):
        path_tr_mom = self.model_mom + "shanghaitech/frames/training/" + bg_name
        test_index = bg_name.strip().split('bg_index_')[1]
        path_tt_mom = self.model_mom + "shanghaitech/original/testing/frames/"

        path_tr_tot = os.listdir(path_tr_mom)
        path_tr_tot = sorted(path_tr_tot, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
        path_tr_tot = sorted(path_tr_tot, key=lambda s: int(s.strip().split('index_')[1].strip().split('_frame')[0]))
        path_tr_tot = [path_tr_mom + '/' + v for v in path_tr_tot]

        path_all_tt = os.listdir(path_tt_mom)
        path_subset = [v for v in path_all_tt if test_index + '_' in v]
        path_subset = sorted(path_subset, key=lambda s: int(s.strip().split("_")[1].lstrip("0")))
        path_tt_tot = []
        for single_path in path_subset:
            single_path_tr = path_tt_mom + single_path
            path_from_single_path = os.listdir(single_path_tr)
            path_from_single_path = sorted(path_from_single_path, key=lambda s: int(s.strip().split('.jpg')[0]))
            path_tt_from_single_path = [single_path_tr + '/' + v for v in path_from_single_path]
            path_tt_tot.append(path_tt_from_single_path)
        path_tt_tot = [v for j in path_tt_tot for v in j]

        imshape = np.array([128, 224, 3])
        targshape = np.array([128, 224, 3])
        return np.array(path_tr_tot), np.array(path_tt_tot), imshape, targshape

    def read_streetscene(self):
        path_tr_mom = self.model_mom + 'streetscene/Train/'
        train_index_all = sorted(os.listdir(path_tr_mom))
        path_tr_mom = [path_tr_mom + v for v in train_index_all]
        path_tr_all = [sorted(os.listdir(v)) for v in path_tr_mom]
        path_tt_mom = self.model_mom + 'streetscene/Train/'
        test_index_all = sorted(os.listdir(path_tt_mom))
        path_tt_mom = [path_tt_mom + v for v in test_index_all]
        path_tt_all = [sorted(os.listdir(v)) for v in path_tt_mom]

        path_tr_tot = [single_path_name + '/' + v for single_path_name, j in zip(path_tr_mom, path_tr_all) for v in j]
        path_tt_tot = [single_path_name + '/' + v for single_path_name, j in zip(path_tt_mom, path_tt_all) for v in j]
        imshape = np.array([720, 1280, 3])
        targshape = np.array([288, 512, 3])

        return np.array(path_tr_tot), np.array(path_tt_tot), imshape, targshape

    def forward(self, bg_name=None):
        if "ucsd" in self.data_set:
            num = int(self.data_set.strip().split("sd")[-1])
            return self.read_ucsd_ped(num)
        elif self.data_set is "avenue":
            return self.read_avenue_data()
        elif self.data_set is "shanghaitech":
            return self.read_shanghaitech(bg_name)
        else:
            print("===============the required dataset doesn't exist yet===============")


def read_frame_interval_by_dataset(data_set, tr_im, time_step, concat, interval, delta=None):
    """This function is used to read the frames as sequence
    Args:
        data_set: "ucsd1", "ucsd2", "avenue", "shanghaitech", "streetscene"
        tr_im: the loaded filenames using function get_video_data(), need to be an array [total_num_training_data]
        time_step: int, the number of input frames.
        concat: "conc_tr". The model predicts a single frame given a input sequence
        interval: [int], the stride between each two of input frames, if interval is 2, then the input is 0,2,4...
        delta: int, the time difference between last of the input frame and output frame.
    Output:
        tr_im_temporal: the sequential frames. [time_steps, 2, num_input_frame]
        in_shape: the number of input frame
        oup_shape: the number of output frame
    Ops:
        1. Arrange the input frames based on the video index --> name_space
        2. Then for each video, arrange the input frame from earliest to latest
        3. Then use function read_frame_interval to read the input as sequential
        4. Concate all the sequential frames for each video together
    """
    if "ucsd" in data_set:
        video_index = [v.strip().split('_jpg/')[1].strip().split('_')[0] for v in tr_im]
        num_test_video = np.shape(np.unique(video_index))[0]
        name_space = np.unique(video_index)
        print("There are %d videos for dataset %s" % (num_test_video, data_set))
        tr_temporal_tot = []
        for single_index in range(num_test_video):
            name_use = name_space[single_index] + '_'
            tr_subset = sorted([v for v in tr_im if name_use in v],
                               key=lambda s: int(s.strip().split(name_use)[-1].lstrip('0').split('.jpg')[0]))
            tr_subset = np.array(tr_subset)
            if concat is not "full_predict":
                tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta)
            tr_temporal_tot.append(tr_temp)
        tr_temporal_tot = np.array([y for x in tr_temporal_tot for y in x])
    elif data_set is "avenue":
        video_index = [v.strip().split('_video_')[1].strip().split('_frame_')[0] for v in tr_im]
        num_test_video = np.shape(np.unique(video_index))[0]
        name_space = np.unique(video_index)
        name_space = ['_video_' + v + '_frame_' for v in name_space]
        print("There are %d videos for dataset %s" % (num_test_video, name_space))
        tr_temporal_tot = []
        for single_index in range(num_test_video):
            name_use = name_space[single_index]
            tr_subset = sorted([v for v in tr_im if name_use in v],
                               key=lambda s: int(s.strip().split(name_use)[-1].strip().split('.jpg')[0]))
            tr_subset = np.array(tr_subset)
            if concat is not "full_predict":
                tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta)
            tr_temporal_tot.append(tr_temp)
        tr_temporal_tot = np.array([y for x in tr_temporal_tot for y in x])
        print("the shape of concatenate tr path", np.shape(tr_temporal_tot))
    elif data_set is "shanghaitech":
        if "bg_index" in tr_im[0]:
            video_index = [v.strip().split('video_')[1].strip().split('_frame_')[0] for v in tr_im]
            num_test_video = np.shape(np.unique(video_index))[0]
            name_space = np.unique(video_index)
            name_space = ['video_' + v + '_frame_' for v in name_space]
            bg_index_name = tr_im[0].strip().split('training/')[1].strip().split('/video')[0]
            bg_im_mom = tr_im[0].strip().split('shanghaitech')[0]
            bg_im_path = bg_im_mom + 'shanghaitech_%s_mean.jpg' % bg_index_name
        else:
            video_index = [v.strip().split('frames/')[1].strip().split('/')[0] for v in tr_im]
            num_test_video = np.shape(np.unique(video_index))[0]
            name_space = np.unique(video_index)
            name_space = [v + '/' for v in name_space]
            bg_im_path = None
        print("There are %d videos for dataset %s" % (num_test_video, name_space))
        tr_temporal_tot = []
        for single_index in range(num_test_video):
            name_use = name_space[single_index]
            tr_subset = sorted([v for v in tr_im if name_use in v],
                               key=lambda s: int(s.strip().split(name_use)[-1].strip().split('.jpg')[0]))
            tr_subset = np.array(tr_subset)
            tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta,
                                                               bg=bg_im_path)
            tr_temporal_tot.append(tr_temp)
        tr_temporal_tot = np.array([y for x in tr_temporal_tot for y in x])
        print("the shape of concatenate tr path", np.shape(tr_temporal_tot))
    else:
        tr_subset = tr_im
        tr_temp, in_shape, out_shape = read_frame_interval(tr_subset, time_step, concat, interval, delta)
        tr_temporal_tot = np.array(tr_temp)
    return tr_temporal_tot, in_shape, out_shape


def read_frame_interval(all_cat, time_step, concat, interval, delta, bg=None, neg=False):
    all_cat_new = []
    for single_interval in interval:
        all_cat_use = all_cat
        num_cat = np.shape(all_cat_use)[0]
        crit = single_interval * (time_step - 1) + delta
        for i, v in enumerate(all_cat_use):
            if i + crit < num_cat:
                init = np.linspace(i, i + single_interval * (time_step - 1), time_step, dtype='int32')
                if neg is False:
                    end = init[-1] + delta
                else:
                    big_num = init[-1] + num_cat // 10
                    small_num = init[0] - num_cat // 10
                    tot_num = []
                    if big_num <= num_cat:
                        tot_num.append(np.arange(big_num, num_cat))
                    if small_num >= 0:
                        tot_num.append(np.arange(0, small_num))
                    tot_num = [int(v) for j in tot_num for v in j]
                    end = np.random.choice(tot_num, 1, replace=False)
                if bg:
                    tt = np.concatenate([[all_cat_use[end]],
                                         [bg],
                                         ['0' for i in range(time_step - 2)]], axis=0)
                else:
                    tt = np.concatenate([[all_cat_use[end]],
                                         ['0' for i in range(time_step - 1)]], axis=0)
                all_cat_new.append([all_cat_use[init], tt])
    inp_shape = np.shape(all_cat_new[0][0])
    oup_shape = np.shape(all_cat_new[1][0])
    return all_cat_new, inp_shape, oup_shape


def rgb2hls(image):
    image = np.array(image)
    im = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return im


def hls2rgb(image):
    image = np.array(image)
    im = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    return im


def darker_tf(image, darker_value):
    """this function is a tensorflow version of darker image.
    The image can be [num_frame,batch_size, imh, imw, ch], or just [batch_size,imh,imw,ch]
    or only [imh, imw, ch]
    It needs to be float32, in the range of [0,1] [rgb]
    1) Transfer the image to hsv
    2) unscatter based on the last channel v, and multiple v with the darker value
    3) Then clip the last to make sure the maximum value is 1.0, and min is 0.0
    4) concate them based on the last channel
    5) transfer it back to rgb
    """
    hls_image = tf.py_function(rgb2hls, inp=[image], Tout=tf.uint8)
    hls_image = tf.cast(hls_image, tf.float32)
    hls_image = tf.unstack(hls_image, num=3, axis=-1, name="unstack_hls")
    v = tf.multiply(hls_image[1], darker_value)
    v = tf.clip_by_value(v, clip_value_min=0.0, clip_value_max=255.0)
    hls_image[1] = v
    hls_new = tf.stack(hls_image, axis=-1, name="stack_hls")
    hls_new = tf.cast(hls_new, tf.uint8)
    rgb_new = tf.py_function(hls2rgb, inp=[hls_new], Tout=tf.uint8)
    return rgb_new


def small_func(v, targ_shape, augment_option, darker_value, data_set, apply_bi):
    v = tf.image.decode_jpeg(tf.read_file(v), channels=3)
    if "add_dark" in augment_option:
        v = darker_tf(v, darker_value)
    if data_set is not "moving_mnist" and apply_bi is True:
#    if targ_shape[0] != 64:
        v = tf.expand_dims(tf.cast(v, tf.float32), axis=0)
        v = tf.image.resize_bilinear(tf.divide(v, 255.0), targ_shape)
        v = tf.squeeze(v, axis=0)
    else:
        v = tf.divide(tf.cast(v, tf.float32), 255.0)
    v = tf.reshape(v, [targ_shape[0], targ_shape[1], 3])
    return v


def concat_im(model_mom, data_set, x, inp_shape, oup_shape, targ_shape, augment_option, darker_value, learn_opt,
              conc_option, train_index, apply_bi):
    """this function is used to concatenate the input and output tensors
    Args:
        model_mom: the path to load the mean for each dataset, as same as the
        input for the get_video_data function
        data_set: str, "ucsd1","ucsd2", "avenue", "shanghaitech","kth"
        inp_shape and oup_shape: [num_input_frame],[num_output_frame]
        targ_shape: [imh, imw] for the input. array
        learn_opt: str, "learn_fore" or "learn_full". "learn_fore" means we subtract
        the background from the input frame. "learn_full" means the input frame
        of the model is the original frame
        conc_option: "conc_tr"
        train_index: str, "bg_index_01"..."bg_index_12"
    Output:
        im_tot: [input_im, output_im, data_mean]
    """
    inp = tf.reshape(x[0], inp_shape)
    oup_tot = tf.reshape(x[1], oup_shape)
    oup = oup_tot[:1]
    if "shanghaitech" in data_set and train_index == None:
        bg_tensor = oup_tot[1:2]
    if augment_option == "add_dark_auto":
        darker_space = [1.6, 1.0, darker_value]
        darker_value = tf.random.shuffle(darker_space)[0]
    elif augment_option == "add_dark_manu":
        darker_value = darker_value
    elif augment_option == "none":
        darker_value = 0.0

    inp_im = tf.map_fn(lambda v: small_func(v, targ_shape, augment_option, darker_value, data_set, apply_bi), inp,
                       dtype=tf.float32)
    out_im = tf.map_fn(lambda v: small_func(v, targ_shape, augment_option, darker_value, data_set, apply_bi), oup,
                       dtype=tf.float32)  # num_conc_image, imh, imw, ch
    if "shanghaitech" in data_set and train_index == None:
        print("loading mean from image")
        bg_im = tf.map_fn(lambda v: small_func(v, targ_shape, augment_option, darker_value, data_set, False), bg_tensor,
                          dtype=tf.float32)
        if "learn_fore" in learn_opt:
            inp_im = tf.subtract(inp_im, bg_im)
            out_im = tf.subtract(out_im, bg_im)
        data_mean = bg_im
    if train_index != None or data_set != "shanghaitech":
        if "learn_rest" in learn_opt or "learn_fore" in learn_opt:
            print("loading mean from npy")
            data_mean = calc_mean_std_data(model_mom, data_set, train_index,
                                           targ_shape=targ_shape)
            inp_im = tf.subtract(inp_im, data_mean)
            out_im = tf.subtract(out_im, data_mean)
        else:
            data_mean = tf.constant(0.0, shape=[1, targ_shape[0], targ_shape[1], 1])
    if "ucsd" in data_set or data_set == "kth_walk" or "moving_mnist" in data_set:
        print("converting gray channel to single channel for ucds")
        im_tot = [tf.reduce_mean(inp_im, axis=-1, keep_dims=True), tf.reduce_mean(out_im, axis=-1, keep_dims=True),
                  tf.reduce_mean(data_mean, axis=-1, keep_dims=True)]
    else:
        im_tot = [inp_im, out_im, data_mean]
    if "shanghaitech" in data_set:
        im_tot = [inp_im, out_im, data_mean]
    return im_tot


def calc_mean_std_data(model_mom, data_set, tr_index=None, tensor=True, targ_shape=np.array([128, 196])):
    """This function is for loading the avg image
    Args:
        model_mom: the path to load the mean image, as same as the model mom used in the get_video_data function
        data_set: "ucsd1", "ucsd2", "avenue", "shanghaitech"
        tr_index: the bg index for shanghaitech dataset "bg_index_01"..."bg_index_12"
        tensor: bool True/False. If True, the loaded mean is tensor, else npy
        targ_shape: the shape of the input to the model
    Output:
        avg: targ_shape
    """
    path_mom = model_mom
    if data_set is not "shanghaitech":
        path2read = "gt/%s_mean_%d_%d.npy" % (data_set, targ_shape[0], targ_shape[1])
    else:
        path2read = path_mom + "%s_%s_mean.npy" % (data_set, tr_index)
    mean_value = np.load(path2read)
    mean_value = np.expand_dims(mean_value, axis=0)  # [batch_size, imh, imw, ch] ch=1/3
    if tensor == True:
        return tf.constant(mean_value, dtype=tf.float32)
    else:
        return mean_value


def dataset_input(model_mom, data_set, im_filename, learn_opt, temp_shape, imshape, targ_shape, batch_size, conc_option,
                  augment_option="none", darker_value=0.0, shuffle=True, train_index=None, epoch_size=1):
    """This function is used to read the images
    Args:
        model_mom: path to read the mean image
        im_filename: tensor or array of filenames, with shape: [N, 2, k]
        learn_opt: str, "learn_fore"/"learn_full"
        temp_shape: [inp_shape, oup_shape]
        targ_shape: [imh, imw]
        batch_size: int
        conc_opt: "conc_tr"
        shuffle: bool
        train_index: the bg index for shanghaitech dataset
        epoch: the number of repeatition for the filenames for each epoch, default 1
    return:
        image: [batch_size, time_step, imh, imw, ch]
    """
    inp_shape, oup_shape = temp_shape
    images = tf.convert_to_tensor(im_filename, dtype=tf.string)
    if shuffle == True:
        images = tf.random.shuffle(images)
    transform1 = tf.data.Dataset.from_tensor_slices(images)
    transform1 = transform1.repeat(epoch_size)
    apply_bi = True
    if "shanghaitech" in data_set and shuffle == False:
        apply_bi = True
    if imshape[0] == targ_shape[0]:
        apply_bi = False
    print("I am applying bilinear operatioin", apply_bi)

    transform2 = transform1.map(lambda x: concat_im(model_mom,
                                                    data_set,
                                                    x, inp_shape, oup_shape, targ_shape=targ_shape,
                                                    augment_option=augment_option,
                                                    darker_value=darker_value,
                                                    learn_opt=learn_opt,
                                                    conc_option=conc_option,
                                                    train_index=train_index,
                                                    apply_bi=apply_bi))
    transform2 = transform2.apply(tf.data.experimental.ignore_errors())

    im_out = transform2.batch(batch_size, drop_remainder=True)

    return im_out


def calculate_mean_sigma(image, targ_shape):
    """image_shape: [batch_size, num_im,  targ_h, targ_w, 3]
    return mean, std [batch_size, num_im], [batch_size, num_im]
    """
    batch_size = targ_shape[0]
    num_input = image.get_shape().as_list()[1]
    mean, var = tf.nn.moments(image, axes=(-1, -2, -3))
    std = tf.expand_dims(tf.sqrt(var), axis=-1)
    targ_prod = np.prod(targ_shape[1]).astype('float32')
    im_element = tf.constant(value=1.0 / np.sqrt(targ_prod), shape=[batch_size, num_input, 1], dtype=tf.float32,
                             name="pen")
    std_final = tf.reduce_max(tf.concat([std, im_element], axis=-1), axis=-1)
    return tf.reshape(mean, [batch_size, num_input, 1, 1, 1]), tf.reshape(std_final, [batch_size, num_input, 1, 1, 1])


def return_crop_im_back_original_im(cropped_image, crop_shape, num_crop_h_w, stride_size, targ_shape):
    """this function is to return the cropped image back to the original image scale
    cropped_image: [num_frame, num_box, batch_size, crop_h, crop_w, ch]
    targ_shape: [num_frame, batch_size, fh,fw, ch]
    """
    num_crop_h, num_crop_w = num_crop_h_w
    stride_h, stride_w = stride_size
    crop_h, crop_w = crop_shape
    imh, imw = targ_shape[-3:-1]
    im_empty = np.zeros(targ_shape)
    cropped_image = np.transpose(cropped_image, (1, 0, 2, 3, 4, 5))
    for i in range(num_crop_h):
        if i != num_crop_h - 1:
            start_h = i * stride_h
        else:
            start_h = imh - crop_h
        for j in range(num_crop_w):
            im_index = j + i * num_crop_w
            crop_im_use = cropped_image[im_index]
            if j != num_crop_w - 1:
                start_w = j * stride_w
            else:
                start_w = imw - crop_w
            im_empty[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w, :] = crop_im_use
    return im_empty


def get_crop_image(image, crop_h, crop_w):
    """this function is for getting the cropped image
    image: [num_frame, imh, imw, ch]
    crop_h, int define the height of the cropped image
    crop_w, int, define the width of the cropped image
    """
    imh, imw, ch = image.get_shape().as_list()[-3:]
    stride_h, stride_w = int(np.ceil(crop_h / 2)), int(np.ceil(crop_w / 2))
    num_crop_h, num_crop_w = int(np.floor((imh - 1) / stride_h)), int(np.floor((imw - 1) / stride_w))
    im_tot = []
    for i in range(num_crop_h):
        if i != num_crop_h - 1:
            tl_h = i * stride_h
        else:
            tl_h = imh - crop_h
        for j in range(num_crop_w):
            if j != num_crop_w - 1:
                tl_w = j * stride_w
            else:
                tl_w = imw - crop_w
            im_crop = tf.image.crop_to_bounding_box(image, tl_h, tl_w, crop_h, crop_w)
            im_tot.append(im_crop)
    return im_tot, [stride_h, stride_w], [num_crop_h, num_crop_w]

def save_avenue_gt_frame_level_label():
    """this function is for saving the frame-level label for avenue dataset
    """
    path = "/home/li/gpu/media/data/anomaly_data/Avenue/dataset/testing_label_mask"
    path_all = sorted(os.listdir(path), key=lambda s: int(s.strip().split('_label')[0]))
    gt_tot = []
    for single_path in path_all:
        pixel_label = io.loadmat(path + '/' + single_path)["volLabel"][0]
        gt_vec = [np.sum(v) for v in pixel_label]
        gt_vec = np.array(gt_vec)
        gt_new = (gt_vec != 0).astype('int32')
        print("there are %d normal and %d abnormal frames in video %s" % (
        np.shape(gt_new)[0] - np.sum(gt_new), np.sum(gt_new), single_path))
        gt_tot.append(gt_new)
    gt_tot = np.array(gt_tot)
    print("the shape of gt ", np.shape(gt_tot))
    np.save("/home/li/gpu/media/data/anomaly_data/Avenue/gt_label", gt_tot)
