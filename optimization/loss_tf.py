#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:19:52 2019
This is the tensorflow version of these loss functions
@author: li
"""
import tensorflow as tf
import numpy as np


def calculate_cosine_dist(gen_frames, gt_frames, batch_size):
    """Since the tensorflow implementation require the input to be unit-normalized,
    I first need to normalize the ground truth latent space, and the predicted latent 
    space, then I can pass them to the tf.loss.cosine_dist function
    return:
        [num_frame, batch_size]
    """
#     num_frame, batch_size = gen_frames.get_shape().as_list()[:2]
    gen_frames = tf.reshape(gen_frames, [1, batch_size, -1])
    gt_frames = tf.reshape(gt_frames, [1, batch_size, -1])
    gen_frames = tf.divide(gen_frames, tf.sqrt(tf.reduce_sum(tf.square(gen_frames), axis=-1, keep_dims=True)))
    gt_frames = tf.divide(gt_frames, tf.sqrt(tf.reduce_sum(tf.square(gt_frames), axis=-1, keep_dims=True)))
    loss = tf.losses.cosine_distance(gt_frames, gen_frames, axis=-1, reduction="none")
    return loss


def test_cosine_loss_func():
    from scipy.spatial.distance import cosine
    gen_frame = np.random.random([2, 4, 4, 3])
    gt_frame = np.random.random([2, 4, 4, 3])
    gen_re = np.reshape(gen_frame, [2, -1])
    gt_re = np.reshape(gt_frame, [2, -1])
    cosine_loss = []
    for single_gen, single_gt in zip(gen_re, gt_re):
        _single_cosine = cosine(single_gt, single_gen)
        cosine_loss.append(_single_cosine)
    # ---below is the tensorflow part---#
    gen_frame_tf = tf.constant(gen_frame, dtype=tf.float32)
    gt_frame_tf = tf.constant(gt_frame, dtype=tf.float32)
    cos_loss_tf = calculate_cosine_dist(gen_frame_tf, gt_frame_tf)
    with tf.Session() as sess:
        cos_loss_npy_from_tf = sess.run(fetches=cos_loss_tf)
    print("------numpy version------")
    print(np.mean(cosine_loss))
    print("-------tensorflow version----")
    print(cos_loss_npy_from_tf)
    print(np.shape(cos_loss_npy_from_tf))
    print(np.mean(cos_loss_npy_from_tf))


def train_op(tot_loss, lr, var_opt, name):
    """
    training optimizer
    """
    #    optimizer = tf.train.RMSPropOptimizer(learning_rate = lr)
    epsilon = 1e-4  # added on 18th of July
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon, name=name)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(tot_loss, var_list=var_opt)
        print("================================================")
        print("I am printing the non gradient")
        for grad, var in grads:
            if grad is None:
                print("no gradient", grad, var)
        opt = optimizer.apply_gradients(grads)
    return opt
