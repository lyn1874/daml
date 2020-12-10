#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 17:33 2010
This script list all the hyperparameters
It includes args and also the learning rate
@author: li
"""
import argparse


def give_learning_rate_for_init_exp(args):
    if "ucsd" in args.data_set:
        epoch = 50
        if args.num_encoder_layer == 4:
            lrate_g_step, lrate_z_step = 25, 25
        elif args.num_encoder_layer == 5:
            lrate_g_step, lrate_z_step = 18, 18 
        batch_size = 10
    elif args.data_set is "avenue":
        epoch = 50
        if "crop" in args.input_option:
            epoch = 40
            batch_size = 1
        lrate_g_step, lrate_z_step = 25, 20
        batch_size = 10
    elif "kth_walk" in args.data_set:
        epoch = 50
        lrate_g_step, lrate_z_step = 25, 25
        batch_size = 10
    elif "moving_mnist" in args.data_set:
        epoch = 25
        lrate_g_step, lrate_z_step = 15, 10
        batch_size = 100
    elif "shanghaitech" in args.data_set:
        if not args.bg_index_pool:
            epoch = 30
            lrate_g_step, lrate_z_step = 15, 10 
        else:
            epoch = 50
            lrate_g_step, lrate_z_step = 25, 20
        batch_size = 10
    if args.model_type is "many_to_one":
        lrate_g_step = epoch

    lrate_g = 0.0001
    lrate_z = 0.0001
    args.batch_size = batch_size

    print("------------------------------------------------------")
    print("------------The training statistics-------------------")
    print("------------------------------------------------------")
    print("----The dataset-----", args.data_set)
    print("----There are %d epochs in total with learning the AE for %d epochs and motion for %d epochs" % (epoch, 
                                                                                                            lrate_g_step, 
                                                                                                            lrate_z_step))
    print("----The initial learning rate for AE is %.5f" % lrate_g)
    print("----The initial learning rate for motion is %.5f" % lrate_z)
    print("----The batch size is %d" % batch_size)

    return [lrate_g_step, lrate_g], [lrate_z_step, lrate_z], [epoch, batch_size]


def give_args():
    """This function is used to give the argument"""
    parser = argparse.ArgumentParser(description='Decoupled Appearance and Motion Learning for Anomaly Detection')

    parser.add_argument('-ds', '--data_set', type=str, default="avenue", metavar='DATA_SET',
                        help='dataset')
    # parser.add_argument('-bs', '--batch_size', type=int, default=4, metavar='BATCH_SIZE',
    #                     help='input batch size for training (default: 100)')
    # parser.add_argument('-ep', '--max_epoch', type=int, default=50, metavar='EPOCHS',
    #                     help='maximum number of epochs')
    parser.add_argument('-od', '--output_dim', type=int, default=3, metavar='OUTPUT_DIM',
                        help='the output dimension')
#     parser.add_argument('-ts', '--time_step', type=int, default=6, metavar='TIME_STEP',
#                         help='the number of input frames')
#     parser.add_argument('-in', '--single_interval', type=int, default=2, metavar='SINGLE_INTERVAL',
#                         help='the gap between every two frames')
#     parser.add_argument('-de', '--delta', type=int, default=6, metavar='DELTA',
#                         help='the gap between last input frame and output frame')
    
    parser.add_argument('--datadir', type=str, 
                        help="the location of the dataset, i.e., /project/bo/anomaly_data/")
    parser.add_argument('--expdir', type=str,
                        help="the location of the model ckpts")    
    parser.add_argument('--version', help="experiment versions", type=int)

#     parser.add_argument('-ne', '--num_encoder_layer', type=int, default=4, metavar='NUM_ENCODER_LAYER',
#                         help='the number of encoder layers')
#     parser.add_argument('-nd', '--num_decoder_layer', type=int, default=4, metavar='NUM_DECODER_LAYER',
#                         help='the number of decoder layers')

    parser.add_argument('--model_type', type=str, default="2d_2d_pure_unet", 
                        help="what kind of autoencoder am I using")
    parser.add_argument('--motion_method', type=str, default="conv3d",
                        help='what kind of motion model am I using, conv3d, convlstm')

    parser.add_argument('-ao', '--augment_option', type=str, default="none", metavar='AUGMENT_OPTION',
                        help='whether to apply the augmention on the training data')
    parser.add_argument('-dv', '--darker_value', type=float, default=1.0, metavar='DARKER_VALUE',
                        help='the degree of brightness, 0.0 means using original frames')
    parser.add_argument('-dt', '--darker_type', type=str, default="none", metavar='DARKER_TYPE',
                        help='the darker type, either auto or manu')    
    parser.add_argument('-io', '--input_option', type=str, default="original", metavar='INPUT_OPTION',
                        help='whether the input for the model is cropped or not')


    parser.add_argument('-regu', '--regu_par', type=float, default=0.001, metavar='L2_REGU_PAR',
                        help='L2 regularization parameter')
    parser.add_argument('-zmr', '--z_mse_ratio', type=float, default=0.001, metavar='Z_MSE_RATIO',
                        help='weight for latent space mse loss')
    parser.add_argument('-lo', '--learn_opt', type=str, default='learn_fore', metavar='LEARN_OPT',
                    help='Whether the background frame needs to be subtracted from the original frame')

    parser.add_argument('-sw', '--sliding_window', type=int, default=200, metavar='SLIDING_WINDOW',
                    help='The sliding window size for shanghaitech evaluation')
    parser.add_argument('-em', '--eval_method', type=str, default="future", metavar='EVAL_METHOD',
                    help='The evaluation method for shanghaitech')
    parser.add_argument('--opt', type=str, default="save_score", 
                        help="only used at test time, what kind of operation am I doing there? check_pred, check_recons, save_score")

    return parser.parse_args()


args = give_args()

