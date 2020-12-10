#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:30:15 2019
This script is for doing everything without using tensorflow
@author: li
"""
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def read_test_index(model_mom, data_set):
    if "ucsd" in data_set:
        if data_set == "ucsd1":
            path2read = model_mom + 'UCSDped1/Test_jpg'
            gt_path = "gt/UCSDped1_gt.npy"
        elif data_set == "ucsd2":
            path2read = model_mom + 'UCSDped2/Test_jpg'
            gt_path = "gt/UCSDped2_gt.npy"
        tt_ind = [v.strip().split('_')[0] for v in os.listdir(path2read) if '.jpg' in v]
        tt_ind = sorted(np.unique(tt_ind))
        test_index_use = tt_ind
    elif data_set == "avenue":
        test_index_use = ["testing_video_%d_" % i for i in range(22)[1:]]
        gt_path = "gt/Avenue_gt.npy"
    elif "shanghaitech" in data_set:
        path2read = model_mom + 'shanghaitech/original/testing/test_frame_mask'
        gt_all = sorted(os.listdir(path2read))
        gt_all = [path2read + '/' + v for v in gt_all]
        gt = []
        for single_gt in gt_all:
            gt.append(np.load(single_gt))
        test_index_use = [v.strip().split("test_frame_mask/")[1].strip().split('.npy')[0] + '/' for v in gt_all]
    if "shanghaitech" not in data_set:
        gt = np.load(gt_path, allow_pickle=True)
    return test_index_use, gt


def calc_psnr(tds_dir, single_test_index, ano_score):
    max_value = np.load(tds_dir + '/psnr_%s.npy' % single_test_index)
    max_value_use = max_value[1]
    ano_score_update = 10 * np.log10(max_value_use / ano_score)
    return ano_score_update


def get_model_mom(shared, home):
    # print("-------------------------------------------------------------------------")
    # print("----------NOTE: THE PATH NEEDS TO BE USER-DEFINED------------------------")
    # print("-------------------------------------------------------------------------")

    if shared is "pure_project":
        path_mom = "/project/bo/"
    elif shared is "gpu_users":
        path_mom = "/media/data/"
    elif shared is "gpu_project":
        path_mom = "/groups/wall2-ilabt-iminds-be/dianne/bo/"
    elif shared is "mac_gpu":
        path_mom = "/Users/bo/gpu/media/data/"
    model_mom_for_load_data = path_mom + 'anomaly_data/'
    if "project" in shared:
        path_mom = path_mom + 'exp_data/'
    if home is False:
        if shared is "gpu_users" or shared is "gpu_project":
            path_mom = "/home/li/gpu" + path_mom
            model_mom_for_load_data = "/home/li/gpu" + model_mom_for_load_data

    return model_mom_for_load_data, path_mom


def compare_mse_cos_for_avenue(shared, home, version):
    accu, threshold_use = check_accu_for_each_video(shared, home, "avenue", 6, 2, 2, 4, [0, 1], version)
    print(threshold_use)
    accu_all, gt_box_tot, ano_score = accu
    num_test_video = np.shape(accu_all)[0]
    color_group = ['r', 'g']
    title_space = ['z-mse', 'z-cos']
    for iterr in range(num_test_video):
        accu_sub = accu_all[iterr]  # [2,2]
        print(np.shape(accu_sub), accu_sub)
        gt_box_subset = gt_box_tot[iterr]  # []
        ano_score_subset = ano_score[iterr]  # [2, num_frame]
        row_ind = [iterr for iterr, v in enumerate(gt_box_subset) if v > 0]
        num_frame = np.shape(ano_score_subset)[1]
        print(iterr, [np.max(v) for v in ano_score_subset])
        fig = plt.figure(figsize=(7, 2.5))
        for j in range(2):
            if j == 0:
                gt_box_subset_use = gt_box_subset / (np.max(gt_box_subset)) * 1e-3
            else:
                gt_box_subset_use = gt_box_subset / np.max(gt_box_subset) * 0.5
            gt_box_subset_use = [v if v > 0 else None for v in gt_box_subset_use]

            ax = fig.add_subplot(1, 2, j + 1)
            ax.plot(np.arange(num_frame), ano_score_subset[j], color_group[0] + '.', markersize=5)
            ax.plot(np.arange(num_frame), gt_box_subset_use, 'b.', markersize=5)
            ax.plot(np.arange(num_frame)[row_ind], ano_score_subset[j][row_ind], color_group[1] + '.', markersize=5)
            ax.plot(np.arange(num_frame), np.repeat(threshold_use[j], num_frame), color_group[0])
            ax.grid(ls=':', alpha=0.5)
            ax.set_title(title_space[j] + 'tpr %.2f fpr %.2f' % (accu_sub[j][0] * 100, accu_sub[j][1] * 100),
                         fontsize=10)


def check_accu_for_each_video(shared, home, data_set, time_step, delta, single_interval, num_enc_layer, index_use,
                              version, calc_box_size=False, aug=None):
    path_for_load_data, model_path = get_model_mom(shared, home)
    test_index_all, gt = read_test_index(path_for_load_data, data_set)
    if data_set is "avenue":
        gt_region_path = path_for_load_data + 'Avenue/gt_box/'
        all_gt = os.listdir(gt_region_path)
        all_gt = sorted(all_gt, key=lambda s: int(s.strip().split('_label.npy')[0]))
        all_gt = [gt_region_path + v for v in all_gt]
    else:
        gt_region_path = None
        all_gt = None
    if not aug:
        tds_dir_mom = model_path + 'ano_%s_motion_end2end/tds/' % data_set
        use_mark = "pred_score"
    tds_dir_single = tds_dir_mom + 'time_%d_delta_%d_gap_%d_2d_2d_pure_unet_conv3d_learn_fore_enc_%d_version_%d' % (
                                    time_step, delta, single_interval,
                                    num_enc_layer, version)
    if aug:
        tds_dir_single = tds_dir_single + '/%s_%s' % (aug[0], aug[1])
    if aug:
        threshold = np.load(tds_dir_single + '/opt_threshold_%s_%.1f.npy' % (aug[0] + '_' + aug[1], aug[-1]))
    else:
        threshold = np.load(tds_dir_single + '/opt_threshold_no_norm.npy')
    threshold_use = [threshold[index_use[0] + 2], threshold[index_use[1] + 2]]
    accu = give_auc_score_for_per_video(data_set, tds_dir_single, use_mark, test_index_all, all_gt, index_use,
                                        threshold_use, calc_box=calc_box_size)
    #    if gt_region_path:
    #        accu, acc_for_diff_crit, box_avg_size, num_box = accu
    return accu, threshold_use


def give_auc_score_for_per_video(data_set, single_tds, use_mark, test_index, gt_box_path, index_use, threshold_use,
                                 calc_box=False, gt=None):
    """this function is used to calculate the auc score for per video given the loaded threshold
    single_tds: the path mom for the model
    use_mark: "recons_score" or "pred_score"
    use_mark: include all the name before the test_index, such as recons_score_add_rain_torrential_0.1_
    
    test_index: the loaded test_index
    gt_box_path: shape is as same as test_index
    index_use: [0, -2]. it defines which criteria do I want to check. 0 is the z mse, -2 is the prediction pixel-mse
    """
    ano_score_tot = []
    gt_tot = []
    gt_box_size_tot = []
    accu_all_video = []
    legend_tot = ["z-mse", "z-cos", "z-l1norm", "p-mse", "p-psnr"]
    legend = [legend_tot[i] for i in index_use]

    for test_index_iter, test_index_use in enumerate(test_index):
        ano_score = np.load(single_tds + '/%s_%s.npy' % (use_mark, test_index_use))
        ano_score_subset = [ano_score[:, single_iter] for single_iter in index_use]
        if gt_box_path:
            gt_subset = np.load(gt_box_path[test_index_iter])[-np.shape(ano_score)[0]:]
            gt_box_size = np.array([np.sqrt((v[2] - v[0]) * (v[3] - v[1])) for v in gt_subset])
            gt_label = (gt_box_size != 0).astype('int32')
        else:
            gt_label = gt[test_index_iter][-np.shape(ano_score)[0]:]
            gt_box_size = []
        accu_per_video = [give_tp(ano_score_subset[iterr], gt_label, threshold_use[iterr], data_set) for iterr in
                          range(np.shape(index_use)[0])]
        gt_box_size_tot.append(gt_box_size)
        ano_score_tot.append(ano_score_subset)
        gt_tot.append(gt_label)
        accu_all_video.append(accu_per_video)

    if calc_box is True:
        gt_box_vector = [v for j in gt_box_size_tot for v in j]
        gt_tot_vector = [v for j in gt_tot for v in j]

        accuracy_for_diff_crit = []

        for index_iter, single_index in enumerate(index_use):
            ano_score_subset = [j[index_iter] for j in ano_score_tot]
            ano_score_subset = [v for j in ano_score_subset for v in j]

            accuracy, box_avg_size, num_frame_per_box = detect_accu_bins(np.array(ano_score_subset),
                                                                         threshold_use[index_iter],
                                                                         np.array(gt_box_vector),
                                                                         np.array(gt_tot_vector))
            _fp, _tp, _threshold = roc_curve(np.array(gt_tot_vector),
                                             np.array(ano_score_subset))
            print("use %s get auc %.2f" % (legend[index_iter], auc(_fp, _tp) * 100))
            accuracy_for_diff_crit.append(accuracy)
        return accu_all_video, accuracy_for_diff_crit, box_avg_size, num_frame_per_box
    else:
        return accu_all_video, gt_box_size_tot, ano_score_tot


def give_tp(pred_prob, gt_label, threshold, data_set):
    if data_set is not "avenue":
        pred_prob = (pred_prob - np.min(pred_prob)) / (np.max(pred_prob) - np.min(pred_prob))

    pred_label = (pred_prob >= threshold).astype('int32')
    tp = [1 for v, j in zip(pred_label, gt_label) if v == 1 and j == 1]
    fp = [1 for v, j in zip(pred_label, gt_label) if v == 1 and j == 0]
    tpr = np.sum(tp) / np.sum(gt_label)
    fpr = np.sum(fp) / (np.shape(gt_label)[0] - np.sum(gt_label))
    print(np.sum(tp), np.sum(fp), np.sum(gt_label), np.shape(gt_label)[0] - np.sum(gt_label), np.shape(gt_label)[0])
    return tpr, fpr


def detect_accu_bins(prob, threshold, box_size, gt_label):
    pred_label = (prob >= threshold).astype('int32')
    box_group = np.linspace(0, np.max(box_size), 20)
    accu = []
    box_avg_size = []
    frame_tot = []
    for single_index in np.arange(np.shape(box_group)[0])[:-1]:
        row = [iterr for iterr, v in enumerate(box_size) if
                v > box_group[single_index] and v <= box_group[single_index + 1]]
        num_frame = np.shape(row)[0]
        if num_frame != 0:
            tt = [1 for _p, _g in zip(pred_label[row], gt_label[row]) if _p == 1 and _g == 1]
            num_anomalous = [1 for _p in gt_label[row] if _p == 1]
            _acc_ = np.sum(tt) / np.sum(num_anomalous)
            _avg_size_ = box_group[single_index] + (box_group[single_index + 1] - box_group[single_index]) / 2
            _num_frame = np.sum(num_anomalous)
        else:
            _acc_ = 0.0
            _avg_size_ = 0.0
            _num_frame = 0.0
        accu.append(_acc_)
        box_avg_size.append(_avg_size_)
        frame_tot.append(_num_frame)

    return accu, box_avg_size, frame_tot



def get_auc_score_efficient(path_for_load_data, tds_dir_single, data_set, stat):
    test_index_all, gt = read_test_index(path_for_load_data, data_set)
    gt_tot = []
    ano_score_tot = []
    ano_value = 0.0
    if "shanghaitech" in data_set:
        test_index_all = [v.strip().split('/')[0] for v in test_index_all]
    print("There are %d test videos" % np.shape(test_index_all)[0])
    for iterr, single_test_index in enumerate(test_index_all):
        ano_score = np.load(tds_dir_single + '/latent_cos_%s.npy' % single_test_index)
        ano_value += np.mean(ano_score)
        if data_set != "avenue":
            ano_score = (ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score))
        _len_gt = len(gt[iterr]) - stat[0] * stat[2] - stat[1] + 2
        gt_tot.append(gt[iterr][-_len_gt:][:np.shape(ano_score)[0]])
#         gt_tot.append(gt[iterr][-np.shape(ano_score)[0]:])
        ano_score_tot.append(ano_score)
    gt_vec = np.array([v for j in gt_tot for v in j])
    ano_vec = np.array([v for j in ano_score_tot for v in j])
    _fpr, _tpr, _ = roc_curve(gt_vec, ano_vec)
    auc_ = auc(_fpr, _tpr)
    eer = _fpr[np.nanargmin(np.absolute((_fpr + _tpr - 1)))]
    print("reporting anomaly detection accuracy on dataset %s" % data_set)
    print("======================================================")
    print("auc score is %.4f with prediction error %.4f with equal error rate %.4f" % (auc_, ano_value, eer))
    return auc_



def get_auc_score_end2end(tds_dir_single, path_for_load_data, data_set, single_interval, 
                          show=True, aug=None, N=20,
                          method="history", return_stat=False):
    test_index_all, gt = read_test_index(path_for_load_data, data_set)
    if "shanghaitech" in data_set:
        test_index_all = [v.strip().split('/')[0] for v in test_index_all]
    #    print("There are %d test videos"%np.shape(test_index_all)[0])
    gt_start = single_interval
    auc_score_tot = []
    opt_threshold = []
    pred_legend = ["z-mse", "z-cos", "z-l1norm", "pred-mse", "pred-psnr"]
    recons_legend = ["recons-mse", "recons-psnr"]
    legend_space = np.concatenate([recons_legend, pred_legend], axis=0)
    error = []
    if "2d_2d_unet_no_shortcut" in tds_dir_single:
        mark_group = ["pred_score"]
    elif "many_to_one" in tds_dir_single:
        mark_group = ["pred_score"]
    else:
        mark_group = ["recons_score", "pred_score"]

    for use_mark in mark_group:
        ano_score_tot, gt_tot = [], []
        error_per_stat = []
        for iterr, single_test_index in enumerate(test_index_all):
            if not aug:
                ano_score_single_test_video = np.load(
                    tds_dir_single + '/%s_%s.npy' % (use_mark, single_test_index))
            else:
                if aug[0] is "add_rain":
                    tds_use = tds_dir_single + '%s_%s_%.1f_%s.npy' % (
                    use_mark, aug[0] + '_' + aug[1], aug[2], single_test_index)
                else:
                    tds_use = tds_dir_single + '%s_%s_%.1f_%s.npy' % (use_mark, aug[0], aug[2], single_test_index)
                ano_score_single_test_video = np.load(tds_use)
            error_per_stat.append(np.mean(ano_score_single_test_video, axis=0))
            ano_score_renew_per_test_video = np.zeros(np.shape(ano_score_single_test_video))
            num_crit = np.shape(ano_score_single_test_video)[1]
            crit_space = np.arange(num_crit) #[:-1]
            if "many_to_one" in tds_dir_single:
                crit_space = [np.arange(num_crit)[-2]]
            for single_use_index in crit_space:
                _ano_score = ano_score_single_test_video[:, single_use_index]
                if single_use_index == num_crit - 1:
                    _ano_score = 10 * np.log10(_ano_score / ano_score_single_test_video[:, single_use_index - 1])
                if data_set == "ucsd1":
                    _ano_score = (_ano_score - np.min(_ano_score)) / (np.max(_ano_score) - np.min(_ano_score))
                if data_set == "ucsd2":
                    _ano_score = (_ano_score - np.min(_ano_score)) / (np.max(_ano_score) - np.min(_ano_score))
                if data_set == "shanghaitech" and use_mark is "pred_score":
                    _ano_score = calc_moving_average(_ano_score, N, method)
                if data_set == "avenue":
                    _ano_score = _ano_score
                if single_use_index == num_crit - 1:
                    _ano_score = 1.0 - _ano_score
                ano_score_renew_per_test_video[:, single_use_index] = _ano_score
            if use_mark is "pred_score":
                gt_per_test_video = gt[iterr][-np.shape(ano_score_single_test_video)[0]:]
            else:
                gt_per_test_video = gt[iterr][gt_start:gt_start + np.shape(ano_score_single_test_video)[0]]
            ano_score_tot.append(ano_score_renew_per_test_video)
            gt_tot.append(gt_per_test_video)
        error.append(np.mean(error_per_stat, axis=0))
        ano_vec = np.array([v for j in ano_score_tot for v in j])
        gt_vec = np.array([v for j in gt_tot for v in j])
        for single_use_index in crit_space:
            gt_temp = gt_vec
            _fpr, _tpr, _threshold = roc_curve(gt_temp, ano_vec[:, single_use_index])
            _auc = auc(_fpr, _tpr)
            optimal_idx = np.argmax(_tpr - _fpr)
            optimal_threshold = _threshold[optimal_idx]
            opt_threshold.append(optimal_threshold)
            auc_score_tot.append(_auc)
    error = [v for j in error for v in j]
    if "2d_2d_unet_no_shortcut" in tds_dir_single:
        error = [error[i] for i in [0, 1, 3]]
    elif "many_to_one" in tds_dir_single:
        error = error 
    else:
        error = [error[i] for i in [0, 2, 3, 5]]
    if show is True:
        auc_score_tot = np.round(np.array(auc_score_tot) * 100, 2)
        print(auc_score_tot)
        print("====================================================================")
        print("{0}         {1}".format("method", "accuracy"))
        print("{0}:    {1}".format("recons-mse", auc_score_tot[0]))
        print("{0}:   {1}".format("recons-psnr", auc_score_tot[1]))
        print("{0}:    {1}".format("latent-mse", auc_score_tot[2]))
        print("{0}:    {1}".format("latent-cos", auc_score_tot[3]))
        print("{0}:     {1}".format("latent-l1", auc_score_tot[4]))
        print("{0}:      {1}".format("pred-mse", auc_score_tot[5]))
        print("{0}:     {1}".format("pred-psnr", auc_score_tot[6]))        
#         [print(_single_legend, np.round(_single_auc * 100, 2)) for _single_legend, _single_auc in
#          zip(legend_space, auc_score_tot)]
        print("====================================================================")

    #    if aug:
    #        np.save(tds_dir_single+'/opt_threshold_%s_%.1f'%(aug[0]+'_'+aug[1], aug[2]), np.array(opt_threshold))
    #    else:
    #        if data_set is "avenue":
    #            np.save(tds_dir_single+'/opt_threshold_no_norm', np.array(opt_threshold))
    #        else:
    #            np.save(tds_dir_single+'/opt_threshold_norm', np.array(opt_threshold))

    return auc_score_tot, error


def calc_moving_average(ano_score, N, method):
    """this function is used to caluclate the normalized score using a sliding window
    ano_score: [number of frames]
    N: define the size of the sliding window
    method: there are three different method, history, future and half_history+half_future
    """
    num_frame = np.shape(ano_score)[0]
    ano_score_new = []
    if N >= num_frame:
        N = num_frame
    if method is "history":
        for i in range(num_frame):
            if i <= N - 1:
                min_ = np.min(ano_score[:N])
                max_ = np.max(ano_score[:N])
            elif i > N - 1:
                min_ = np.min(ano_score[i - N + 1:i + 1])
                max_ = np.max(ano_score[i - N + 1:i + 1])
            ano_score_new.append((ano_score[i] - min_) / (max_ - min_))
    elif method is "future":
        for i in range(num_frame):
            if i <= num_frame - N:
                min_ = np.min(ano_score[i:(i + N)])
                max_ = np.max(ano_score[i:(i + N)])
            else:
                min_ = np.min(ano_score[-N:])
                max_ = np.max(ano_score[-N:])
            #            ano_score_new.append((ano_score[i]-min_)/max_)
            ano_score_new.append((ano_score[i] - min_) / (max_ - min_))
    elif method is "half_half":
        for i in range(num_frame):
            if i <= N // 2:
                min_ = np.min(ano_score[:N])
                max_ = np.max(ano_score[:N])
            elif i >= num_frame - N // 2:
                min_ = np.min(ano_score[-N:])
                max_ = np.max(ano_score[-N:])
            else:
                min_ = np.min(ano_score[(i - N // 2):(i + N // 2)])
                max_ = np.max(ano_score[(i - N // 2):(i + N // 2)])
            ano_score_new.append((ano_score[i] - min_) / (max_ - min_))
    elif method is "mean_future":
        for i in range(num_frame):
            if i <= num_frame - N:
                ano_subset = ano_score[i:(i + N)]
            else:
                ano_subset = ano_score[-N:]
            ano_score_new.append(np.mean(ano_subset))
    return np.array(ano_score_new)











