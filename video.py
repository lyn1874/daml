#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:34:31 2019
This file is used to save the frames from dataset UCSDPed1, UCSDPed2, Avenue, Shanghaitech and kth
Ops:
    1. Check the path_mom in each function, path_mom needs to be the folder that saves the original
    downloaded dataset
    2. Check the path2write, path2write needs to the path that saves the frames (.jpg)
@author: li
"""
import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('--dataset', type=str, help="Which dataset am I working on? Avenue, UCSDped2")
parser.add_argument('--datapath', type=str, help="Data directory")


def save_ucsd_dataset(path2read):
    """Convert tif images to jpg images for UCSD dataset"""
    path_mom = path2read
    for tr_or_tt in ["Train", "Test"]:
        path = path_mom + tr_or_tt
        path2save = path + '_jpg'
        if not os.path.exists(path2save):
            os.makedirs(path2save)
        all_path = sorted(os.listdir(path))
        all_path = [path + '/' + v for v in all_path if tr_or_tt in v and "gt" not in v]
        for path_iter, single_path in enumerate(all_path):
            all_frames = [v for v in os.listdir(single_path) if '.tif' in v]
            path_name_0 = single_path.strip().split(tr_or_tt + '/')[-1]
            for frame_iter, single_frame in enumerate(all_frames):
                im = cv2.imread(single_path + '/' + single_frame)
                cv2.imwrite(path2save + '/' + path_name_0 + '_' + single_frame.strip().split('.tif')[0] + '.jpg', im)
            print("saving %d images" % frame_iter * (path_iter + 1))


def save_avenue_frame(use_str, path2read):
    """Extract frames from the Avenue dataset
    use_str: "training" or "testing"
    path2read: the path to read the videos from the Avenue dataset
    """
    path2write = path2read + "/frames/%s" % use_str
    path2read = path2read + "%s_videos" % use_str
    print("Reading avenue dataset from ", path2read)
    print("Saving the frames in", path2write)
    if not os.path.exists(path2write):
        os.makedirs(path2write)
    path_child = sorted(os.listdir(path2read))
    tot_num = 0.0
    for iterr, single_child in enumerate(path_child):
        video_path = path2read + '/' + single_child
        s_video_name = single_child.split('.avi')[0]
        cap = cv2.VideoCapture(video_path)
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(path2write + '/%s_video_%s_frame_%05d.jpg' % (use_str, s_video_name, i), frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        tot_num += i
        print("There are %d frames for video %s" % (i, single_child))
        

def save_shanghaitech_data(path2read):
    """This function is for saving the shanghaitech frames
    Note: you need to change the path2read and path2write
    """
    path2write = path2read.split("training/")[0]+"frames/training"
#     path2read = '/project_scratch/bo/anomaly_data/shanghaitech/training/videos'
#     path2write = '/project_scratch/bo/anomaly_data/shanghaitech/frames/training'
    if not os.path.exists(path2write):
        os.makedirs(path2write)
    all_tr_video = os.listdir(path2read)
    train_index = []
    for i in range(13)[1:]:
        train_index.append("%02d_" % i)
    for single_index in train_index:
        video_subset = [v for v in all_tr_video if single_index in v]
        video_subset = sorted(video_subset, key=lambda s: int(s.strip().split('_')[1].lstrip("0").split('.avi')[0]))
        path2save_mom = path2write + '/bg_index_' + single_index.strip().split('_')[0]
        if not os.path.exists(path2save_mom):
            os.makedirs(path2save_mom)
        tot_num = 0
        for single_video in video_subset:
            video_path = path2read + '/' + single_video
            video_name = single_video.strip().split('.avi')[0]
            path2save_subset = path2save_mom + '/video_index_' + video_name.strip().split('_')[1]
            cap = cv2.VideoCapture(video_path)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = (cv2.resize(frame / 255.0, dsize=(224, 128)) * 255.0).astype('uint8')
                cv2.imwrite(path2save_subset + '_frame_%05d.jpg' % i, frame)
                i += 1
            cap.release()
            cv2.destroyAllWindows()
            tot_num += i
        print("there are %d frames for background index %s" % (tot_num, single_index))



if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset == "Avenue":
        save_avenue_frame("training", args.datapath)
        save_avenue_frame("testing", args.datapath)
    elif "UCSD" in args.dataset:
        save_ucsd_dataset(args.datapath)
    elif "shanghaitech" in args.dataset:
        save_shanghaitech_data(args.datapath)
    else:
        print("----The required dataset does not exist yet----")


