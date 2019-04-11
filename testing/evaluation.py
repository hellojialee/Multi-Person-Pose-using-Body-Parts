import sys
sys.path.append("..")  # 包含上级目录
import pandas as pd
import json
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tqdm
import cv2
import tensorflow as tf
from keras.utils.multi_gpu_utils import multi_gpu_model

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import util

from config_reader import config_reader
from config import COCOSourceConfig, GetConfig

from keras import backend as K

import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
K.set_learning_phase(1)

# the middle joints heatmap correpondence
limbSeq = [[1, 0], [1, 14], [1, 15], [1, 16], [1, 17], [0, 14], [0, 15], [14, 16], [15, 17],
           [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
           [9, 10], [1, 11], [11, 12], [12, 13], [8, 11], [2, 16], [5, 17]]

mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23],
          [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45],
          [46, 47]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255],
          [128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255]]

#  TODO: when evaluation, change the keypoint id mapping!! Because we use different ids compared with MSCOCO dataset!
dt_gt_mapping = {0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
                 14: 2, 15: 1, 16: 4, 17: 3, 18: None}


def predict(image, params, model, model_params, heat_layers, paf_layers):
    # print (image.shape)
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], heat_layers))
    paf_avg = np.zeros((image.shape[0], image.shape[1], paf_layers))
    multiplier = [x * model_params['boxsize'] / image.shape[0] for x in params['scale_search']]
    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        #
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)


    return heatmap_avg, paf_avg


def find_peaks(heatmap_avg, params):
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)  # TODO: fintune the sigma

        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)  # todo： NMS with a sliding window of 3*3
        map_down[:-1, :] = map[1:, :]
        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]

        map_left_up = np.zeros(map.shape)
        map_left_up[1:, :] = map_left[:-1, :]
        map_right_up = np.zeros(map.shape)
        map_right_up[1:, :] = map_right[:-1, :]
        map_left_down = np.zeros(map.shape)
        map_left_down[:-1, :] = map_left[1:, :]
        map_right_down = np.zeros(map.shape)
        map_right_down[:-1, :] = map_right[1:, :]

        peaks_binary = np.logical_and.reduce((map >= map_left, map >= map_right,
                                              map >= map_up, map >= map_down, map >= map_right_up,
                                              map >= map_right_down,
                                              map >= map_left_up, map >= map_left_down,
                                              map > params['thre1']))  # fixme: finetue it

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    return all_peaks


def find_connections(all_peaks, paf_avg, image_width, params):
    connection_all = []
    special_k = []

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, mapIdx[k][0] // 2]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    mid_num = max(int(norm), 10)  # todo: tune it?
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))] \
                                              for I in range(len(startend))])
                    score_midpts = limb_response

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * image_width / norm - 1, 0)
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48

                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) >= params['connect_ration'] * len(
                        score_midpts)  # todo　改成了0.8
                    # fixme: tune 手动调整, 0.7 or 0.8
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, norm,
                                                     0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][
                                                         2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)  # todo: sort by what

            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):
                i, j, s, limb_len = connection_candidate[c][0:4]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j, limb_len]])
                    if (len(connection) >= min(nA, nB)):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    return connection_all, special_k


def find_people(connection_all, special_k, all_peaks, params):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20, 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    for k in range(len(mapIdx)):

        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k])

            for i in range(len(connection_all[k])):

                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    if subset[j][indexA][0].astype(int) == (partAs[i]).astype(int) or subset[j][indexB][0].astype(
                            int) == partBs[i].astype(int):

                        if found >= 2:
                            print('************ error occurs! 3 joints sharing have been found  *******************')
                            continue
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]

                    if subset[j][indexB][0].astype(int) == -1 and \
                            params['len_rate'] * subset[j][-1][1] > connection_all[k][i][-1]:

                        subset[j][indexB][0] = partBs[i]
                        subset[j][indexB][1] = connection_all[k][i][2]
                        subset[j][-1][0] += 1
                        # last number in each row is the total parts number of that person

                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    elif subset[j][indexB][0].astype(int) != partBs[i].astype(int):
                        if subset[j][indexB][1] >= connection_all[k][i][2]:
                            pass

                        else:
                            if params['len_rate'] * subset[j][-1][1] <= connection_all[k][i][-1]:
                                continue
                            subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                            subset[j][indexB][0] = partBs[i]
                            subset[j][indexB][1] = connection_all[k][i][2]
                            subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                            subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])
                    else:
                        pass

                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx

                    membership1 = ((subset[j1][..., 0] >= 0).astype(int))[:-2]
                    membership2 = ((subset[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(subset[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(subset[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)

                        if connection_all[k][i][2] < params['connection_tole'] * min_tolerance or params['len_rate'] * \
                                subset[j1][-1][1] <= connection_all[k][i][-1]:
                            # todo: finetune the tolerance of connection
                            continue  #

                        subset[j1][:-2][...] += (subset[j2][:-2][...] + 1)
                        subset[j1][-2:][:, 0] += subset[j2][-2:][:, 0]

                        subset[j1][-2][0] += connection_all[k][i][2]
                        subset[j1][-1][1] = max(connection_all[k][i][-1], subset[j1][-1][1])
                        subset = np.delete(subset, j2, 0)

                    else:
                        if connection_all[k][i][0] in subset[j1, :-2, 0]:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][0])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][1])
                        else:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][1])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][0])

                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        if connection_all[k][i][2] < subset[j1][c1][1] and connection_all[k][i][2] < subset[j2][c2][1]:
                            continue  # the trick here is useful

                        small_j = j1
                        big_j = j2
                        remove_c = c1

                        if subset[j1][c1][1] > subset[j2][c2][1]:
                            small_j = j2
                            big_j = j1
                            remove_c = c2


                        subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + \
                                                  subset[small_j][remove_c][1]
                        subset[small_j][remove_c][0] = -1  # todo
                        subset[small_j][remove_c][1] = -1
                        subset[small_j][-1][0] -= 1

                # if find no partA in the subset, create a new subset
                elif not found and k < 24:
                    # Fixme: 原始的时候是18,因为我加了limb，所以是24,因为真正的limb是0~16，最后两个17,18是额外的不是limb
                    # FIXME: 但是后面画limb的时候没有把鼻子和眼睛耳朵的连线画上，要改进
                    row = -1 * np.ones((20, 2))
                    row[indexA][0] = partAs[i]
                    row[indexA][1] = connection_all[k][i][2]
                    row[indexB][0] = partBs[i]
                    row[indexB][1] = connection_all[k][i][2]
                    row[-1][0] = 2
                    row[-1][1] = connection_all[k][i][-1]
                    row[-2][0] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    row = row[np.newaxis, :, :]
                    subset = np.concatenate((subset, row), axis=0)
        # todo: solve the unmathced keypoint?
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1][0] < 2 or subset[i][-2][0] / subset[i][-1][0] < 0.45:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return subset, candidate


def process(input_image, params, model, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image)  # B,G,R order
    # print(input_image)
    heatmap_avg, paf_avg = predict(oriImg, params, model, model_params, heat_layers, paf_layers)

    all_peaks = find_peaks(heatmap_avg, params)
    connection_all, special_k = find_connections(all_peaks, paf_avg, oriImg.shape[0], params)
    subset, candidate = find_people(connection_all, special_k, all_peaks, params)

    keypoints = []
    for s in subset[..., 0]:
        keypoint_indexes = s[:18]
        person_keypoint_coordinates = []
        for index in keypoint_indexes:
            if index == -1:
                X, Y = 0, 0
            else:
                X, Y = candidate[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        person_keypoint_coordinates_coco = [None] * 17

        for dt_index, gt_index in dt_gt_mapping.items():
            if gt_index is None:
                continue
            person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[18]))
        # s[18] is the score, s[19] is the number of keypoint
    return keypoints


def get_image_name(coco, image_id):
    return coco.imgs[image_id]['file_name']


def predict_many(coco, images_directory, validation_ids, params, model, model_params, heat_layers, paf_layers):
    assert (not set(validation_ids).difference(set(coco.getImgIds())))

    keypoints = {}
    for image_id in tqdm.tqdm(validation_ids):
        image_name = get_image_name(coco, image_id)
        image_name = os.path.join(images_directory, image_name)
        keypoints[image_id] = process(image_name, dict(params), model, dict(model_params), heat_layers+1, paf_layers)
        # fixme: heat_layers + 1 if you use background keypoint  !!!
    return keypoints


def format_results(keypoints, resFile):
    format_keypoints = []
    # todo: do we need to sort the detections by scores before evaluation ?
    for image_id, people in keypoints.items():
        for keypoint_list, score in people:
            format_keypoint_list = []
            for x, y in keypoint_list:
                for v in [int(x), int(y), 1 if x > 0 or y > 0 else 0]:
                    format_keypoint_list.append(v)

            format_keypoints.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": format_keypoint_list,
                "score": score,
            })

    json.dump(format_keypoints, open(resFile, 'w'))


def validation(model, dump_name, validation_ids=None, dataset='val2017'):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'dataset/coco/link2coco2017'

    # # #############################################################################
    # 在验证集上测试代码
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataset)
    print(annFile)
    cocoGt = COCO(annFile)

    if validation_ids == None:   # todo: we can set the validataion image ids here  !!!!!!
        validation_ids = cocoGt.getImgIds()
    # # #############################################################################

    resFile = '%s/results/%s_%s_%s100_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('the path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    keypoints = predict_many(cocoGt, os.path.join(dataDir, dataset), validation_ids, params, model, model_params,
                             config.heat_layers, config.paf_layers)
    format_results(keypoints, resFile)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = validation_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval


def per_image_scores(eval_result):
    def convert_match_to_score(match):
        matches = match['gtMatches'][:, np.array(match['gtIgnore']) == 0]
        scores = {
            'image_id': match['image_id'],
            'gt_person_count': matches.shape[1],
        }

        for i in range(matches.shape[0]):
            okp_threshold = eval_result.params.iouThrs[i]
            scores['matched_%.2f' % okp_threshold] = sum(matches[i, :] != 0)
        scores['average'] = np.mean(np.sum(matches != 0, axis=1)) / scores['gt_person_count']

        return scores

    evalImgs = eval_result.evalImgs
    scores = [convert_match_to_score(image_match) for image_match in evalImgs if image_match is not None]

    return pd.DataFrame(scores)


if __name__ == "__main__":
    config = GetConfig('Canonical')
    params, model_params = config_reader(figstr='testing/config')
    from posenet.mymodel3 import get_testing_model

    # with tf.device("/cpu:0"):
    #     model_single = get_testing_model(np_branch1=config.paf_layers, np_branch2=config.heat_layers, stages=1)
    # model = multi_gpu_model(model_single, gpus=2)
    model = get_testing_model(np_branch1=config.paf_layers, np_branch2=config.heat_layers+1, stages=3)  # fixme: background + 1
    training_dir = './training/'
    trained_models = [
        'weights'
        # 'weights-cpp-lr'
        # 'weights-python-last',
    ]

    optimal_epoch_loss = 'val_weight_stage6_L1_loss'
    weights_path = 'training/cpu_weights/cpu_model.h5'
    # '../model/keras/model.h5'    # orginal weights converted from caffe
    model.load_weights(weights_path)
    eval_result_original = validation(model, dump_name='pose_focal_4scale', dataset='val2017')  # val2017

    print('over!')


