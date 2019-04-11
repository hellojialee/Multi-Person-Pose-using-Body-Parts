import tensorflow as tf
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import util
from config_reader import config_reader
from config import COCOSourceConfig, GetConfig
from scipy.ndimage.filters import gaussian_filter
from posenet.mymodel3 import get_testing_model
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
K.set_learning_phase(1)

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

limbSeq = [[1, 0], [1, 14], [1, 15], [1, 16], [1, 17], [0, 14], [0, 15], [14, 16], [15, 17],
           [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
           [9, 10], [1, 11], [11, 12], [12, 13], [8, 11], [2, 16], [5, 17]]


mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23],
          [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45],
          [46, 47]]

# visualize
colors = [[	128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
          [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170],
          [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255]]

dt_gt_mapping = {0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
                 14: 2, 15: 1, 16: 4, 17: 3, 18: None}


def show_color_vector(oriImg, paf_avg, heatmap_avg):
    hsv = np.zeros_like(oriImg)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(paf_avg[:, :, 16], 1.5 * paf_avg[:, :, 16])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    limb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(limb_flow, alpha=.5)
    plt.title('[a body part heatmap] \n close this window and do next')
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])  # show a keypoint
    plt.imshow(heatmap_avg[:, :, 2], alpha=.5)
    plt.title('[a keypoint heatmap] \n close this window and do next')
    plt.show()


def process(input_image, params, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image)  # B,G,R order.
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]  # 按照图片高度进行缩放
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], heat_layers))  # fixme if you change the number of keypoints
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], paf_layers))
    multiplier = multiplier
    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        output_blobs = model_single.predict(input_img)
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps. notice: 模型定义中paf是branch1,heatmap是branch2
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)


    all_peaks = []
    peak_counter = 0
    # --------------------------------------------------------------------------------------- #
    # ------------------------  show the limb and foreground channel  -----------------------#
    # --------------------------------------------------------------------------------------- #

    show_color_vector(oriImg, paf_avg, heatmap_avg)

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # ------------------------- find keypoints  ---------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)  # fixme: use gaussian blure?
        # map = map_ori
        # map up 是值
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

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        # np.nonzero: Return the indices of the elements that are non-zero
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]  # 列表解析式，生产的是list
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # ----------------------------- find connections -----------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

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
                    mid_num = max(int(norm), 10)
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))] \
                                      for I in range(len(startend))])

                    score_midpts = limb_response

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48

                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > params['connect_ration'] * len(score_midpts)  # fixme: tune 手动调整, 本来是 > 0.8*len
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, norm,
                                                     0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][2]])
                        # How to undersatand the criterion?

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)

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

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # --------------------------------- find people ------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

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
                    if subset[j][indexA][0].astype(int) == (partAs[i]).astype(int) or subset[j][indexB][0].astype(int) == partBs[i].astype(int):
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
                        # candidate的格式为：  (343, 490, 0.8145177364349365, 27), ....
                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                        # the second last number in each row is the score of the overall configuration

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

                    membership1 = ((subset[j1][..., 0] >= 0).astype(int))[:-2]  # 用[:,0]也可
                    membership2 = ((subset[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(subset[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(subset[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)

                        if connection_all[k][i][2] < params['connection_tole'] * min_tolerance or params['len_rate'] * subset[j1][-1][1] <= connection_all[k][i][-1]:
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

                        subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + subset[small_j][remove_c][1]
                        subset[small_j][remove_c][0] = -1  # todo
                        subset[small_j][remove_c][1] = -1
                        subset[small_j][-1][0] -= 1

                elif not found and k < 24:
                    row = -1 * np.ones((20, 2))
                    row[indexA][0] = partAs[i]
                    row[indexA][1] = connection_all[k][i][2]
                    row[indexB][0] = partBs[i]
                    row[indexB][1] = connection_all[k][i][2]
                    row[-1][0] = 2
                    row[-1][1] = connection_all[k][i][-1]
                    row[-2][0] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    row = row[np.newaxis, :, :]  # 为了进行concatenate，需要插入一个轴
                    subset = np.concatenate((subset, row), axis=0)

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1][0] < 4 or subset[i][-2][0] / subset[i][-1][0] < 0.45:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = cv2.imread(input_image)  # B,G,R order
    # canvas = oriImg
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

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[-2]))  # s[19] is the score

    for i in range(len(keypoints)):
        print('the {}th keypoint detection result is : '.format(i), keypoints[i])

    draw_list = [0] + list(range(5, 22))
    for i in draw_list:
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])][..., 0]
            if -1 in index:
                continue
            # 在上一个cell中有　canvas = cv2.imread(test_image) # B,G,R order
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 3), int(angle), 0,
                                       360, 1)

            cv2.circle(cur_canvas, (int(Y[0]), int(X[0])), 4, color=[0, 0, 0], thickness=2)
            cv2.circle(cur_canvas, (int(Y[1]), int(X[1])), 4, color=[0, 0, 0], thickness=2)

            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='try_image/tokyo2.jpg', help='input image')  # required=True
    parser.add_argument('--output', type=str, default='result.jpg', help='output image')
    parser.add_argument('--model', type=str, default='training/cpu_weights/cpu_model.h5',
                        help='path to the weights file')

    config = GetConfig('Canonical')

    args = parser.parse_args()
    input_image = args.image
    output = args.output
    keras_weights_file = args.model

    tic = time.time()
    print('start processing...')

    # with tf.device("/cpu:0"):
    model_single = get_testing_model(np_branch1=config.paf_layers, np_branch2=config.heat_layers+1, stages=3)

    model_single.load_weights(keras_weights_file)
    # print(model_single.get_weights()[-1].shape)
    print('----------  the weight has been loaded ----------------')


    # load config
    params, model_params = config_reader()
    tic = time.time()
    # generate image with body parts
    canvas = process(input_image, params, model_params, config.heat_layers+1, config.paf_layers)  # fixme: background + 1

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    cv2.namedWindow("press the keyboard to finish", cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL 自动适合的窗口大小
    cv2.imshow("press the keyboard to finish", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output, canvas)
