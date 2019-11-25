#!/usr/bin/env python
"""
Python script for generating the training and validation hdf5 data from MPII dataset
原始项目中更新了部分程序，此代码暂时还没有运行尝试过
"""
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import h5py
import json

from zipfile import ZipFile
from scipy.io import loadmat
from pprint import pprint
from time import time

dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset/mpii/link2mpii'))

mpii_dir = os.path.join(dataset_dir, "")
keypoints_file = os.path.join(mpii_dir, "mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")
img_dir = os.path.join(mpii_dir, "images")

tr_hdf5_path = os.path.join(dataset_dir, "mpii_train_dataset.h5")
val_hdf5_path = os.path.join(dataset_dir, "mpii_val_dataset.h5")

val_size = 2645

def load_anno(keypoints, n):
    # crazy format :-/

    annolist = keypoints.annolist[0, n]

    assert annolist.image.shape == (1, 1)
    assert annolist.image[0, 0].name.shape == (1,)
    image = annolist.image[0, 0].name[0]

    annorects = []

    assert annolist.annorect.shape[0] == 1 or annolist.annorect.shape==(0,0), annolist.annorect.shape

    for i in range(annolist.annorect.shape[1]):

        annorect_scale = None
        if 'scale' in vars(annolist.annorect[0, i]) and annolist.annorect[0, i].scale.size > 0:
            assert annolist.annorect[0, i].scale.shape == (1, 1), annolist.annorect[0, i].scale.shape
            annorect_scale = float(annolist.annorect[0, i].scale[0, 0])

        annorect_objpos_x = None
        annorect_objpos_y = None
        if 'objpos' in vars(annolist.annorect[0, i]) and annolist.annorect[0, i].objpos.size > 0:
            assert annolist.annorect[0, i].objpos.shape == (1, 1)
            assert annolist.annorect[0, i].objpos[0, 0].x.shape == (1, 1)
            annorect_objpos_x = int(annolist.annorect[0, i].objpos[0, 0].x[0, 0])
            assert annolist.annorect[0, i].objpos[0, 0].y.shape == (1, 1)
            annorect_objpos_y = int(annolist.annorect[0, i].objpos[0, 0].y[0, 0])

        annopoints = []

        if 'annopoints' in vars(annolist.annorect[0, i]) and annolist.annorect[0, i].annopoints.size > 0:
            assert annolist.annorect[0, i].annopoints.shape == (1, 1)
            points = annolist.annorect[0, i].annopoints[0, 0].point

            assert points.shape[0] == 1
            for p in range(points.shape[1]):
                point = points[0, p]
                point._fieldnames == ['id', 'x', 'y', 'is_visible']

                assert point.id.shape == (1, 1)
                point_id = int(point.id[0, 0])

                assert point.x.shape == (1, 1)
                point_x = int(point.x[0, 0])

                assert point.y.shape == (1, 1)
                point_y = int(point.y[0, 0])

                point_is_visible = None
                if 'is_visible' in vars(point) and point.is_visible.size > 0:
                    assert point.is_visible.shape == (1, 1) or point.is_visible.shape == (1,), point.is_visible.shape

                    if point.is_visible.shape == (1, 1): point_is_visible = int(point.is_visible[0, 0])
                    if point.is_visible.shape == (1,):   point_is_visible = int(point.is_visible[0])

                annopoints.append({'id': point_id, 'x': point_x, 'y': point_y, 'is_visible': point_is_visible})

        head = None

        if 'x1' in vars(annolist.annorect[0, i]):
            assert (
                   annolist.annorect[0, i].x1.shape, annolist.annorect[0, i].x2.shape, annolist.annorect[0, i].y1.shape,
                   annolist.annorect[0, i].y2.shape) == ((1, 1), (1, 1), (1, 1), (1, 1))
            head = {'x1': int(annolist.annorect[0, i].x1[0, 0]), 'x2': int(annolist.annorect[0, i].x2[0, 0]),
                    'y1': int(annolist.annorect[0, i].y1[0, 0]), 'y2': int(annolist.annorect[0, i].y2[0, 0]) }

        annorects.append(
            {'scale': annorect_scale, 'objpos': {'x': annorect_objpos_x, 'y': annorect_objpos_y}, 'head': head,
             'annopoints': annopoints if len(annopoints) > 0 else None})

    assert annolist.frame_sec.shape == (1, 0) or annolist.frame_sec.shape == (1, 1)
    frame_sec = int(annolist.frame_sec[0, 0]) if annolist.frame_sec.shape == (1, 1) else None

    assert annolist.vididx.size == 0 or annolist.vididx.shape == (1, 1)
    vididx = int(annolist.vididx[0,0]) if annolist.vididx.shape == (1, 1) else None

    assert keypoints.img_train.shape[0] == 1
    img_train = int(keypoints.img_train[0, n])

    assert keypoints.version.shape == (1,)
    version = int(keypoints.version[0])

    single_person = []

    if keypoints.single_person[n, 0].size > 0:

        if keypoints.single_person[n, 0].shape[0] == 1:
            for i in range(keypoints.single_person[n, 0].shape[1]):
                single_person.append(int(keypoints.single_person[n, 0][0,i]))
        elif keypoints.single_person[n, 0].shape[1] == 1:
            for i in range(keypoints.single_person[n, 0].shape[0]):
                single_person.append(int(keypoints.single_person[n, 0][i,0]))
        else:
            assert False, keypoints.single_person[n, 0].shape

    assert keypoints.act[n, 0].act_id.shape == (1, 1)
    act_id = int(keypoints.act[n, 0].act_id[0, 0])

    assert keypoints.act[n, 0].act_name.shape == (0,) or keypoints.act[n, 0].act_name.shape == (1,)
    act_name = keypoints.act[n, 0].act_name[0] if keypoints.act[n, 0].act_name.shape == (1,) else None

    assert keypoints.act[n, 0].cat_name.shape == (0,) or keypoints.act[n, 0].cat_name.shape == (1,)
    cat_name = keypoints.act[n, 0].cat_name[0] if keypoints.act[n, 0].cat_name.shape == (1,) else None

    video_name = None

    if vididx is not None:
        assert keypoints.video_list[0,vididx-1].shape==(1,)
        video_name = "https://www.youtube.com/watch?v=" + keypoints.video_list[0,vididx-1][0]


    return {'image': image, 'annorects': annorects, 'img_train': img_train, 'version': version,
            'single_person': single_person, 'act': {'act_id': act_id, 'act_name': act_name, 'cat_name': cat_name},
            'video_name': video_name, 'vididx': vididx, 'frame_sec': frame_sec}


def load_image(img_dir, img_id):

    img_path = os.path.join(img_dir, img_id)
    t = time()
    img = cv2.imread(img_path)
    print((time()-t)*1000)


    h, w, c = img.shape
    mask_miss = np.zeros((h, w), dtype=np.uint8)
    mask_miss *= 255  # mpii数据集中没有mask

    return img


def process_image(anno):

    if anno['img_train'] != 1:
        return None

    results = { 'joints':[], 'scale_provided':[], 'objpos':[], 'head':[]  }

    for annorect in anno['annorects']:

        if annorect['annopoints'] is None:
            return None

        results['scale_provided'] += [ annorect['scale'] ]
        results['objpos'] += [ [ annorect['objpos']['x'], annorect['objpos']['y'] ]]
        (x1, y1, x2, y2) = (annorect['head']['x1'], annorect['head']['y1'], annorect['head']['x1'], annorect['head']['y2'])

        results['head'] += [ [x1, y1, x2-x1, y2-y1] ]
        assert x2-x1>=0 and y2-y1>=0

        joints = np.ones((16,3), dtype=np.float)
        joints[:,:] = float('nan')
        for j in annorect['annopoints']:
            joints[j['id']][0] = j['x']
            joints[j['id']][1] = j['y']
            joints[j['id']][2] = j['is_visible'] if j['is_visible'] is not None else 0  # not sure it None means 0 but ...

        joints[~np.isfinite(joints[:, 2]), 0:2] = 0
        joints[~np.isfinite(joints[:, 2]), 2] = 2
        assert np.all(np.isfinite(joints)), joints

        results['joints'] += [ joints.tolist() ]

    if len(results['scale_provided'])==0:
        return

    assert len(results['scale_provided']) == len(results['joints']) and len(results['scale_provided']) == len(results['objpos'])


    for p in anno['single_person']:

        pp=p-1

        yield_item = {}

        yield_item['scale_provided']= results['scale_provided'][pp:] + results['scale_provided'][:pp]
        yield_item['joints'] = results['joints'][pp:] + results['joints'][:pp]
        yield_item['objpos'] = results['objpos'][pp:] + results['objpos'][:pp]
        yield_item['head'] = results['head'][pp:] + results['head'][:pp]

        yield yield_item

    return results


def writeImage(grp, img_grp, anno, processed_anno, img, count, image_id):

    anno['count'] = count

    #print(anno)

    if not image_id in img_grp:
        print('Writing image %s' % image_id)
        _, compressed_image = cv2.imencode(".jpg", img)
        img_ds = img_grp.create_dataset(image_id, data=compressed_image, chunks=None)

    key = '%07d' % count
    processed_anno['image']=image_id

    ds = grp.create_dataset(key, data=json.dumps(processed_anno), chunks=None)
    ds.attrs['meta'] = json.dumps(anno)

    print('Writing sample %d' % count)


def process():

    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("dataset")
    tr_write_count = 0
    tr_grp_img = tr_h5.create_group("images")

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")

    count = 0

    print("loading %s ..." % keypoints_file)
    keypoints = loadmat(keypoints_file, struct_as_record=False )
    keypoints = keypoints['RELEASE']
    assert keypoints.shape == (1,1)
    keypoints = keypoints[0, 0]

    numitems = keypoints.annolist.shape[1]

    img = None
    cached_img_id = None

    for i in range(numitems):
        anno = load_anno(keypoints,i)

        for processed in process_image(anno):
            count += 1
            #print(i, count, numitems, anno['image'], processed['scale_provided'] )

            if cached_img_id != anno['image']:
                cached_img_id = anno['image']
                img = load_image(img_dir, cached_img_id)

            if count < val_size:
                writeImage(val_grp, val_grp_img, anno, processed, img, val_write_count, cached_img_id)
                val_write_count += 1
            else:
                writeImage(tr_grp, tr_grp_img, anno, processed, img, tr_write_count, cached_img_id)
                tr_write_count += 1

    tr_h5.close()
    val_h5.close()

    print("Total:", count)

if __name__ == '__main__':
    process()