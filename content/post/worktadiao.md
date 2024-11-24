---
layout:     post
title:      "3D点云目标检测加追踪后处理"
description: "无需标注第一帧的自动追踪算法，轻量准确实时"
excerpt: "航拍+8D重庆+赛国朋克"
date:    2024-11-11
author:     "甜甜圈"
image: "img/fj1.jpg"
published: true 
tags:
    - ubuntu 
URL: "/worktadiao/"
categories: [ "我的工作" ]    
---
## 多场景项目成果展示视频
训练命令：python tools/train.py --cfg_file /home/students/master/2023/luojl/td/tools/cfgs/custom_models/centerpoint.yaml
测试命令：python tools/demo2.py --ckpt /home/students/master/2023/luojl/openpcdet/output/home/students/master/2023/luojl/td/tools/cfgs/custom_models/pointpillar1/default/ckpt/checkpoint_epoch_334.pth
最终测试导出视频图片并且保存一段时间所有结果到json文件命令：tools/demotrack.py --ckpt /home/students/master/2023/luojl/openpcdet/output/home/students/master/2023/luojl/td/tools/cfgs/custom_models/pointpillar1/default/ckpt/checkpoint_epoch_334.pth
生成数据集索引pkl命令：cd openpcdet python -m pcdet.datasets.custom.custom_dataset create_custom_infos /home/students/master/2023/luojl/td/tools/cfgs/dataset_configs/custom_dataset.yaml
查看pkl是否正确命令：python show.py
数据集要长成这样才能训练测试：
![加载中……](/img/work/tdtip2.png)
{{<bilibili src="//player.bilibili.com/player.html?isOutside=true&aid=113537192041463&bvid=BV1qpBxYqEDv&cid=26960006014&p=1">}}
代码和项目使用文档以及数据集会之后上传到我的GitHub上，欢迎交流学习。
#### * 3D点云目标检测网络best in our dataset is centerpoint+pillar：
比较难搞定的配置文件如下：
centerpoint.yaml    
注意： REMOVE_OUTSIDE_BOXES: False不然会报递归溢出 POINT_CLOUD_RANGE: [-74.88, -74.88, 0, 74.88, 74.88, 20]最后一项和POINT_CLOUD_RANGE: [-74.88, -74.88, 0, 74.88, 74.88, 20]最后一项相同，不然会报网络尺度错误,还要x,y的倍数对应关系。
![加载中……](/img/work/tadiaotip1.png)
```html
CLASS_NAMES: ['couple']


DATA_CONFIG:
    _BASE_CONFIG_: /home/students/master/2023/luojl/td/tools/cfgs/dataset_configs/custom_dataset.yaml

    POINT_CLOUD_RANGE: [-74.88, -74.88, 0, 74.88, 74.88, 20]
    DATA_PROCESSOR:
        -   NAME: mask_points_and_boxes_outside_range
            REMOVE_OUTSIDE_BOXES: False

        -   NAME: shuffle_points
            SHUFFLE_ENABLED: {
                'train': True,
                'test': True
            }

        -   NAME: transform_points_to_voxels
            POINT_CLOUD_RANGE: [-74.88, -74.88, 0, 74.88, 74.88, 20]
            MAX_POINTS_PER_VOXEL: 20
            MAX_NUMBER_OF_VOXELS: {
                'train': 150000,
                'test': 150000
            }

MODEL:
    NAME: CenterPoint

    VFE:
        NAME: PillarVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: True
        NUM_FILTERS: [ 64, 64 ]

    MAP_TO_BEV:
        NAME: PointPillarScatter
        NUM_BEV_FEATURES: 64

    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [ 3, 5, 5 ]
        LAYER_STRIDES: [ 1, 2, 2 ]
        NUM_FILTERS: [ 64, 128, 256 ]
        UPSAMPLE_STRIDES: [ 1, 2, 4 ]
        NUM_UPSAMPLE_FILTERS: [ 128, 128, 128 ]

    DENSE_HEAD:
        NAME: CenterHead
        CLASS_AGNOSTIC: False

        CLASS_NAMES_EACH_HEAD: [
            ['couple']
        ]

        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: True
        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'center_z', 'dim', 'rot']
            HEAD_DICT: {
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
            }

        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 1
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2

        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

        POST_PROCESSING:
            SCORE_THRESH: 0.7
            POST_CENTER_LIMIT_RANGE: [-80, -80, -10.0, 80, 80, 10.0]
            MAX_OBJ_PER_SAMPLE: 500
            NMS_CONFIG:
                NMS_TYPE: nms_gpu
                NMS_THRESH: 0.7
                NMS_PRE_MAXSIZE: 4096
                NMS_POST_MAXSIZE: 500

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]

        EVAL_METRIC: waymo


OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 300

    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
```
customdataset.yaml
```html
DATASET: 'CustomDataset'
DATA_PATH: '/home/students/master/2023/luojl/'

# If this config file is modified then pcdet/models/detectors/detector3d_template.py:
# Detector3DTemplate::build_networks:model_info_dict needs to be modified.
POINT_CLOUD_RANGE: [ -190.22, -107.33, -420.26,  807.27 , 208.22, 680.14] # x=[-70.4, 70.4], y=[-40,40], z=[-3,1]
# Point Cloud Center:        [ 6.32  2.91 30.86]        
# Point Cloud Minimums:      [ -19.22 -107.33  -42.26]  
# Point Cloud Maximums:      [87.27 28.22 68.14]        
# Initial Translation:       [  -6.32   -2.91 -235.56]  
DATA_SPLIT: {
    'train': train,
    'test': val
}

INFO_PATH: {
    'train': [custom_infos_train.pkl],
    'test': [custom_infos_val.pkl],
}

GET_ITEM_LIST: ["points"]
FOV_POINTS_ONLY: False

POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'intensity'],
    src_feature_list: ['x', 'y', 'z', 'intensity'],
}

# Same to pv_rcnn[DATA_AUGMENTOR]
DATA_AUGMENTOR:
    DISABLE_AUG_LIST: ['placeholder']
    AUG_CONFIG_LIST:
        - NAME: gt_sampling
          # Notice that 'USE_ROAD_PLANE'
          USE_ROAD_PLANE: False
          DB_INFO_PATH:
              - custom_dbinfos_train.pkl # pcdet/datasets/augmentor/database_ampler.py:line 26
          PREPARE: {
             filter_by_min_points: ['couple:5'],
             filter_by_difficulty: [-1],
          }

          SAMPLE_GROUPS: ['couple:5']
          NUM_POINT_FEATURES: 4
          DATABASE_WITH_FAKELIDAR: False
          REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
          LIMIT_WHOLE_SCENE: True

        - NAME: random_world_flip
          ALONG_AXIS_LIST: ['x']

        - NAME: random_world_rotation
          WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]

        - NAME: random_world_scaling
          WORLD_SCALE_RANGE: [0.95, 1.05]

DATA_PROCESSOR:
    - NAME: mask_points_and_boxes_outside_range
      REMOVE_OUTSIDE_BOXES: True

    - NAME: shuffle_points
      SHUFFLE_ENABLED: {
        'train': True,
        'test': False
      }

    - NAME: transform_points_to_voxels
      VOXEL_SIZE: [0.05, 0.05, 0.1]
      MAX_POINTS_PER_VOXEL: 5
      MAX_NUMBER_OF_VOXELS: {
        'train': 16000,
        'test': 40000
      }

```
pcdet/custom.py仅加载点云数据不加载图像
```html
import copy
import pickle
import os
 
import numpy as np
from skimage import io
 
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_custom
from ..dataset import DatasetTemplate
# 定义属于自己的数据集，集成数据集模板
class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        print('root_path',root_path)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = os.path.join(self.root_path, ('training' if self.split != 'test' else 'testing'))
 
        split_dir = os.path.join(self.root_path, 'ImageSets',(self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None
 
        self.custom_infos = []
        self.include_custom_data(self.mode)
        self.ext = ext
 
    # 用于导入自定义数据
    def include_custom_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Custom dataset.')
        custom_infos = []
 
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)
        
        self.custom_infos.extend(custom_infos)
 
        if self.logger is not None:
            self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))
    
    # 用于获取标签的标注信息
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # 线程函数，主要是为了多线程读取数据，加快处理速度
        print(sample_id_list)
        # 处理一帧
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # 创建一个用于存储一帧信息的空字典
            info = {}
            # 定义该帧点云信息，pointcloud_info
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            # 将pc_info这个字典作为info字典里的一个键值对的值，其键名为‘point_cloud’添加到info里去
            info['point_cloud'] = pc_info
            '''
            # image信息和calib信息都暂时不需要
            # image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # info['image'] = image_info
            # calib = self.get_calib(sample_idx)
            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            # info['calib'] = calib_info
            '''
            if has_label:
                # 通过get_label函数，读取出该帧的标签标注信息
                obj_list = self.get_label(sample_idx)
                # 创建用于存储该帧标注信息的空字典
                annotations = {}
                # 下方根据标注文件里的属性将对应的信息加入到annotations的键值对，可以根据自己的需求取舍
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                #
                # annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
 
                # 统计有效物体的个数，即去掉类别名称为“Dontcare”以外的
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                # 统计物体的总个数，包括了Dontcare
                num_gt = len(annotations['name'])
                # 获得当前的index信息
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
 
                # 从annotations里提取出从标注信息里获取的location、dims、rots等信息，赋值给对应的变量
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 由于我们的数据集本来就是基于雷达坐标系标注，所以无需坐标转换
                #loc_lidar = calib.rect_to_lidar(loc)
                loc_lidar = self.get_calib(loc)
                # 原来的dims排序是高宽长hwl,现在转到pcdet的统一坐标系下,按lhw排布
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                
                # 由于我们基于雷达坐标系标注，所以获取的中心点本来就是空间中心，所以无需从底面中心转到空间中心
                # bottom center -> object center: no need for loc_lidar[:, 2] += h[:, 0] / 2
                # print("sample_idx: ", sample_idx, "loc: ", loc, "loc_lidar: " , sample_idx, loc_lidar)
                # get gt_boxes_lidar see https://zhuanlan.zhihu.com/p/152120636
                # loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                print('gt_boxes_lidar',gt_boxes_lidar)
                # 将雷达坐标系下的真值框信息存入annotations中
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                # 将annotations这整个字典作为info字典里的一个键值对的值
                info['annos'] = annotations
            
            return info
            # 后续的由于没有calib信息和image信息，所以可以直接注释
            '''
            #     if count_inside_pts:
            #         points = self.get_lidar(sample_idx)
            #         calib = self.get_calib(sample_idx)
            #         pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #         fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
            #         pts_fov = points[fov_flag]
            #         corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
            #         num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
            #         for k in range(num_objects):
            #             flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
            #             num_points_in_gt[k] = flag.sum()
            #         annotations['num_points_in_gt'] = num_points_in_gt
            # return info
            '''
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
        # 此时返回值infos是列表，列表元素为字典类型
                
    # 用于获取标定信息
    def get_calib(self, loc):
        # calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        # assert calib_file.exists()
        # return calibration_kitti.Calibration(calib_file)
        
        # loc_lidar = np.concatenate([np.array((float(loc_obj[2]),float(-loc_obj[0]),float(loc_obj[1]-2.3)),dtype=np.float32).reshape(1,3) for loc_obj in loc])
        # return loc_lidar
        # 这里做了一个由相机坐标系到雷达坐标系翻转（都遵从右手坐标系），但是 -2.3这个数值具体如何得来需要再看下
 
        # 我们的label中的xyz就是在雷达坐标系下,不用转变,直接赋值
        loc_lidar = np.concatenate([np.array((float(loc_obj[0]),float(loc_obj[1]),float(loc_obj[2])),dtype=np.float32).reshape(1,3) for loc_obj in loc])
        return loc_lidar
                
    # 用于获取标签
    def get_label(self, idx):
        # 从指定路径中提取txt内容
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        # 主要就是从这个函数里获取具体的信息
        return object3d_custom.get_objects_from_label(label_file)
 
    # 用于获取雷达点云信息
    def get_lidar(self, idx, getitem):
        """
            Loads point clouds for a sample
                Args:
                    index (int): Index of the point cloud file to get.
                Returns:
                    np.array(N, 4): point cloud.
        """
        # get lidar statistics
        if getitem == True:
            lidar_file = self.root_split_path + '/velodyne/' + ('%s.bin' % idx)
            
        else:
            lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
 
    # 用于数据集划分
    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
 
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
 
    # 创建真值数据库
    # Create gt database for data augmentation
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
            import torch
    
            database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
            db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)
    
            database_save_path.mkdir(parents=True, exist_ok=True)
            all_db_infos = {}
    
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
    
            for k in range(len(infos)):
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
                info = infos[k]
                sample_idx = info['point_cloud']['lidar_idx']
                points = self.get_lidar(sample_idx,False)
                annos = info['annos']
                names = annos['name']
                # difficulty = annos['difficulty']
                # bbox = annos['bbox']
                gt_boxes = annos['gt_boxes_lidar']
    
                num_obj = gt_boxes.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
    
                for i in range(num_obj):
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]
    
                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)
    
                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                        #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                        db_info = {'name': names[i], 'path': db_path,  'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0], 'score': annos['score'][i]}
                        
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))
    
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)
    # 生成预测字典信息
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N,7), Tensor
                pred_scores: (N), Tensor
                pred_lables: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_smaples):
            ret_dict = {
                'name': np.zeros(num_smaples), 'alpha' : np.zeros(num_smaples),
                'dimensions': np.zeros([num_smaples, 3]), 'location': np.zeros([num_smaples, 3]),
                'rotation_y': np.zeros(num_smaples), 'score': np.zeros(num_smaples),
                'boxes_lidar': np.zeros([num_smaples, 7])
            }
            return ret_dict
 
        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
 
            # Define an empty template dict to store the prediction information, 'pred_scores.shape[0]' means 'num_samples'
            pred_dict = get_template_prediction(pred_scores.shape[0])
            # If num_samples equals zero then return the empty dict
            if pred_scores.shape[0] == 0:
                return pred_dict
 
            # No calibration files
 
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes,None)
 
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes[:, 6]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
 
            return pred_dict
 
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
 
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
 
            # Output pred results to Output-path in .txt file 
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = [0,0,50,50]
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl: lidar -> camera
 
                    for idx in range(len(loc)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                            % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                bbox[0], bbox[1], bbox[2], bbox[3],
                                dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                single_pred_dict['score'][idx]), file=f)
            return annos
    '''def evaluation(self,det_annos,class_names,**kwargs):
        if 'name' not in self.custom_infos[0].keys():
            #如果robosense_infos里没有信息，直接返回空字典
            return None,{}
        #参数det_annos 是验证集val下面的所有infos,是一个列表，每个元素是每一帧的字典数据
        #这里 的info是从model出来的，由generate_prediction_dicts函数得到，字典的键key:
        # name , box_center,box_size,box_rotation,tracked_id,    scores,pred_labels,pred_lidar,frame_id
      

        from .kitti_object_eval_python import eval as kitti_eval
        
        #复制一下参数det_annos
        #copy.deepcopy()在元组和列表的嵌套上的效果是一样的，都是进行了深拷贝（递归的）
        #eval_det_info的内容是从model预测出来的结果，等于det_annos
        eval_det_info = copy.deepcopy(det_annos)
       
        

        #调用函数，预测得到ap的值1
        #ap_result_str,ap_dict = kitti_eval.get_coco_eval_result1(eval_gt_infos,eval_det_info,class_names)1
        ap_result_str,ap_dict = kitti_eval.get_official_eval_result(eval_gt_infos,eval_det_info,class_names)
        print('aaaaaa')

        return ap_result_str,ap_dict '''

    def evaluation(self, det_annos, class_names, **kwargs):
            if 'annos' not in self.custom_infos[0].keys():
                return None, {}
 
            from .kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils
            print('detinfo',det_annos)
 
            eval_det_annos =  kitti_utils.transform_annotations_to_kitti_format(det_annos, class_names)
            #kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]
            eval_gt_annos =  kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, class_names)
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos,eval_det_annos,  class_names)
 
            return ap_result_str, ap_dict
    # 用于返回训练帧的总个数
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs
 
        return len(self.custom_infos)
 
    # 用于将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的self.prepare_data()
    def __getitem__(self, index):  ## 修改如下
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)
 
        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx, True)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }
 
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
 
        data_dict = self.prepare_data(data_dict=input_dict)
 
        return data_dict
 
# 用于创建自定义数据集的信息
def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    print('data_path',data_path)
    data_path = r'/home/oseasy/桌面/1/OpenPCDet-master/kitti'
    train_split, val_split = 'train', 'val'
   # 定义文件的路径和名称
    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'custom_infos_trainval.pkl'
    test_filename = save_path / 'custom_infos_test.pkl'
 
    print('---------------Start to generate data infos---------------')
 
    dataset.set_split(train_split)
    
    # 执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    # 下面是得到train.txt中序列相关的所有点云数据的信息，并且进行保存
    custom_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)
 
    dataset.set_split(val_split)
    # 对验证集的数据进行信息统计并保存
    custom_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info val file is saved to %s' % val_filename)
 
    with open(trainval_filename, 'wb') as f:
        pickle.dump(custom_infos_train + custom_infos_val, f)
    print('Custom info trainval file is saved to %s' % trainval_filename)
 
 
    dataset.set_split('test')
    # kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    custom_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(custom_infos_test, f)
    print('Custom info test file is saved to %s' % test_filename)
 
    
 
    print('---------------Start create groundtruth database for data augmentation---------------')
    # 用trainfile产生groundtruth_database
    # 只保存训练数据中的gt_box及其包围点的信息，用于数据增强    
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(info_path=train_filename, split=train_split)
 
    print('---------------Data preparation Done---------------')

if __name__=='__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['couple'], # 1.修改类别
            data_path=ROOT_DIR / 'data' /'custom4',
            save_path=ROOT_DIR / 'data' /'custom4'
        )

```
#### * 3D点云目标检测跟踪后处理网络 bytetrackv2：




