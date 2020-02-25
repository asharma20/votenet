# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import glob
import open3d as o3d
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import pc_util
from ap_helper import parse_predictions
from dump_helper import softmax
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd_detection_dataset import DC

class App:
    def __init__(self, args):
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)
        self.flag_exit = False
        self.visualize = args.visualize
        self.save_data = args.save_data
        self.output_dir = args.output_dir
        self.kinect_setup(args)
        self.num_point = args.num_point
        self.DUMP_CONF_THRESH = args.threshold # Dump boxes with obj prob larger than that.
        self.fx, self.fy, self.cx, self.cy = [600.5037841796875, 600.29217529296875, 639.47564697265625, 365.94244384765625]
        self.eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
            'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
            'conf_thresh': self.DUMP_CONF_THRESH, 'dataset_config': DC}

        demo_dir = os.path.join(BASE_DIR, 'demo_files')
        if args.pretrained_model == 'sunrgbd':
            checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        else:
            checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        # Init the model and optimzier
        self.MODEL = importlib.import_module('votenet') # import network module
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
            sampling='seed_fps', num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr).to(self.torch_device)
        logging.info('Constructed model.')

        # Load checkpoint
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logging.info("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))

        self.net.eval() # set model to eval mode (for bn and dp)

    def preprocess_point_cloud(self, point_cloud):
        ''' Prepare the numpy point cloud (N,3) for forward pass '''
        point_cloud = point_cloud[:,0:3] # do not use color for now
        floor_height = np.percentile(point_cloud[:,2],0.99)
        height = point_cloud[:,2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
        point_cloud = pc_util.random_sampling(point_cloud, self.num_point)
        pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
        return pc

    def kinect_setup(self, args):
        o3d.io.AzureKinectSensor.list_devices()

        if args.kinect_config is not None:
            config = o3d.io.read_azure_kinect_sensor_config(args.kinect_config)
        else:
            config = o3d.io.AzureKinectSensorConfig()

        device = args.device
        if device < 0 or device > 255:
            logging.info('Unsupported device id, fall back to 0')
            device = 0

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    def handle_close(self, evt):
        self.flag_exit = True
        logging.info('Close Nod Depth!')
        sys.exit()

    def get_data(self, idx=None):
        rgbd = self.sensor.capture_frame(True)
        if rgbd is None:
            logging.info('Sensor frame is None')
            return None

        rgb = np.asarray(rgbd.color)
        depth = np.asarray(rgbd.depth)

        point_cloud = []
        for v in range(rgb.shape[0]):
            for u in range(rgb.shape[1]):
                color = rgb[v, u, :]
                Z = depth[v, u]
                if Z == 0:
                    continue
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                X = X / 1000
                Y = Y / 1000
                Z = Z / 1000
                point = [X,Y,Z,color[0],color[1],color[2],0]
                point_cloud.append(point)
        point_cloud = np.array(point_cloud)
        return point_cloud

    def get_bbox_wireframe(self, end_points):
        # INPUT
        point_clouds = end_points['point_clouds'].cpu().numpy()
        batch_size = point_clouds.shape[0]

        # NETWORK OUTPUTS
        objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
        pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
        pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

        # OTHERS
        pred_mask = end_points['pred_mask'] # B,num_proposal

        # Dump predicted bounding boxes
        for i in range(batch_size):
                pc = point_clouds[i,:,:]
                objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)
                # Dump various point clouds
                if np.sum(objectness_prob>self.DUMP_CONF_THRESH)>0:
                    num_proposal = pred_center.shape[1]
                    obbs = []
                    for j in range(num_proposal):
                        obb = DC.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                        pred_size_class[i,j], pred_size_residual[i,j])
                        obbs.append(obb)
                    if len(obbs)>0:
                        obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                        mesh_list = pc_util.get_oriented_bbox(obbs[np.logical_and(objectness_prob>self.DUMP_CONF_THRESH, pred_mask[i,:]==1),:])
                        mesh_vert = o3d.utility.Vector3dVector(np.array(mesh_list.vertices))
                        bbox_points = o3d.geometry.OrientedBoundingBox.create_from_points(mesh_vert)
                        bbox_wf = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox_points)
                        colors = [[1, 0, 0] for i in range(len(bbox_wf.lines))]
                        bbox_wf.colors = o3d.utility.Vector3dVector(colors)
                        return bbox_wf

    def get_geometries(self, point_cloud, end_points, idx=0):
        points = point_cloud[:, 0:3]
        colors = point_cloud[:, 3:6]
        c_norm = (colors - colors.min()) / (colors.max() - colors.min())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(c_norm)

        bbox_wf = self.get_bbox_wireframe(end_points)

        if bbox_wf is not None:
            return (pcd, bbox_wf)
        return (pcd,)

    def run(self):
        logging.info('Start running...')
        idx = 0
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        if self.visualize:
            logging.info('Creating visualizer...')
            set_bounding_box = False
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='point cloud', width=1280, height=960)
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector([])
            self.point_cloud.colors = o3d.utility.Vector3dVector([])
            self.bbox = o3d.geometry.LineSet()
            self.bbox.points = o3d.utility.Vector3dVector([])
            self.bbox.lines = o3d.utility.Vector2iVector([])
            self.bbox.colors = o3d.utility.Vector3dVector([])

        while not self.flag_exit:
            point_cloud = self.get_data(idx)
            if point_cloud is None:
                continue
            pc = self.preprocess_point_cloud(point_cloud)
            inputs = {'point_clouds': torch.from_numpy(pc).to(self.torch_device)}
            tic = time.time()
            with torch.no_grad():
                end_points = self.net(inputs)
            toc = time.time()
            logging.info('Inference time: %f'%(toc-tic))
            end_points['point_clouds'] = inputs['point_clouds']
            pred_map_cls = parse_predictions(end_points, self.eval_config_dict)
            logging.info('Finished detection. %d object detected.'%(len(pred_map_cls[0])))

            if self.visualize:
                geometries = self.get_geometries(point_cloud, end_points, idx)
                pcd = geometries[0]
                if len(geometries) == 2:
                    bbox = geometries[1]
                else:
                    bbox = o3d.geometry.LineSet()
                    bbox.points = o3d.utility.Vector3dVector([])
                    bbox.lines = o3d.utility.Vector2iVector([])
                    bbox.colors = o3d.utility.Vector3dVector([])
                self.point_cloud.points = pcd.points
                self.point_cloud.colors = pcd.colors
                self.bbox.points = bbox.points
                self.bbox.lines = bbox.lines
                self.bbox.colors = bbox.colors
                self.point_cloud.transform(flip_transform)
                self.bbox.transform(flip_transform)
                if idx == 0:
                    self.vis.add_geometry(self.point_cloud)
                    self.vis.add_geometry(self.bbox)
                self.vis.update_geometry(self.point_cloud)
                self.vis.update_geometry(self.bbox)
                self.vis.poll_events()
                self.vis.update_renderer()

            if self.save_data:
                print(idx)
                if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
                self.MODEL.dump_results(end_points, self.output_dir, DC, inference_switch=True,
                        DUMP_CONF_THRESH=self.DUMP_CONF_THRESH, idx_beg=idx)
                logging.info('Dumped detection results to folder %s'%(self.output_dir))
            idx += 1
        self.vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kinect_config',
                        default='config/kinect_config.json',
                        type=str, help='input json kinect config')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('--visualize',
                        default=False,
                        action='store_true',
                        help='visualize point cloud and bounding boxes')
    parser.add_argument('--save_data',
                        default=False,
                        action='store_true',
                        help='dump data during inferencing')
    parser.add_argument('--verbose',
                        default=False,
                        action='store_true',
                        help='verbosity')
    parser.add_argument('--num_point',
                        type=int,
                        default=20000,
                        help='number of sampling points of point cloud')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.8,
                        help='confidence threshold, only display bounding boxes with obj prob larger than this')
    parser.add_argument('--pretrained_model',
                        default='sunrgbd',
                        choices=['sunrgbd','scannet'],
                        type=str, help='output folder to save inference')
    parser.add_argument('--output_dir',
                        default='votenet_output',
                        type=str, help='output folder to save inference')
    args = parser.parse_args()
    v = App(args)
    v.run()