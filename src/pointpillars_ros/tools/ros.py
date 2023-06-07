#!/usr/bin/env python3
 
import rospy
 
from sensor_msgs.msg import PointCloud2,PointField
 
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
 
import time
import numpy as np
from pyquaternion import Quaternion
 
import argparse
import glob
from pathlib import Path
 
import numpy as np
import torch
import scipy.linalg as linalg
 
import sys
sys.path.append("/home/d/pointpillars_ros/src/pointpillars_ros")
 
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
 
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
 
 
 
class Pointpillars_ROS:
    def __init__(self):
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)
 
 
    def init_ros(self):
        """ Initialize ros parameters """
        config_path = rospy.get_param("/config_path", "/home/d/pointpillars_ros/src/pointpillars_ros/tools/cfgs/kitti_models/pointpillar.yaml")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/d/pointpillars_ros/src/pointpillars_ros/tools/pointpillar_7728.pth")
        # # subscriber
        # self.sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.lidar_callback, queue_size=1,
        #                                  buff_size=2 ** 12)
        #
        # # publisher
        # self.sub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
        # self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)
        return config_path, ckpt_path
 
 
    def init_pointpillars(self, config_path, ckpt_path):
        """ Initialize second model """
        logger = common_utils.create_logger() # 创建日志
        logger.info('-----------------Quick Demo of Pointpillars-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件
 
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        # 加载权重文件
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式
 
 
    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix
 
 
    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """
        pcl_msg = pc2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=True)
        #这里的field_names可以不要，不要就是把数据全部读取进来。也可以用field_names = ("x", "y", "z")这个只读xyz坐标
        #得到的pcl_msg是一个generator生成器，如果要一次获得全部的点，需要转成list
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        #print(np_p.shape)
        # 旋转轴
        #rand_axis = [0,1,0]
        #旋转角度
        #yaw = 0.1047
        #yaw = 0.0
        #返回旋转矩阵
        #rot_matrix = self.rotate_mat(rand_axis, yaw)
        #np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T
 
        # convert to xyzi point cloud
        x = np_p[:, 0].reshape(-1)
        #print(np.max(x),np.min(x))
        y = np_p[:, 1].reshape(-1)
        z = np_p[:, 2].reshape(-1)
        if np_p.shape[1] == 4: # if intensity field exists
            i = np_p[:, 3].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        points = np.stack((x, y, z, i)).T
        # 组装数组字典
        input_dict = {
            'frame_id': msg.header.frame_id,
            'points': points
        }
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict) # 数据预处理
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict) # 将数据放到GPU上
        pred_dicts, _ = self.model.forward(data_dict) # 模型前向传播
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = pred_dicts[0]['pred_boxes'][mask].detach().cpu().numpy()
        label = pred_dicts[0]['pred_labels'][mask].detach().cpu().numpy()
        num_detections = boxes_lidar.shape[0]
 
        arr_bbox = BoundingBoxArray()
        for i in range(num_detections):
            bbox = BoundingBox()
 
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2]) #+ float(boxes_lidar[i][5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]
 
            arr_bbox.boxes.append(bbox)
 
        arr_bbox.header.frame_id = msg.header.frame_id
        #arr_bbox.header.stamp = rospy.Time.now()
 
        if len(arr_bbox.boxes) is not 0:
            pub_bbox.publish(arr_bbox)
            self.publish_test(points, msg.header.frame_id)
 
    def publish_test(self, cloud, frame_id):
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = frame_id
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]  # ,PointField('label', 16, PointField.FLOAT32, 1)
        #creat_cloud不像read，他必须要有fields,而且field定义有两种。一个是上面的，一个是下面的fields=_make_point_field(4)
        msg_segment = pc2.create_cloud(header = header,fields=fields,points = cloud)
 
        pub_velo.publish(msg_segment)
        #pub_image.publish(image_msg)
 
 
def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str_('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)
 
    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str_('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)
 
    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str_('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)
 
    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str_('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)
 
    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]
 
    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str_('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)
 
    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]
if __name__ == '__main__':
    global sec
    sec = Pointpillars_ROS()
 
    rospy.init_node('pointpillars_ros_node', anonymous=True)
 
    # subscriber
 
    sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, sec.lidar_callback, queue_size=1,
                                    buff_size=2 ** 12)
 
 
    # publisher
    pub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
    pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)
 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")