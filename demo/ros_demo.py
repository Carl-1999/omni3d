import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

from mySORT_3d import *
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

import math

import message_filters
from geometry_msgs.msg import PoseStamped


logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util, vis

from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone


class ROSImageSubscriber:
	def __init__(self, model, cfg, focal_length, principal_point, threshold, display=False):
		self.model = model
		self.cfg = cfg
		self.bridge = CvBridge()
		self.focal_length = focal_length
		self.principal_point = principal_point
		self.threshold = threshold
		self.display = display
		self.T =  np.array([
			[0, 0, 1],  # 将 z 轴映射为 x 轴
			[-1, 0, 0],  # 将 x 轴映射为 y 轴
			[0, 1, 0]   # 将 y 轴映射为 z 轴
		])

		self.class_names = ('pedestrian', 'car', 'cyclist', 'van', 'truck',
							'traffic cone', 'barrier', 'motorcycle', 'bicycle',
							'bus', 'trailer', 'books', 'bottle', 'camera', 'cereal box',
							'chair', 'cup', 'laptop', 'shoes', 'towel', 'blinds', 'window',
							'lamp', 'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door',
							'toilet', 'desk', 'box', 'bookcase', 'picture', 'table', 'counter',
							'bed', 'night stand', 'pillow', 'sofa', 'television', 'floor mat',
							'curtain', 'clothes', 'stationery', 'refrigerator', 'bin', 'stove', 'oven', 'machine')

		category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')

		if category_path.startswith(util.CubeRCNNHandler.PREFIX):
			category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

		metadata = util.load_json(category_path)
		self.cats = metadata['thing_classes']
		print(f"cats : {self.cats}")

		# 使用 message_filters 订阅图像和位姿话题
		# image_sub = message_filters.Subscriber("/camera/rgb/image_color", Image)
		image_sub = message_filters.Subscriber("/camera/rgb/image_processed", Image)
		pose_sub = message_filters.Subscriber("/vehicle_pose", PoseStamped)

		# 创建时间同步器，设置队列长度和时间容忍度（slop）
		self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, pose_sub], queue_size=10, slop=0.1)
		self.ts.registerCallback(self.callback)

		rospy.Subscriber("/camera/rgb/image_processed", Image, self.callback_image, queue_size=1)
		rospy.Subscriber('/vehicle_pose', PoseStamped, self.callback_pose,queue_size=1)

		self.publisher = rospy.Publisher("/detection_results", String, queue_size=10)  # 发布检测结果到 "/detection_results"

		self.pub_box_3D = rospy.Publisher('/Boxes_3D', BoundingBoxArray, queue_size=10)
		self.pub_label_3D = rospy.Publisher('/label_3D', MarkerArray, queue_size=10)

		self.pub_box_grid = rospy.Publisher('/Boxes_grid', BoundingBoxArray, queue_size=10)
		self.pub_label_grid = rospy.Publisher('/label_grid', MarkerArray, queue_size=10)

		self.pub_box_object = rospy.Publisher('/Boxes_object', BoundingBoxArray, queue_size=10)
		self.pub_label_object = rospy.Publisher('/label_object', MarkerArray, queue_size=10)
		print(f"caaaaaaaaaaaaaaa")

	def callback_pose(self,msg):
		# 打印位置和方向
		position = msg.pose.position
		orientation = msg.pose.orientation

		rospy.loginfo("Received Pose:")
		rospy.loginfo("Position: x=%f, y=%f, z=%f", position.x, position.y, position.z)
		rospy.loginfo("Orientation: x=%f, y=%f, z=%f, w=%f", orientation.x, orientation.y, orientation.z, orientation.w)
		rospy.loginfo(f" Pose timestamp: {msg.header.stamp}")

	def callback_image(self, msg):
		# 将 ROS 图像消息转换为 OpenCV 图像
		cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		rospy.loginfo(f"Image timestamp: {msg.header.stamp}")

	def callback(self, image_msg, pose_msg):

		rospy.loginfo("Callback received image and pose")  # 打印日志

		# # 将 ROS 图像消息转换为 OpenCV 图像
		# cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
		#
		# # 获取位姿信息
		# vehicle_pose = pose_msg.pose
		# rospy.loginfo(
		# 	f"Vehicle Pose: Position - x: {vehicle_pose.position.x}, y: {vehicle_pose.position.y}, z: {vehicle_pose.position.z}, "
		# 	f"Orientation - x: {vehicle_pose.orientation.x}, y: {vehicle_pose.orientation.y}, z: {vehicle_pose.orientation.z}, w: {vehicle_pose.orientation.w}")
		# self.process_image(cv_image, vehicle_pose)

	def process_image(self, im, vehicle_pose):
		# 用车辆位姿进行处理
		# 您可以将位姿信息传递给检测流程以进行进一步的处理或使用
		self.model.eval()

		focal_length = self.focal_length
		principal_point = self.principal_point
		thres = self.threshold

		min_size = self.cfg.INPUT.MIN_SIZE_TEST
		max_size = self.cfg.INPUT.MAX_SIZE_TEST
		augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

		image_shape = im.shape[:2]  # h, w
		h, w = image_shape

		if focal_length == 0:
			focal_length_ndc = 4.0
			focal_length = focal_length_ndc * h / 2

		if len(principal_point) == 0:
			px, py = w / 2, h / 2
		else:
			px, py = principal_point

		K = np.array([
			[focal_length, 0.0, px],
			[0.0, focal_length, py],
			[0.0, 0.0, 1.0]
		])

		aug_input = T.AugInput(im)
		_ = augmentations(aug_input)
		image = aug_input.image

		batched = [{
			'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
			'height': image_shape[0], 'width': image_shape[1], 'K': K
		}]

		dets = self.model(batched)[0]['instances']
		n_det = len(dets)

		# Create and publish 3D boxes and labels based on detections
		boxes_3D = BoundingBoxArray()
		boxes_3D.header.stamp = rospy.Time.now()
		boxes_3D.header.frame_id = 'ground'
		markers_3D = MarkerArray()

		# Process detections and create bounding boxes
		if n_det > 0:
			detection_results = []
			for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
					dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions,
					dets.pred_pose, dets.scores, dets.pred_classes
			)):

				if score < thres:
					continue

				# Add 3D bounding boxes to the array
				box_3D = BoundingBox()
				box_3D.pose.position.x = center_cam[0].item()
				box_3D.pose.position.y = center_cam[1].item()
				box_3D.pose.position.z = center_cam[2].item()
				box_3D.dimensions.x = dimensions[0].item()
				box_3D.dimensions.y = dimensions[1].item()
				box_3D.dimensions.z = dimensions[2].item()
				box_3D.label = cat_idx
				boxes_3D.boxes.append(box_3D)

				# Create a marker for visualization
				marker_3D = Marker()
				marker_3D.header.frame_id = 'ground'
				marker_3D.type = Marker.TEXT_VIEW_FACING
				marker_3D.text = self.class_names[cat_idx]
				marker_3D.pose.position = box_3D.pose.position
				markers_3D.markers.append(marker_3D)

			self.pub_box_3D.publish(boxes_3D)
			self.pub_label_3D.publish(markers_3D)

def setup(args):
	"""
	Create configs and perform basic setups.
	"""
	cfg = get_cfg()
	get_cfg_defaults(cfg)

	config_file = args.config_file

	if config_file.startswith(util.CubeRCNNHandler.PREFIX):
		config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

	cfg.merge_from_file(config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()
	default_setup(cfg, args)

	return cfg


def main(args):
	rospy.init_node("image_subscriber_node")
	cfg = setup(args)
	# print(cfg)
	model = build_model(cfg)

	logger.info("Model:\n{}".format(model))
	DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
		cfg.MODEL.WEIGHTS, resume=True
	)
	# 初始化Sort维护tracker
	mot_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)

	ROSImageSubscriber(
		model=model,
		cfg=cfg,
		focal_length=args.focal_length,
		principal_point=args.principal_point,
		threshold=args.threshold,
		display=args.display
	)

	rospy.spin()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
	parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
	parser.add_argument("--principal-point", type=float, default=[], nargs=2,
						help="principal point for image inputs (in px)")
	parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
	parser.add_argument("--display", default=False, action="store_true",
						help="Whether to show the images in matplotlib")

	parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

	args = parser.parse_args()

	main(args)
#export LD_LIBRARY_PATH=/home/gtf/anaconda3/envs/cubercnn/lib:$LD_LIBRARY_PATH