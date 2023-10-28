#!/usr/bin/python3

import math
import rospy
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image

# 我的双目相机参数

left_camera_matrix = np.array([[320.0, 0.0, 320.0],
                               [0.0, 320.0, 240.0],
                               [0.0, 0.0, 1.0]])

left_distortion = np.zeros((1, 5), dtype=np.float64)

right_camera_matrix = np.array([[320.0, 0.0, 320.0],
                                [0.0, 320.0, 240.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)

right_distortion = np.zeros((1, 5), dtype=np.float64)

R = np.matrix([
    [1.0000, 0.0000, 0.0000],
    [0.0000, 1.0000, 0.0000],
    [0.0000, 0.0000, 1.0000]], dtype=np.float64)

T = np.array([0.0, 0.095, 0.0], dtype=np.float64)  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

cv2.namedWindow("win", cv2.WINDOW_NORMAL)

def show_img(img):
    cv2.imshow('win',img)
    # # 从视差地图获取深度地图
    # depth_map = (320 * 0.095) / img
    # depth_map = depth_map * 16
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(img, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16
    # 鼠标回调事件
    cv2.setMouseCallback("win", onmouse_pick_points, threeD)
    cv2.waitKey(20)

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

img_l=None
img_r=None
seq_l=None
seq_r=None

bridge=cv_bridge.CvBridge()

def left(image:Image):
    seq=image.header.seq
    if seq_r==seq:
        show_img(depth_map(bridge.imgmsg_to_cv2(image),bridge.imgmsg_to_cv2(img_r)))
    else:
        global img_l,seq_l
        img_l=image
        seq_l=seq
    # print(f'left {image.header.stamp.secs}.{image.header.stamp.nsecs} {image.header.seq}')
def right(image:Image):
    seq=image.header.seq
    if seq_l==seq:
        show_img(depth_map(bridge.imgmsg_to_cv2(img_l),bridge.imgmsg_to_cv2(image)))
    else:
        global img_r,seq_r
        img_r=image
        seq_r=seq
    # print(f'right {image.header.stamp.secs}.{image.header.stamp.nsecs} {image.header.seq}')

def onmouse_pick_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_map = param
            print('\n像素坐标 x = %d, y = %d' % (x, y))
            print("世界坐标xyz 是: ", depth_map[y][x][0] , depth_map[y][x][1] , depth_map[y][x][2] , "m")
            distance = math.sqrt(depth_map[y][x][0] ** 2 + depth_map[y][x][1] ** 2 + depth_map[y][x][2] ** 2)
            #distance = distance / 1000.0
            print("距离是: ", distance, "m")

rospy.init_node('parallax_node')
rospy.Subscriber('/airsim_node/drone_1/front_left/Scene',Image,left)
rospy.Subscriber('/airsim_node/drone_1/front_right/Scene',Image,right)
rospy.spin()

