#!/usr/bin/python3

import math
import numpy as np
import cv2
import rospy
import os
print(os.getcwd())
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_32FC1)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_32FC1)

cv2.namedWindow("depth")

# # 创建用于调节参数的bar
# cv2.namedWindow("config", cv2.WINDOW_NORMAL)
# cv2.createTrackbar("minDisparity", "config", 0, 255, lambda x: None)
# cv2.createTrackbar("numDisparities", "config", 1, 255, lambda x: None)
# cv2.createTrackbar("blockSize", "config", 1, 10, lambda x: None) 
# cv2.createTrackbar("P1", "config", 1, 255, lambda x: None) 
# cv2.createTrackbar("P2", "config", 1, 255, lambda x: None)
# cv2.createTrackbar("disp12MaxDiff", "config", 0, 100, lambda x: 49)
# cv2.createTrackbar("preFilterCap", "config", 1, 255, lambda x: None) 
# cv2.createTrackbar("uniquenessRatio", "config", 0, 255, lambda x: None)
# cv2.createTrackbar("speckleWindowSize", "config", 0, 255, lambda x: None) 
# cv2.createTrackbar("speckleRange", "config", 0, 255, lambda x: None)

# 初始化全局变量
left_frame_cache = None
right_frame_cache = None
left_timestamp = None
right_timestamp = None

def image_callback_left(data):
    global left_frame_cache, left_timestamp, right_frame_cache, right_timestamp
    bridge = CvBridge()
    left_frame = bridge.imgmsg_to_cv2(data, "bgr8")
    left_timestamp = data.header.stamp

    if right_frame_cache is not None and left_timestamp == right_timestamp:
        process_images(left_frame, right_frame_cache)
        right_frame_cache = None
    else:
        left_frame_cache = left_frame

def image_callback_right(data):
    global left_frame_cache, left_timestamp, right_frame_cache, right_timestamp
    bridge = CvBridge()
    right_frame = bridge.imgmsg_to_cv2(data, "bgr8")
    right_timestamp = data.header.stamp

    if left_frame_cache is not None and left_timestamp == right_timestamp:
        process_images(left_frame_cache, right_frame)
        left_frame_cache = None
    else:
        right_frame_cache = right_frame

def process_images(left_frame, right_frame):
    # 在这里处理左右目的RGB图像，并执行你的深度计算和显示视差图的逻辑
    # 你可以将你的现有代码逻辑嵌入到这个循环中
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)


    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                numDisparities=32,
                                blockSize=blockSize,
                                P1=8 * img_channels * blockSize * blockSize,
                                P2=32 * img_channels * blockSize * blockSize,
                                disp12MaxDiff=-1,
                                preFilterCap=125,  #值越大，能够容忍的像素亮度差异就越大，这可能有助于保留图像中的一些细节信息，但也可能会增加噪声。相反，值越小，能够容忍的像素亮度差异就越小
                                uniquenessRatio=0,
                                speckleWindowSize=0,
                                speckleRange=0,
                                mode=cv2.STEREO_SGBM_MODE_HH)
    # minDisparity = cv2.getTrackbarPos("minDisparity", "config")
    # numDisparities = cv2.getTrackbarPos("numDisparities", "config")
    # blockSize = cv2.getTrackbarPos("blockSize", "config")
    # P1 = cv2.getTrackbarPos("P1", "config") * img_channels * blockSize * blockSize
    # P2 = cv2.getTrackbarPos("P2", "config") * img_channels * blockSize * blockSize
    # disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "config") - 50
    # preFilterCap = cv2.getTrackbarPos("preFilterCap", "config")
    # uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "config")
    # speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "config")
    # speckleRange = cv2.getTrackbarPos("speckleRange", "config")
    # mode=cv2.STEREO_SGBM_MODE_HH
    # stereo = cv2.StereoSGBM_create(minDisparity,
    #                         numDisparities,
    #                         blockSize,
    #                         P1,
    #                         P2,
    #                         disp12MaxDiff,
    #                         preFilterCap,  #值越大，能够容忍的像素亮度差异就越大，这可能有助于保留图像中的一些细节信息，但也可能会增加噪声。相反，值越小，能够容忍的像素亮度差异就越小
    #                         uniquenessRatio,
    #                         speckleWindowSize,
    #                         speckleRange,
    #                         mode=cv2.STEREO_SGBM_MODE_HH)
    #计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # # 调用 insertDepth32f 函数填充深度图
    # insertDepth32f(disp)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    bridge = CvBridge()
    # 创建一个Image消息
    image_msg = bridge.cv2_to_imgmsg(disparity, encoding="16SC1")
    depth_pb.publish(image_msg)

    # 鼠标回调事件
    cv2.setMouseCallback("depth", onmouse_pick_points, threeD)


    cv2.imshow("depth_hui", disp)
    cv2.imshow("depth", dis_color)
    cv2.waitKey(1)

def onmouse_pick_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            threeD = param
            print('\n像素坐标 x = %d, y = %d' % (x, y))
            print("世界坐标xyz 是: ", threeD[y][x][0] , threeD[y][x][1] , threeD[y][x][2] , "m")
            distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
            #distance = distance / 1000.0
            print("距离是: ", distance, "m")

if __name__ == "__main__":
    rospy.init_node("depth_map_node")
    rospy.Subscriber("/airsim_node/drone_1/front_left/Scene", Image, image_callback_left)
    rospy.Subscriber("/airsim_node/drone_1/front_right/Scene", Image, image_callback_right)
    depth_pb = rospy.Publisher("/airsim_node/drone_1/front_depth", Image, queue_size=10)
    rospy.spin()

    cv2.destroyAllWindows()