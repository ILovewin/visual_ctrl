#!/usr/bin/python3

import rospy
import numpy as np
import math
import std_msgs.msg
import cv_bridge
import cv2
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
from airsim_ros.msg import Circle, CirclePoses, CircleCenter

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

state = 1
last_yaw = 0.0

def write_to_file(var1, var2):
    with open('/home/zenyeah/IntelligentUAVChampionshipBase-RMUA2023/pos_ctrl/src/visual/scripts/depth.txt', 'a') as f:
        f.write(str(var1) + '\n')

    with open('/home/zenyeah/IntelligentUAVChampionshipBase-RMUA2023/pos_ctrl/src/visual/scripts/time.txt', 'a') as f:
        f.write(str(var2) + '\n')

def get_ring_orthogonal(mask):
    global last_yaw
    global trusted_depth
    height = mask.shape[0]
    width = mask.shape[1]
    result = CircleCenter()

    result.header.frame_id = "pixel"
    result.header.stamp = rospy.Time.now()
    result.position.z = 0.0
    result.position.y = height // 2
    result.position.x = width // 2
    result.yaw = 0
    center_x = 0
    center_y = 0

    try:
        # 找到所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # 初始化一个列表来保存所有轮廓的周长
        perimeters = []

        # 遍历所有轮廓
        for contour in contours:
            # 计算这个轮廓的周长
            perimeter = cv2.arcLength(contour, True)

            # 如果周长大于130，将这个周长添加到列表中
            if perimeter > 130:
                perimeters.append(perimeter)

        # 根据周长对轮廓进行排序
        sorted_indices = sorted(range(len(perimeters)), key=lambda i: perimeters[i])

        # 如果列表长度大于等于2，再提取第一第二的周长
        if len(perimeters) >= 2:
            # 根据周长对轮廓进行排序
            sorted_indices = sorted(range(len(perimeters)), key=lambda i: perimeters[i])

            # 提取周长最大和第二大的两个轮廓
            largest_contour = contours[sorted_indices[-1]]
            second_largest_contour = contours[sorted_indices[-2]]

            print('The largest contour has a perimeter of:', perimeters[sorted_indices[-1]])

            # 获取最大轮廓的位置信息
            x, y, rect_width, rect_height = cv2.boundingRect(largest_contour)

            # 计算中心点坐标
            center_x = x + rect_width / 2
            center_y = y + rect_height / 2

            # 找到所有白色像素的坐标
            white_pixels = np.where(mask == 255)

            # white_pixels现在是一个包含两个数组的元组，分别表示白色像素的y坐标和x坐标
            y_coords, x_coords = white_pixels

            depth_values = threeD[y_coords, x_coords, 2]

            depth_values = np.abs(depth_values)

            filtered_values = depth_values[depth_values <= 7]
            count = len(filtered_values)
            if count == 0:
                return result
            else:
                result.position.z = filtered_values.sum() / count
                if result.position.z < 1:
                    result.position.z = 0
                    return result



        else:
            print('Not enough contours to extract the two largest.')
            return result

    except cv2.error as e:
            rospy.logwarn(str(e))

    if(state != 4):
        result.position.x = center_x
        result.position.y = center_y

    else:
        # 找到轮廓中y值最小的点
        top_vertex = min(largest_contour, key=lambda point: point[0][1])

        # 计算中心点与y最小点的连线上距离中心的距离是中心的与y最小点距离的三分之一的点坐标
        result.position.x = center_x + (top_vertex[0][0] - center_x) / 3
        result.position.y = center_y + (top_vertex[0][1] - center_y) / 3

    j = int(center_y)
    left_depth, right_depth = 0.0, 0.0
    left_count, right_count = 0, 0
    for i in range(mask.shape[1]):
        if mask[j, i] > 0:
            point_depth = abs(depth_by_color_image_scale(i, j))
            if(point_depth > 8):
                continue
            if i < result.position.x:
                left_depth = left_depth + point_depth
                left_count += 1
            else:
                right_depth = right_depth + point_depth
                right_count += 1

    if left_count != 0 and right_count != 0:

        left_depth = float(left_depth / left_count)
        right_depth = float(right_depth / right_count)
        
        # 已知的对边长度和斜边长度
        opposite_side_length = left_depth - right_depth
        hypotenuse_length = 1.2

        if(abs(opposite_side_length) > hypotenuse_length):
            result.yaw = last_yaw

        else:
            # 计算角度（以弧度为单位）
            angle_radians = math.asin(opposite_side_length / hypotenuse_length)
            # 将弧度转换为度
            angle_degrees = math.degrees(angle_radians)
            print("角度（以度为单位）:", angle_degrees)
            result.yaw = angle_degrees

        last_yaw = result.yaw

    
    rospy.loginfo("the center point is: %f %f %f %f", result.position.x, result.position.y, result.position.z, result.yaw)

    #write_to_file(result.position.z, result.header.stamp)
    
    return result


def depth_by_color_image_scale(i, j):
    return threeD[j][i][2]


# rgb_frame = cv2.imread('/home/zenyeah/Documents/AirSim/2023-10-08-15-18-01/images/img_drone_1_0_0_1696749667963708000.png')
def update_view_mask(rgb_frame):
    hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)

    lower_bound1 = np.array([0, 150, 80]) #red1
    upper_bound1 = np.array([5, 255, 255])
    lower_bound2 = np.array([170, 150, 30]) #red2
    upper_bound2 = np.array([180, 255, 255])
    lower_bound3 = np.array([156, 100, 30]) #red3
    upper_bound3 = np.array([180, 255, 255])
    lower_bound4 = np.array([26, 20, 120])  # yellow
    upper_bound4 = np.array([34, 255, 255])
    lower_bound5 = np.array([0, 60, 200]) #yellow mix red mix orange
    upper_bound5 = np.array([10, 200, 255])

    if(state == 1 or state == 3):
        mask1 = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
        mask3 = cv2.inRange(hsv_frame, lower_bound3, upper_bound3)
        mask5 = cv2.inRange(hsv_frame, lower_bound5, upper_bound5)
        mask = cv2.bitwise_or(mask3, cv2.bitwise_or(mask1, mask5))
        element = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))

    elif(state == 2):
        mask4 = cv2.inRange(hsv_frame, lower_bound4, upper_bound4)
        mask = mask4
        element = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))

    else:
        mask1 = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
        mask3 = cv2.inRange(hsv_frame, lower_bound3, upper_bound3)
        mask = cv2.bitwise_or(mask1, mask3)
        element = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)

    cv2.imshow("view_mask", mask)
    cv2.waitKey(1)
    
    return mask

bridge=cv_bridge.CvBridge()

def left(image:Image):
    mask = update_view_mask(bridge.imgmsg_to_cv2(image)) 
    result = get_ring_orthogonal(mask)
    if result.position.z > 0:
        cam_point = CircleCenter()
        cam_point.header.stamp = rospy.Time.now()
        cam_point.position.x = result.position.z
        cam_point.position.y = (result.position.x - left_camera_matrix[0, 2]) * result.position.z / left_camera_matrix[0, 0]
        cam_point.position.z = (result.position.y - left_camera_matrix[1, 2]) * result.position.z / left_camera_matrix[1, 1]
        cam_point.yaw = result.yaw
        cam_pub.publish(result)

def process(State):
    global state
    state = State

def depth_map(msg):
    try:
    # 将ROS Image消息转换为OpenCV格式
        disparity = bridge.imgmsg_to_cv2(msg, desired_encoding="16SC1")
    except CvBridgeError as e:
        print(e)
    global threeD
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16


if __name__ == "__main__":   
    rospy.init_node('visual_identify_node')
    rospy.Subscriber('/airsim_node/drone_1/front_depth', Image, depth_map)
    rospy.Subscriber('/airsim_node/drone_1/front_left/Scene', Image, left)
    rospy.Subscriber('/airsim_node/drone_1/state', std_msgs.msg.Int32, process)
    cam_pub = rospy.Publisher('camPoint', CircleCenter, queue_size=10)
    rospy.spin()






