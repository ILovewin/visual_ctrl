# visual_ctrl
2023无人机智能感知竞赛的视觉识别与pd控制的初版代码

## 项目架构

- visual：视觉模块

  --scripts：存放视觉脚本文件

  ​	--depth.py：深度图

  ​	--identify.py：结合深度信息以及图像处理提取圆环信息

  ​	--parallax.py：视差图

- pos_ctrl：官方提供的位置发送控制指令包

- airsim_ros：与仿真通讯的依赖包以及pid控制器

## 问题说明

1. ### 圆环识别

   ```
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
           
   # 提取周长最长的轮廓
   	largest_contour = contours[sorted_indices[-1]]
   ```

   代码中这样处理会出现一个问题：当mask图像有噪声点时，`cv2.arcLength()`函数提取的最大周长轮廓会是出现的噪声点，所以不能用

   `cv2.arcLength()`函数来提取圆环轮廓，建议通过提取轮廓外接矩形的长宽来提取所需轮廓，如下：

   ```
   # 找到所有轮廓
   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
   max_area = 0
   max_contour = None
   
   # 遍历所有轮廓
   for contour in contours:
       # 获取最大轮廓的位置信息
       _, _, rect_width, rect_height = cv2.boundingRect(contour)
       area = rect_width * rect_height
       if area > max_area:
           max_area = area
           max_contour = contour
   ```

2. ### 深度处理

   经过测试，20帧图像提取的深度信息可用的只有几帧，而且可信度不是很高，所以并没有采用这版代码的深度图像，后面我们团队改用了solvepnp算法去计算无人机与圆环的距离，效果还行，如果仍想继续采用深度图去提取距离，可尝试收集无人机飞行的三轴线速度，时间戳，yaw偏转等参数去拟合一个多元回归模型，结合可信位置的深度信息去估算不可信位置对应的深度信息