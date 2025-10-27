# CMP9767M  

The source code provided here implements the methods described in the paper "Automatic Detection, Positioning and Counting of Grape Bunches Using Robots"

<p align="center">
  <img src="https://github.com/XuminGaoGithub/Grape_bunches_count_using_robots/blob/master/catkin_ws/Demo_grape_bunches_count.gif" width="1500" alt="Demo" />
</p>
                                                              

## Solution
My focus area is **Perception**. In this task, the robot can **accurately count all grape bunches in the vineyard and obtain the 3D spatial coordinates of all grape bunches**. Firstly, the robot uses **yolov3** to detect grape bunches, which can effectively solve the detection of multiple targets with partially overlapping regions in the image. Then calculating euclidean distance to **track** center of targets from different camera frames when robot is moving. After that, according to the target's center coordinates on color image and the corresponding depth distance on depth image, calculating target's **3D position** in world coordinate frame by using TF tree in ROS. Meanwhile, using **spatial qualified method** to filter out target's location noises. Finally, the robot can complete the counting task and publish the 3D poses of all grape bunches to the rviz interface. For the navigation, the robot realizes **autonomous navigation** around the grape trellis using the pre-established topology map and movebase package in ROS. **The github page** can be found https://github.com/XuminGaoGithub/Grape_bunches_count_using_robots.



## Prerequisites

**1.** Ensure your system is Ubuntu18.04 with Python2. Meanwhile, installs L-CAS Ubuntu/ROS software distribution:  https://github.com/LCAS/CMP9767M/wiki/useful-resources#install-the-l-cas-ubuntu-distribution-for-the-module-on-your-own-pc  

**2.** This work is based on https://github.com/LCAS/CMP9767M. Clone it into your catkin_ws workspace.  

**3.** Ensure your system has installed cuda_11.5 and cudnn_7.6.5 which can run Yolov3. If you use other versoins of cuda and cudnn, please follow the instruction https://github.com/leggedrobotics/darknet_ros to compile the darknet_ros package.

**4.** Ensure your system has installed opencv_4.1.0 



## Installation and Run

**1. Clone repo and recompile**   
**Clone** my repository into your workspace. And recompile it:  
`cd catkin_ws/`  
`catkin_make`  
`~/catkin_ws/devel/setup.bash`  

Meanwhile, please follow the instruction https://github.com/leggedrobotics/darknet_ros , if you encounter some compilation problems in the part of darknet_ros. Also, this tutorial (https://github.com/LCAS/CMP9767M/wiki/Workshop-4---Robot-Vision) will be helpful for establishing darknet_ros.

Before you recompile, please **empty ./catkin_ws/build folder**.  

**Noting:**

* If you cann't find the weights model (which was trained to detect grape bunches using Yolov3) in the directory (./catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/), you need to download it from the following link: 

  https://github.com/XuminGaoGithub/Grape_bunches_count_using_robots/blob/master/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/my_yolov3_14000.weights
 

* The link of training dataset as below:

  https://github.com/XuminGaoGithub/Grape_bunches_count_using_robots/tree/master/catkin_ws/src/darknet_ros/darknet/data/voc2007

  

**2. Introduction of main file directories**  

**./catkin_ws/Demo_grape_bunches_count.gif**    
This is a video recording which can view the whole of counting execution process using robots. 

* For a brief description of the counting results, the number of grape bunches counted by the robot in the Demo_grape_bunches_count.gif  is **84**, but actually this number is not fixed. Because, although the robot automatically navigates according to the same topology node every time, the local path planning between nodes is not exactly the same every time. This also leads to the fact that the images obtained by the robot's camera are not exactly the same every time; In addition, due to the self error of the robot's odometry, the position is not exactly the same at the same point in every time of task, and there will lead to subtle differences in the 3D position of the target in every time of task. 

  So now, it can explain why the counting results of the robot are different every time. I did ten times experiments, and take the average of these results (86, 89, 77, 78, 79, 89, 84, 88, 87, 90). The average is **84.7**. Please execute a few more times and you will see some different counting results around 80, but I believe the real result should be between 80-90.

**./catkin_ws/src/CMP9767M-master**    
This is a basic package for establishing the environment of this project. From https://github.com/LCAS/CMP9767M

**./catkin_ws/src/darknet_ros**    
This is a Ros package for darknet, where robot will use Yolov3 to implement the detection of grape bunches and publish detection information including bounding_boxes, classes, detection probability. From https://github.com/leggedrobotics/darknet_ros

**./catkin_ws/src/grape_bunches_count**    
This is the core function implementation package of this project, I will introduce several main Python files below:   
* ./catkin_ws/src/grape_bunches_count/src/detection_count.py  
   *This Python file is the core program to realize detection, tracking, 3D location and counting of grape bunches, and make robot realize autonomous navigation.*  


* ./catkin_ws/src/grape_bunches_count/src/simple_depth_register_node.py  
   *This Python file is the node which is used to register color image and depth image, so as to obtain the depth distance of target's center.*
   
* ./catkin_ws/src/grape_bunches_count/src/navigation.py  
   *This Python file is topology_navigation node.*  
   



**3. Install packages**  
Update: `sudo apt-get update && sudo apt-get upgrade`  
then install following packages:  

```
sudo apt-get install ros-melodic-opencv-apps ros-melodic-rqt-image-view ros-melodic-image-geometry ros-melodic-uol-cmp9767m-base ros-melodic-uol-cmp9767m-tutorial ros-melodic-find-object-2d ros-melodic-video-stream-opencv ros-melodic-topic-tools ros-melodic-rqt-tf-tree ros-melodic-image-view ros-melodic-robot-localization ros-melodic-thorvald ros-melodic-velodyne-description ros-melodic-kinect2-description ros-melodic-topological-navigation ros-melodic-teleop-tools ros-melodic-fake-localization ros-melodic-carrot-planner ros-melodic-amcl ros-melodic-topological-navigation-msgs ros-melodic-topological-utils ros-melodic-strands-navigation ros-melodic-gmapping ros-melodic-robot-pose-publisher
```



**4. Run**  
There are two methods to do it.

 **1) One way**

Creat mongodb file to store the data of topology map and topology navigation:    
`cd && mkdir mongodb`  
Launch gazebo simulation environment:    
`roslaunch bacchus_gazebo vineyard_demo.launch world_name:=vineyard_small`  
Launch topology_navigation node:    
`roslaunch uol_cmp9767m_tutorial topo_nav.launch`  
Load map into mongodb(only once):  
`rosrun topological_utils load_yaml_map.py $(rospack find grape_bunches_count)/maps/map.yaml`  
Launch topology_navigation rviz:  
`rviz -d $(rospack find uol_cmp9767m_tutorial)/config/topo_nav.rviz`  
Launch darknet_ros:    
`roslaunch darknet_ros yolo_v3.launch`    
Run registration node which is used to register color image and depth image:      
`rosrun grape_bunches_count simple_depth_register_node.py`    
Run count:    
`rosrun grape_bunches_count detection_count.py`  
Run autonomous navigation:  
`rosrun grape_bunches_count navigation.py` 

**Finally, don't forget to start up your gazebo simulation.**

After runing all of above commands, you can do **add->By topic->MarkerArray** in the rviz. You can see that the poses of targets are visualized as markers(SPHERE) in the rviz.   




 **2) Another way**

There is a quick way to run above all of commands (If this is first time to run, you have to use 1) to load map into mongodb), at first:      
`roslaunch grape_bunches_count run.launch`  

 ***Waiting for gazebo simulatation is ready and start up it***, then  
`roslaunch grape_bunches_count topo_nav.launch`  

The reason why needs to do it by two steps is that topo_nav.rviz must be launched after topo_nav.launch taking effect. Otherwise, the rviz display of the topology map is a little abnormal, although it does not affect the whole execution process.  



**5. Tips**:  

* Setup.bash to your ~/.bashrc  
`sudo gedit ~/.bashrc`   
Add the lines to the bottom of the file  
`source /home/<USERNAME>/catkin_ws/devel/setup.bash`  
Don't forget to source your environment
`source ~/.bashrc`    


* Before you restart the simulation, run `killall -9 gzserver` on terminal window in order to ensure all simulator instances have been terminated. 



## Contact

Xumin Gao (https://github.com/XuminGaoGithub/Grape_bunches_count_using_robots)
