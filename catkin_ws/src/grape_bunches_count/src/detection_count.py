#!/usr/bin/env python
import rospy,tf,image_geometry
from geometry_msgs.msg import Pose,PoseStamped,PoseArray
from sensor_msgs.msg import Image,CameraInfo
import message_filters
from visualization_msgs.msg import Marker,MarkerArray
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from darknet_ros_msgs.msg import BoundingBoxes
from tracking.centroidtracker import CentroidTracker
from collections import OrderedDict
import numpy,time


class Count(object):
   """
   Counting task steps:
   (1) The robot receive the topic from navigation node to perform navigate,
       it can autonomous navigation around the grape trellis using the pre-established
       topology map and movebase package in ROS
   (2) Detection: receives the bounding_boxes of targets from darknet_Yolov3
   (3) Tracking: calculating euclidean distance to track center of targets from different camera frames
       when robot is moving
   (4) 3D location and counting: according to the center coordinates of targets on color image and
       the corresponding depth on depth image, calculating the 3D position of targets in  world coordinate frame
       by using TF tree in ROS. Meanwhile,using spatial qualified method to filter out target's location noises.
       Finally, robot can complete the counting task and publish the 3D poses of all grape bunches to the rviz interface.
       Also, (1)-(3) are very important for getting more accurate counting result.
   """

   def __init__(self):
      #subscribe camera_info topic
      self.camera_info_sub = rospy.Subscriber('/thorvald_001/kinect2_front_camera/hd/camera_info',
                                              CameraInfo, self.camera_info_callback)
      #wait for the camera_info topic to become available
      rospy.wait_for_message('/thorvald_001/kinect2_front_camera/hd/image_color_rect', Image)

      #subscribe 'image_color_rect' and 'registered_depth_image' by synchronization mode
      self.image_sub = message_filters.Subscriber('/thorvald_001/kinect2_front_camera/hd/image_color_rect', Image)
      self.depth_sub = message_filters.Subscriber('/thorvald_001/kinect2_front_sensor/sd/registered_depth_image', Image)
      self.tss = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub],queue_size=10, slop=0.3)

      #subscribe target's bounding_boxes from darknet_Yolov3
      rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self.boundingBoxCallback,queue_size=10)
      self.tss.registerCallback(self.image_callback)
      self.boxes = None  # initialize boxes to receive bounding boxes of targets from yolov3

      #subscribe 'Execute_count' points and 'No_count' points from navigation node
      self.nav_Sub = rospy.Subscriber('/thorvald_001/nav_points', String, self.nav_callback)
      self.nav_points = ''  # initialize navigation_points to receive information from navigation node

      #initialize publisher for detecting_image, tracking_image, registered depth image with showing target's center
      self.detectionImage_pub = rospy.Publisher("/thorvald_001/kinect2_front_camera/hd/detection_image", Image,queue_size=1)
      self.trackingImage_pub = rospy.Publisher("/thorvald_001/kinect2_front_camera/hd/tracking_image", Image,queue_size=1)
      self.registered_depthImage_showTargets_pub = rospy.Publisher("/thorvald_001/kinect2_front_sensor/sd/registered_depthImage_showTargets",
                                                                   Image,queue_size=1)

      #initialize publisher for the 3D pose of targets in order to count. Meanwhile,initialize publisher for the marker of
      #evevry target to rviz, which can visualize 3D distribution of all targets on the rviz
      self.pub_all = rospy.Publisher('All_target_poses', PoseArray, queue_size=1)
      self.pub_all_markers = rospy.Publisher('All_target_markers', MarkerArray, queue_size=1)
      self.all_poses = PoseArray()  # initialize PoseArray to store the 3D pose of all of targets
      self.all_markers = MarkerArray()  # initialize MarkerArray to store the marker of all of targets

      self.rate = rospy.Rate(1)  # publish message at 1Hz

      self.bridge = CvBridge()  # initialize CvBridge() to convert image from a ROS image message to a CV image
      self.tf_listener = tf.TransformListener()  # create listener for transforms

      self.sample_point_count = 1  #10,set the number of sampling random points
      self.effective_sample_point = 1  #5,set the number of effective sample point to exclude some noises
      self.offset = 0  #5, set the pixle distance when sampling random points

      #initialize the CentroidTracker for tracking the targets.
      #Due to robot alway keeps moving,setting the maxDisappeared frames to 1
      self.ct = CentroidTracker(1)
      self.n = 0  # initialize the variable of camera frame
      self.objects = OrderedDict()  # initialize the variable for tracking the centroid of targets

      self.dist_threshold = 0.1  # set the distance limit(meter) between different grape brunches to exclude some noises

      self.rate.sleep()  # suspend until next cycle


   # Function to get PinholeCameraModel from image_geometry and camera_info
   def camera_info_callback(self, data):
      self.camera_model = image_geometry.PinholeCameraModel()
      self.camera_model.fromCameraInfo(data)
      self.camera_info_sub.unregister()  # Only subscribe once


   # Function to get target's bounding_boxes from darknet_Yolov3
   def boundingBoxCallback(self, data):
      self.boxes = data.bounding_boxes


   # Function to receive the topic from navigation node,
   # due to the kinect has a limited perceptual range and best perceptual range is 0.5 ~ 4.5 m.
   # so there are two type of waypoints: 'Execute_count' and 'No_count'.
   # when robot moves in 'Execute_count' waypoints (these points are close to grape trellis),it executes count.
   # Conversely,it doesn't executes count.
   def nav_callback(self, data):
      self.nav_points = data.data


   # Function for sampling random points from detected box of targets
   # and calculate the coodinate of average points.Sadly, it doesn't work well because
   # there are gaps between grape particles, but I am sure it will work well for the solid object,
   # so keep this piece of coding
   def randomPoints(self, xmin, xmax, ymin, ymax, count):
      result = []
      xoffset = abs(xmax - xmin)
      yoffset = abs(ymax - ymin)
      for i in range(count):
         x = xmin + int(xoffset * numpy.random.uniform(0, 1))
         y = ymin + int(yoffset * numpy.random.uniform(0, 1))
         result.append([x, y])
      return result


   # This callback function handles detection->tracking->3d location->counting for targets.
   def image_callback(self, color_data, depth_data):
      # if camera_model is not available,
      # end the callback at this time until it becomes available
      if self.camera_model is None:
         return

      # if color_image and depth_image is not available,
      # end the callback at this time until it becomes available
      if (color_data is None) or (depth_data is None):
         return

      image = self.bridge.imgmsg_to_cv2(color_data, desired_encoding='bgr8')  # convert 'ROS' color_image to OpenCV image
      track_img = image.copy()  # copy image for the visualizaion of tracking
      registered_depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")  # convert 'ROS' registered depth_image image to OpenCV image
      show_registered_depth_image = registered_depth_image.copy()  # copy image for the visualizaion of registered depth_image

      if (self.nav_points == 'Execute_count'):  # if robot receives 'Execute_count' from navigation node,executing count.
         print('Execute_count')  # print 'Execute_count' on terminal
         cv2.putText(image, 'Execute_count', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255),
                     5) # print 'Execute_count' on image

         bounding_box_list = []  # initialize the list to store the bounding_boxes from darknet_Yolov3
         if self.boxes == None:  # check for at least one target found
            print("No target found")
         else:   # find targets
            ''' (1) First step-> detection: receives the bounding_boxes of targets from darknet_Yolov3 
            and calculate the center coordinates of target. Meanwhile, according to the depth information 
            from registered depth_image, filtering out targets with valid depth information. Finally, 
            adding these targets to the bounding_box_list '''
            for box in self.boxes:
               objClass = box.Class
               prob = box.probability
               xmin = box.xmin
               xmax = box.xmax
               ymin = box.ymin
               ymax = box.ymax
               w=xmax-xmin
               h=ymax-ymin
               target_center_x=xmin+int(w/2.0)
               target_center_y=ymin+int(h/2.0)

               # obtain the depth of target's center from registered depth_image
               D = registered_depth_image[int(target_center_y), int(target_center_x)]

               # check for whether the depth of target's center is valid.
               # And limit the depth range according to best perceptual range of kinect(0.5 ~ 4.5 m)
               if not (numpy.isnan(D) or D == numpy.inf) and D >= 0.5 and D <= 4.5:
                  bounding_box_list.append((target_center_x, target_center_y)) # fill the bounding_box_list
                  cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0),
                                3)  # draw target's bounding_box on color_image
                  cv2.circle(image, (int(target_center_x), int(target_center_y)), 5, (0, 255, 0),
                             -1) # draw target's center on color_image
                  cv2.circle(show_registered_depth_image, (int(target_center_x), int(target_center_y)), 5, 255,
                             -1)  # draw target's center on registered depth_image
                  cv2.putText(image, "grape", (max(xmin, 15), max(ymin - 5, 15)), cv2.FONT_ITALIC, 0.6,
                              (0, 0, 255), 2)  # print the lable of target to the color_image


            ''' (2) Second step-> tracking: calculating euclidean distance to track centroids of targets from 
            different camera frames when robot moves. It is very important step to get 
            more accurate counting number of targets  '''
            old_objects = self.objects  # store target's ID and center from last frame
            old_object_keys = old_objects.keys()  # obtain the key value(also ID) of target's center from last frame
            self.objects = self.ct.update(bounding_box_list)  #use 'update' to obtain target's ID and center from current frame
            new_objects = self.objects # store target's ID and center from current frame

            # Comparing the IDs between the last frame and current frame,
            # if there are new IDs, they are considered as newly detected targets
            new_points = []
            for (objectID, centroid) in new_objects.items():  # loop over objects from current frame
               if (objectID in old_object_keys): #if objectID already existed in the last frame,do nothing
                  continue
               else: #if objectID do not exist in the last frame,it is the newly detected targets
                  new_points.append(list(centroid)) #store all of the newly detected targets

            # draw both the ID of the object and the centroid of the
            # object on the current output frame
            for (objectID, centroid) in self.objects.items():
              text = "ID{}".format(objectID)
              cv2.putText(track_img, text, (centroid[0] - 10, centroid[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
              cv2.circle(track_img, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)

            ''' (3) Third step-> 3D location and counting: 
            (3-1):
             Samples random points from detected box of target and filter out the valid sample points. 
             Then transform valid sample points from 2D_image_coordinate_frame-> 3D_camera_coordinate_frame
             -> 3D_world[map]_coordinate_frame. Finally, calculates average position of valid sample points 
             in world[map] frame as the final position of target. These step apply to every newly detected target. 
            '''
            if len(new_points)>0:
               for i in range(len(new_points)): # loop over newly detected targets
                  centroid_x = new_points[i][0]
                  centroid_y = new_points[i][1]

                  # Sample random points from detected box of every target, and calculate the coodinate of average points later
                  im_pixels = self.randomPoints(centroid_x - self.offset, centroid_x + self.offset,
                                                centroid_y - self.offset, centroid_y + self.offset,
                                                self.sample_point_count)

                  # initialize the 3d points list to store the coordinates of
                  # above sample points on camera coordinate frame
                  points_3d = []

                  # check for whether the depth of these sample points is valid.
                  # And limit the depth range according to best perceptual range of kinect(0.5 ~ 4.5 m)
                  # Then transform valid sample points from 2D image coordinate frame to 3D camera coordinate frame and store.
                  for u, v in im_pixels:
                     D = registered_depth_image[int(v), int(u)]
                     if not (numpy.isnan(D) or D == numpy.inf) and D>=0.5 and D<=4.5: # the best perceptual range for kinect: 0.5 ~ 4.5m
                        cv2.circle(image, (int(u), int(v)), 1, (0, 255, 0), -1)  # draw valid sample point on the color_image
                        camera_coord = self.PxielToCamera(u, v,D) # transform points from image coordinate frame to camera coordinate frame
                        points_3d.append(camera_coord) # store the 3d coordinates of valid sample points on the camera coordinate frame

                  # if the number of valid sample point is less than number of effective sample point,
                  # this newly detected target is not considered effective
                  if len(points_3d) < self.effective_sample_point:
                     continue

                  # initialize the 3d points list to store the coordinates of
                  # above valid sample points in world coordinate frame
                  final_points = []

                  # transform above 3d coordinates of valid sample point
                  # from camera coordinate frame to world[map] coordinate frame and store.
                  for p in points_3d:
                     map_point = self.CameraToWorldFrame(p) # convert to point in [map] frame
                     final_points.append(map_point) ## store the 3d coordinates of valid sample points in world[map] coordinate frame

                  # calculate average position of above valid sample points in world[map] frame, these step is for every newly detected target
                  Xs = [p.pose.position.x for p in final_points]
                  Ys = [p.pose.position.y for p in final_points]
                  Zs = [p.pose.position.z for p in final_points]
                  avgX = sum(Xs) * 1.0 / len(Xs)
                  avgY = sum(Ys) * 1.0 / len(Ys)
                  avgZ = sum(Zs) * 1.0 / len(Zs)

                  # setting 3d point in map frame with position and orientation
                  detected_pose = self.update_target_pose(avgX,avgY,avgZ)

                  ''' (3) third step-> 3D location and counting: 
                  (3-2): Calculate pose distance between the current detected target and every existing targets, 
                  and compare all of distances with threshold in order to determine whether the current detected target
                  overlaps with existing targets. if it doesn't overlap, the current detected target is stored. Otherwise, 
                  it will not be stored. Finally, it can count and obtain the sum_number of all of targets.            
                  '''
                  if self.n == 0:
                     # if it is the first detected target when robot executes count, it will be store directly
                     self.all_poses.poses.append(Pose(detected_pose.pose.position, detected_pose.pose.orientation)) # store the pose of first detected target
                     self.marker(detected_pose, 0.15, 0.15, 0.15, 0, 1, 0, 1)  # generate the maker for first target

                  else:
                     #calculate pose distance between current detected target and every existing targets.
                     # if all of distances>threshold, it confirms that it is a new target, otherwise it overlaps with
                     # existing targets

                     R_list=[] # use to store the all of distances between current detected target and every existing target

                     # calculate pose distance between current detected target and every existing targets
                     for pose in self.all_poses.poses:
                        Rx = abs(pose.position.x - detected_pose.pose.position.x)
                        Ry = abs(pose.position.y - detected_pose.pose.position.y)
                        Rz = abs(pose.position.z - detected_pose.pose.position.z)
                        R = (Rx + Ry + Rz) / 3.0 #calculate the average distance from x,y,z direction
                        R_list.append(R)

                     num=0
                     for i in range(len(R_list)):
                        if R_list[i] > self.dist_threshold: # compare above all of distances with threshold
                           num=num+1
                     if num == len(R_list):
                        # if the number (which the distance > threshold) == the number of all of existing targets
                        # the current detected target is stored.
                        self.all_poses.poses.append(Pose(detected_pose.pose.position, detected_pose.pose.orientation))
                        self.marker(detected_pose, 0.15, 0.15, 0.15, 0, 1, 0, 1)  # generate the maker of every target
                     else:
                        print('Overlapping target')

                  self.n = self.n + 1

            print('target_number:', len(self.all_poses.poses)) # print the number of current targets which is already stored
            #print('\n------------current_frame------------------\n')

            self.update_all_target_pose(self.all_poses) #publish poses of all of targets
            self.markers(self.all_markers) # publish markers(SPHERE) of all of targets in order to visualize the poses of all of targets in the rviz

      else:  # if robot receives 'No_count' from navigation node,it doesn't execute count.
         print('No_count')
         cv2.putText(image, 'No_count', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)  # print 'No_count' on image
         self.markers(self.all_markers)  # publish markers(SPHERE) of all of targets in order to visualize the poses of all of targets in the rviz

      # print target sum_number on detection_image
      count_number = "sum_number:{}".format(len(self.all_poses.poses))
      cv2.putText(image, count_number, (50, 150),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)
      # print 'tracking' label on tracking_image
      cv2.putText(track_img, 'tracking', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)

      # publish the detectionImage, trackingImage, registered depth image with showing target's center
      try:
         self.detectionImage_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
         self.trackingImage_pub.publish(self.bridge.cv2_to_imgmsg(track_img, "bgr8"))
         self.registered_depthImage_showTargets_pub.publish(self.bridge.cv2_to_imgmsg(show_registered_depth_image))
      except CvBridgeError as e:
         print(e)

      #resize the image size and show on windows
      xs = 0.5 # the scale of x axis
      ys = 0.5 # the scale of y axis
      image = cv2.resize(image, (0, 0), fx=xs, fy=ys)
      track_img = cv2.resize(track_img, (0, 0), fx=xs, fy=ys)
      show_registered_depth_image *= 1.0 / 8.0  # scale for visualisation (max range 8.0 m)
      show_registered_depth_image = cv2.resize(show_registered_depth_image, (0, 0), fx=xs, fy=ys)
      cv2.imshow("registered_depth_image", show_registered_depth_image)
      cv2.imshow("Tracking", track_img)
      cv2.imshow("detection", image)
      #cv2.imwrite('result.png',image)
      cv2.waitKey(1)


   # Function for transforming point from image coordinate frame to camera coordinate frame
   # from Greg's lab workshop-> https://github.com/LCAS/CMP9767M/blob/master/uol_cmp9767m_tutorial/scripts/image_projection_3.py
   def PxielToCamera(self, pxX, pxY,d):
      camera_coords = self.camera_model.projectPixelTo3dRay((pxX, pxY)) #project the image coords (x,y) into 3D ray in camera coords
      camera_coords = [x / camera_coords[2] for x in camera_coords]  # adjust the resulting vector so that z = 1
      camera_coords = [x * d for x in camera_coords]  # multiply the vector by depth
      return camera_coords


   # Function for transforming point from camera coordinate frame to world[map] coordinate frame.
   # from Greg's lab workshop-> https://github.com/LCAS/CMP9767M/blob/master/uol_cmp9767m_tutorial/scripts/image_projection_3.py
   def CameraToWorldFrame(self,camera_3d_point):
      # at first, define a point with position and orientation in camera coordinate frame
      object_location = PoseStamped()
      object_location.header.frame_id = "thorvald_001/kinect2_front_rgb_optical_frame"
      object_location.pose.orientation.w = 1.0
      object_location.pose.position.x = camera_3d_point[0]
      object_location.pose.position.y = camera_3d_point[1]
      object_location.pose.position.z = camera_3d_point[2]
      object_location.pose.orientation.x = 0
      object_location.pose.orientation.y = 0
      object_location.pose.orientation.z = 0
      object_location.pose.orientation.w = 1.0
      p_camera = self.tf_listener.transformPose('/map', object_location) #transform to map frame
      return p_camera


      # Function for setting a 3d point in map frame with position and orientation
   def update_target_pose(self, x, y, z):
      pub_pose = PoseStamped()
      # set position values for target location
      pub_pose.pose.position.x = x
      pub_pose.pose.position.y = y
      pub_pose.pose.position.z = z

      # set orientation values for target location
      quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)  # determine quaternion values from euler angles of (0,0,0)
      pub_pose.pose.orientation.x = quaternion[0]
      pub_pose.pose.orientation.y = quaternion[1]
      pub_pose.pose.orientation.z = quaternion[2]
      pub_pose.pose.orientation.w = quaternion[3]
      pub_pose.header.frame_id = "map"
      # self.pub.publish(pub_pose) # publish pose of target
      return pub_pose


   # Function for publishing poses of all of targets
   def update_all_target_pose(self,all_poses):
      all_poses.header.frame_id = "map"
      self.pub_all.publish(all_poses)


   # Function for generating the maker(SPHERE) of every target in order to visiulize the target pose in the rviz
   def marker(self,pose,sx,sy,sz,r,g,b,a):
      marker = Marker()
      marker.header.frame_id = '/map'
      marker.id = 0
      marker.action = Marker.ADD
      marker.ns = "grape_model"
      marker.type = 2  # refer to http://docs.ros.org/en/api/visualization_msgs/html/msg/Marker.html
      marker.pose.position.x = pose.pose.position.x
      marker.pose.position.y = pose.pose.position.y
      marker.pose.position.z = pose.pose.position.z
      marker.pose.orientation.x = 0.0
      marker.pose.orientation.y = 0.0
      marker.pose.orientation.z = 0.0
      marker.pose.orientation.w = 1.0
      marker.scale.x = sx
      marker.scale.y = sy
      marker.scale.z = sz
      marker.color.r = r
      marker.color.g = g
      marker.color.b = b
      marker.color.a = a
      marker.lifetime = rospy.Duration()
      self.all_markers.markers.append(marker) #store the marker to MarkerArray


   # Function for publishing markers(many SPHERES) of all of targets in order to visualize the poses of all of targets in the rviz
   # refer to http://docs.ros.org/en/fuerte/api/rviz/html/marker__array__test_8py_source.html
   def markers(self,markerArray):
      id=0
      for m in markerArray.markers:
         m.id = id
         id += 1
      self.pub_all_markers.publish(markerArray)


if __name__ == '__main__':

   # Start up the count node and run until shutdown by interrupt
   rospy.init_node('target_count', anonymous=True) #initialize ROS node
   try:
      count = Count() #instantiate the class of Count()
      rospy.spin()
   except rospy.ROSInterruptException:
      rospy.loginfo("count node terminated.")
   cv2.destroyAllWindows() #close all terminal windows when process is shut down

