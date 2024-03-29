#!/usr/bin/env python2

import rospy,image_geometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy
import tf
from geometry_msgs.msg import Pose,PoseStamped,PoseArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from simple_depth_register_node import DepthRegisterNode
import message_filters
from geometry_msgs.msg import Point
from math import cos, sin
import time
from visualization_msgs.msg import Marker,MarkerArray
from collections import OrderedDict
from pyimagesearch.centroidtracker import CentroidTracker
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes
import ros_numpy

ct = CentroidTracker(1) #3 frames for keep tracking

class Detector(object):
   """ Detector for the target -- identified by pink marker """
   """ Target's location is published as a PoseStamped message with its x, y, z with respect to  """
   """   the Kinect v2's image """
   #color2depth_aspect = (84.1 / 1920) / (70.0 / 512) #(3)


   def __init__(self):

      # initialize ROS node
      rospy.init_node('target_detector', anonymous=True)

      # initialize publisher for target pose, PoseStamped message, and set initial sequence number
      self.pub = rospy.Publisher('Target_pose', PoseStamped, queue_size=1)
      self.pub_marker = rospy.Publisher('Target_marker', Marker, queue_size=1)
      self.pub_all = rospy.Publisher('All_target_poses', PoseArray, queue_size=1)
      self.pub_all_markers = rospy.Publisher('All_target_markers', MarkerArray, queue_size=1)
      #self.pub_pose = PoseStamped()
      #self.pose = PoseStamped()
      self.all_poses = PoseArray()
      self.all_markers = MarkerArray()

      self.rate = rospy.Rate(1.0)                      # publish message at 1 Hz

      # initialize values for locating target on Kinect v2 image
      self.target_u = 0                        # u is pixels left(0) to right(+)
      self.target_v = 0                        # v is pixels top(0) to bottom(+)
      self.target_d = 0                        # d is distance camera(0) to target(+) from depth image
      self.last_d = 0                          # last non-zero depth measurement

      self.camera_model = image_geometry.PinholeCameraModel()

      self.p_camera=PoseStamped()
      self.image = None
      self.image_depth_ros = None
      self.register_depth_image = None
      self.track_img = None
      self.copy_register_depth_image = None

      self.sample_point_count = 1 #10
      self.effective_sample_point = 1  # 5
      self.offset = 0 #5
      self.dist_thresh = 0.1

      self.n = 0
      self.detected_pose_number=0
      self.objects = OrderedDict()

      # Convert image from a ROS image message to a CV image
      self.bridge = CvBridge()
      #self.image_depth_ros = None

      self.tf_listener = tf.TransformListener()  # create listener for transforms

      self.camera_info_sub = rospy.Subscriber('/thorvald_001/kinect2_front_camera/hd/camera_info',
                                              CameraInfo, self.camera_info_callback)

      # Wait for the camera_info topic to become available
      rospy.wait_for_message('/thorvald_001/kinect2_front_camera/hd/image_color_rect', Image)
      self.nav_Sub = rospy.Subscriber('/thorvald_001/nav_points', String, self.nav_callback)
      self.nav_points = ''


      begin_time = time.time()

      # (1)Subscribe to color and registered depth images
      #rospy.Subscriber('/simple_depth_register_node/depth_registered', Image, self.register_depth_callback,queue_size=1,buff_size=524288000)
      #rospy.Subscriber('/thorvald_001/kinect2_front_camera/hd/image_color_rect', Image, self.image_callback, queue_size=1,buff_size=524288000)

      # (2)subscribe to input topics

      self.depth_sub = message_filters.Subscriber('/simple_depth_register_node/depth_registered', Image)
      self.image_sub = message_filters.Subscriber('/thorvald_001/kinect2_front_camera/hd/image_color_rect',Image)
      #self.Bounding_Boxes = message_filters.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes)
      #self.tss = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub,self.Bounding_Boxes],
                                                             #queue_size=10,slop=0.1)
      self.tss = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub],queue_size=10, slop=0.1)
      self.tss.registerCallback(self.image_callback)

      #self.detection_image=None
      #rospy.Subscriber('darknet_ros/detection_image', Image, self.detection_callback,queue_size=10)
      self.boxes = None
      rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self.boundingBoxCallback,queue_size=10)


      end_time = time.time()
      run_time = end_time - begin_time
      print("run_time", run_time)
      print('\n')

      self.rate.sleep()                        # suspend until next cycle

   # This callback function handles processing Kinect color image, looking for the pink target.
   def randomPoints(self, xmin, xmax, ymin, ymax, count):
      result = []
      xoffset = abs(xmax - xmin)
      yoffset = abs(ymax - ymin)
      for i in range(count):
         x = xmin + int(xoffset * numpy.random.uniform(0, 1))
         y = ymin + int(yoffset * numpy.random.uniform(0, 1))
         result.append([x, y])
      return result

   def camera_info_callback(self, data):
      self.camera_model.fromCameraInfo(data)
      self.camera_info_sub.unregister()  # Only subscribe once


   def register_depth_callback(self, data):
      #print('type(data)',type(data.data))
      self.image_depth_ros = data.data

   def boundingBoxCallback(self, data):
      # boxes = data.boundingBoxes
      self.boxes = data.bounding_boxes

   def detection_callback(self, data):
      # boxes = data.boundingBoxes
      #print('data.data',type(data))
      self.detection_image = data.data

   def nav_callback(self, data):
      self.nav_points = data.data

   #def image_callback(self, data): #(1)
   def image_callback(self, color_data, depth_data):#(2)
      self.image_depth_ros = depth_data

      if self.camera_model is None:
         return

      # convert ROS image to OpenCV image
      self.image = self.bridge.imgmsg_to_cv2(color_data, desired_encoding='bgr8') #(2)
      self.track_img = self.image.copy()

      self.register_depth_image = self.bridge.imgmsg_to_cv2(self.image_depth_ros,"32FC1")
      self.copy_register_depth_image = self.register_depth_image.copy()

      #detectionimage = self.bridge.imgmsg_to_cv2(self.detection_image,desired_encoding="passthrough")
      #cv2.imshow("detection_image", detection_image)

      if (self.nav_points == 'detect'):
      #if (1 == 1): #for debug
         bounding_box_list = []
         # Check for at least one target found
         if self.boxes == None:
            print("No target found")
         else:
            #n = len(self.boxes)
            #print("len(self.boxes):",len(self.boxes))
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
               #cv2.rectangle(yolo_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

               D = self.register_depth_image[int(target_center_y), int(target_center_x)]  # when processing register_depth_image, it is transposed.
               #print('depth:',D)
               if not (numpy.isnan(D) or D == numpy.inf) and D >= 0.5 and D <= 8:  # the valid depth range of kinectv2: 0.5 ~ 8 M
                  bounding_box_list.append((target_center_x, target_center_y))
                  cv2.circle(self.image, (int(target_center_x), int(target_center_y)), 10, (0, 0, 255), -1)
                  cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
                  cv2.circle(self.copy_register_depth_image, (int(target_center_x), int(target_center_y)), 10, 255, -1)

            ##tracking the centroid of targets##
            old_objects = self.objects
            old_object_keys = old_objects.keys() #cannot delete this line
            # print('old_objects \n', old_objects)
            # print('old_objects.keys()_1', old_objects.keys())
            self.objects = ct.update(bounding_box_list)
            new_objects = self.objects
            #print('new_objects \n', new_objects)
            # print('old_objects.keys()_2', old_objects.keys())
            #print('old_object_keys', old_object_keys)
            #print('new_objects.keys()', new_objects.keys())
            uncommoned_keys = [i for i in new_objects.keys() if i not in old_object_keys]
            #print('difference_keys', uncommoned_keys)

            #print('new_objects.values()', new_objects.values())
            # print('new_objects.values()', new_objects.values()[1])

            new_points = []
            # loop over the tracked objects
            for (objectID, centroid) in new_objects.items():
               #print('objectID, centroid', objectID, centroid)
               if (objectID in old_object_keys):
                  continue
               else:
                  new_points.append(list(centroid))

            #print('new_points', new_points)
            #print('\n')

            # loop over the tracked objects
            for (objectID, centroid) in self.objects.items():
              # draw both the ID of the object and the centroid of the
              # object on the output frame
              text = "ID {}".format(objectID)
              #print('objectID:',objectID)
              cv2.putText(self.track_img, text, (centroid[0] - 10, centroid[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
              cv2.circle(self.track_img, (centroid[0], centroid[1]), 12, (0, 255, 0), -1)

            if len(new_points)>0:
               for i in range(len(new_points)):
                  centroid_x = new_points[i][0]
                  centroid_y = new_points[i][1]
                  #print('new_points[i]',new_points[i])
                  # Sample random points from inside box and calculate the coodinate of average points  #From https://github.com/verlab/3DSemanticMapping_JINT_2020/blob/master/auto/src/projector.py
                  im_pixels = self.randomPoints(centroid_x - self.offset, centroid_x + self.offset,
                                                centroid_y - self.offset, centroid_y + self.offset,
                                                self.sample_point_count)
                  #print("im_pixels",im_pixels)
                  points_3d = []
                  final_points = []
                  for u, v in im_pixels:
                     #print('u,v',u,v)

                     #(1)
                     D = self.register_depth_image[int(v), int(u)]#when processing register_depth_image, it is transposed.
                     #print('depth:',D)

                     if not (numpy.isnan(D) or D == numpy.inf) and D>=0.5 and D<=8: #the valid depth range of kinectv2: 0.5 ~ 8 M
                        camera_coord = self.PxielToCamera(u, v,D)
                        points_3d.append(camera_coord)
                        cv2.circle(self.image, (int(u), int(v)), 1, (0, 255, 0), -1)
                  #print('camera_points_3d:',points_3d)
                  #print('len(points_3d):', len(points_3d))

                  #print('len(points_3d)',len(points_3d))
                  if len(points_3d) < self.effective_sample_point:
                     #print ("No depth data available!")
                     continue

                  # Select points and transform them into map frame
                  #print('points_3d',points_3d)
                  for p in points_3d:
                     # Convert to point in [map] frame
                     map_point = self.CameraToWorldFrame(p)
                     # Save point in final points
                     final_points.append(map_point)
                     #print(final_points)

                  # Calculate average position in [map] frame
                  Xs = [p.pose.position.x for p in final_points]
                  Ys = [p.pose.position.y for p in final_points]
                  Zs = [p.pose.position.z for p in final_points]
                  avgX = sum(Xs) * 1.0 / len(Xs)
                  avgY = sum(Ys) * 1.0 / len(Ys)
                  avgZ = sum(Zs) * 1.0 / len(Zs)

                  #print('Avarage coordinate(avgX,avgY,avgZ) in the map frame:',avgX,avgY,avgZ)

                  detected_pose = PoseStamped()
                  detected_pose = self.update_target_pose(avgX,avgY,avgZ)
                  #self.detected_pose_number=self.detected_pose_number+1
                  #print('detected_pose_number',self.detected_pose_number)
                  #print('detected_pose.pose:',detected_pose.pose)
                  if self.n == 0: #first target point, we need to store directly
                     self.all_poses.poses.append(Pose(detected_pose.pose.position, detected_pose.pose.orientation))
                     self.marker(detected_pose, 0.15, 0.15, 0.15, 0, 1, 0, 1)  # grape_brunch_marker in the rviz
                     #print('first_self.all_poses.poses:',self.all_poses.poses)
                  else:
                     #check the new point_3d poses avarage distance between current frame and last frame, only distance>thresh, we
                     #confirm it is a new point, otherwise it is interference point (it is overlap with the point which was
                     # detected from last frame)
                     R_list=[]
                     #print('--------------start check-----------')
                     for pose in self.all_poses.poses:  #n=len(self.all_poses.poses)
                        #print("pose.position:", pose.position)
                        #print('detected_pose.pose.position', detected_pose.pose.position)
                        Rx = abs(pose.position.x - detected_pose.pose.position.x)
                        Ry = abs(pose.position.y - detected_pose.pose.position.y)
                        Rz = abs(pose.position.z - detected_pose.pose.position.z)
                        R = (Rx + Ry + Rz) / 3.0
                        #print('R',R)
                        R_list.append(R)

                     #print('len(R_list)',len(R_list))
                     num=0
                     for i in range(len(R_list)):
                        if R_list[i] > self.dist_thresh: #the center distance between two grape brunches
                           num=num+1
                     if num == len(R_list):
                        self.all_poses.poses.append(Pose(detected_pose.pose.position, detected_pose.pose.orientation))
                        self.marker(detected_pose, 0.15, 0.15, 0.15, 0, 1, 0, 1)  # grape_brunch_marker in the rviz
                        #print('self.all_poses.poses:', self.all_poses.poses)
                        #print('-----------add_new------------------ \n')
                     else:
                        print('points overlap')

                  #print('self.n',self.n)
                  self.n = self.n + 1
                  #print('self.all_poses',self.all_poses)

               #cv2.circle(self.image, (int(target_center_x), int(target_center_y)), 10, (0,0,255), -1)
               #cv2.circle(self.register_depth_image, (int(target_center_x), int(target_center_y)), 10, 255, -1)
               #cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print('Target_number:', len(self.all_poses.poses))
            print('\n------------current_frame------------------\n')
            #print('self.all_poses:',self.all_poses)
            self.update_all_target_pose(self.all_poses) #pub all targets poses in one frame of camera stream
            self.markers(self.all_markers)
            #self.all_poses = PoseArray() #if use it, it only store target information from current frame of camera
            #self.all_markers = MarkerArray() #if use it, it only store target information from current frame of camera

      if (self.nav_points == 'no_detect'):
         print('nav_point', self.nav_points)


      self.image = cv2.resize(self.image, (0, 0), fx=0.5, fy=0.5)
      self.track_img = cv2.resize(self.track_img, (0, 0), fx=0.5, fy=0.5)
      #self.register_depth_image *= 1.0 / 10.0  # scale for visualisation (max range 10.0 m)
      #self.register_depth_image = cv2.resize(self.register_depth_image, (0, 0), fx=0.5, fy=0.5)
      self.copy_register_depth_image *= 1.0 / 8.0  # scale for visualisation (max range 8.0 m)
      self.copy_register_depth_image = cv2.resize(self.copy_register_depth_image, (0, 0), fx=0.5, fy=0.5)
      cv2.imshow("Track", self.track_img)
      cv2.imshow("result", self.image)
      #cv2.imshow("register_depth_image", self.register_depth_image)
      cv2.imshow("register_depth_image", self.copy_register_depth_image)
         #cv2.imwrite('result.png',self.image)
      cv2.waitKey(3)

      self.image = None
      self.track_img = None
      self.register_depth_image=None
      self.copy_register_depth_image = None
      self.boxes = None




   def PxielToCamera(self, pxX, pxY,d):
      # Get a 3d point on the camera coordinate by projectPixelTo3dRay
      #  FROM _> https://answers.ros.org/question/241624/converting-pixel-location-in-camera-frame-to-world-frame/?answer=242060#post-id-242060 + Greg's lab on img geometry
      #cam_model_point = self.camera_model.projectPixelTo3dRay(self.camera_model.rectifyPoint((pxX,pxY)))  # project a rectified pixel to a 3d ray. Returns the unit vector in the camera coordinate frame in the direction of rectified pixel (u,v) in the image plane. This is the inverse of project3dToPixel().
      camera_coords = self.camera_model.projectPixelTo3dRay((pxX, pxY))
      camera_coords = [x / camera_coords[2] for x in camera_coords]  # adjust the resulting vector so that z = 1
      camera_coords = [x * d for x in camera_coords]  # multiply the vector by depth
      #print('Target in the camera_coords:', camera_coords)
      return camera_coords

   def CameraToWorldFrame(self,camera_3d_point):
      # define a point with position and orientation in camera coordinates
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

      # Next two steps will wait information and take massive time, so don't use them.
      #object_location.header.stamp = rospy.Time.now()
      #self.tf_listener.waitForTransform(self.camera_model.tfFrame(), '/map', rospy.Time.now(),rospy.Duration(1.0))  # block until a transform is possible or times out.

      self.p_camera = self.tf_listener.transformPose('/map', object_location)
      #print ("Target in the map:", self.p_camera.pose.position.x,self.p_camera.pose.position.y, self.p_camera.pose.position.z)
      return self.p_camera

   def marker(self,pose,sx,sy,sz,r,g,b,a):
      marker = Marker()
      marker.header.frame_id = '/map'
      #marker.header.stamp = rospy.Time.now()
      marker.id = 0  # enumerate subsequent markers here
      marker.action = Marker.ADD  # can be ADD, REMOVE, or MODIFY
      marker.ns = "grape_model"
      marker.type = 2 #http://docs.ros.org/en/api/visualization_msgs/html/msg/Marker.html
      marker.pose.position.x = pose.pose.position.x
      marker.pose.position.y = pose.pose.position.y
      marker.pose.position.z = pose.pose.position.z
      marker.pose.orientation.x = 0.0
      marker.pose.orientation.y = 0.0
      marker.pose.orientation.z = 0.0
      marker.pose.orientation.w = 1.0
      #print('marker',marker)
      marker.scale.x = sx  # artifact of sketchup export
      marker.scale.y = sy  # artifact of sketchup export
      marker.scale.z = sz  # artifact of sketchup export
      marker.color.r = r
      marker.color.g = g
      marker.color.b = b
      marker.color.a = a
      marker.lifetime = rospy.Duration()
      #marker.frame_locked = False
      self.pub_marker.publish(marker)
      #print('marker',marker)
      self.all_markers.markers.append(marker)

   def update_target_pose(self, x, y, z):

      pub_pose = PoseStamped()
      # set position values for target location
      pub_pose.pose.position.x = x      # pose in camera pixels u
      pub_pose.pose.position.y = y      # pose in camera pixels v
      pub_pose.pose.position.z = z      # pose in meters from camera

      # determine quaternion values from euler angles of (0,0,0)
      quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
      pub_pose.pose.orientation.x = quaternion[0]
      pub_pose.pose.orientation.y = quaternion[1]
      pub_pose.pose.orientation.z = quaternion[2]
      pub_pose.pose.orientation.w = quaternion[3]

      # complete header information
      #self.pub_pose.header.seq += 1
      #pub_pose.header.stamp = rospy.Time.now()
      pub_pose.header.frame_id = "map"

      # publish pose of target
      self.pub.publish(pub_pose)
      #self.marker(pub_pose, 0.3, 0.3, 0.3, 0, 1, 0, 1) #grape_brunch_marker in the rviz

      #pub_pose = PoseStamped() #***Very important step, othwewise every component in the list is same(no effect)***

      return pub_pose

   def markers(self,markerArray):
      #From http://docs.ros.org/en/fuerte/api/rviz/html/marker__array__test_8py_source.html
      id=0
      for m in markerArray.markers:
         m.id = id
         id += 1
      self.pub_all_markers.publish(markerArray)

   def update_all_target_pose(self,all_poses):
      # complete header information
      #self.all_poses.header.seq += 1
      #all_poses.header.stamp = rospy.Time.now()
      all_poses.header.frame_id = "map"
      # publish pose of target
      self.pub_all.publish(all_poses) # For iter PoseAray:n=len(all_poses.poses)
      #self.markers(self.all_markers)


if __name__ == '__main__':

   # start up the detector node and run until shutdown by interrupt
   try:
      detector = Detector()
      rospy.spin()

   except rospy.ROSInterruptException:
      rospy.loginfo("Detector node terminated.")

   # close all terminal windows when process is shut down
   cv2.destroyAllWindows()

