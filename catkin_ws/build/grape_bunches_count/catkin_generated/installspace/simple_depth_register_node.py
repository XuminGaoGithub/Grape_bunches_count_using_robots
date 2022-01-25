#!/usr/bin/env python2

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from collections import namedtuple


class DepthRegisterer(object):
    '''class holding the camera setup and performing the actual depth registration'''

    Intrinsics = namedtuple('Intrinsics', ['fx', 'fy', 'cx', 'cy'])

    def __init__(self, x_offset=0, y_offset=0, z_offset=0, depth_scale=1.0):
        self.extrinsics = x_offset, y_offset, z_offset
        self.intrinsics = {}
        self.depth_scale = depth_scale
        self.pixel_grid = None


    def set_intrinsics(self, cam_id, fx, fy, cx, cy):
        self.intrinsics[cam_id] = DepthRegisterer.Intrinsics(fx, fy, cx, cy)
        print('fx, fy, cx, cy',fx, fy, cx, cy)

    def has_intrinsics(self, cam_id):
        return cam_id in self.intrinsics

    def register(self, rgb_image, depth_image):
        '''this is where the magic happens'''
        # generate the huge coordinate matrix only once
        if self.pixel_grid is None:
            # this is basically a 2d `range`
            self.pixel_grid = np.stack((
                np.array([np.arange(depth_image.shape[0]) for _ in xrange(depth_image.shape[1])]).T,
                np.array([np.arange(depth_image.shape[1]) for _ in xrange(depth_image.shape[0])])
                ), axis=2)

        fx_rgb, fy_rgb, cx_rgb, cy_rgb = self.intrinsics['rgb']
        fx_d, fy_d, cx_d, cy_d = self.intrinsics['depth']
        x_offset, y_offset, z_offset = self.extrinsics


        # compute the exact usable (mapped to) size of the registered depth image wrt. the FOVs of the cameras to avoid
        # gaps in the registered depth image (columns/rows without values). later, the registered depth image gets
        # scaled to match the size of the rgb image.
        h = int(depth_image.shape[0] * (rgb_image.shape[0] / fy_rgb) / (depth_image.shape[0] / fy_d))
        w = int(depth_image.shape[1] * (rgb_image.shape[1] / fx_rgb) / (depth_image.shape[1] / fx_d))
        registered_depth_image = np.zeros((h, w), dtype='float32')
        #print('registered_depth_image',registered_depth_image)

        # only consider pixels where actual depth values exist
        valid_depths = depth_image > 0
        valid_pixels = self.pixel_grid[valid_depths]
        #print('depth',depth_image)
        #print("valid_depths", valid_depths)
        #print("valid_pixels", valid_pixels)

        # might seem a little nasty, but computes the registered depth numpy-efficiently
        # apply scaling and extrinsics to depth values
        zs = depth_image[valid_depths] / self.depth_scale + z_offset
        # apply depth cam intrinsics, extrinsics, rgb cam intrinsics, scale down to (w, h)
        ys = (((((valid_pixels[:, 0] - cy_d) * zs) / fy_d + y_offset) * fy_rgb / zs + cy_rgb) / rgb_image.shape[0] * h).astype('int')
        xs = (((((valid_pixels[:, 1] - cx_d) * zs) / fx_d + x_offset) * fx_rgb / zs + cx_rgb) / rgb_image.shape[1] * w).astype('int')

        #np.set_printoptions(threshold='nan')
        #print("zs",zs,depth_image[valid_depths])

        # discard depth values unseen by rgb camera
        valid_positions = np.logical_and(np.logical_and(np.logical_and(ys >= 0, ys < registered_depth_image.shape[0]), xs >= 0), xs < registered_depth_image.shape[1])
        registered_depth_image[ys[valid_positions], xs[valid_positions]] = zs[valid_positions]

        # scale up without smoothing to match rgb image
        registered_depth_image = cv2.resize(registered_depth_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        #print("registered_depth_image.shape",registered_depth_image.shape)
        #cv2.imshow("registered_depth_image", registered_depth_image)

       ############Debug:show the depth image which is projected to the color image
        '''
        pro_img = rgb_image.copy()
        for h in range(0, pro_img.shape[0]):
            for j in range(0, pro_img.shape[1]):
                for c in range(pro_img.shape[2]):
                    if registered_depth_image[h][j]==0:
                        pro_img[h, j, c] = 0
                    #pass
                    #print("x,y,d",j,h,registered_depth_image[h][j])


        copy_src = rgb_image.copy()
        copy_src = cv2.resize(copy_src, (512, 422), interpolation=cv2.INTER_NEAREST)
        pro_img = cv2.resize(pro_img, (512, 424), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('src.png', copy_src)
        cv2.imwrite('pro_img.png', pro_img)

        cv2.imshow("rgb_image", rgb_image)
        cv2.imshow("pro_img", pro_img)
        cv2.imshow ("registered_depth_image", registered_depth_image)
        cv2.waitKey(1)
        '''
        #############


        return registered_depth_image



class DepthRegisterNode(object):
    '''class holding and performing ROS-related stuff like subscriptions, publishing, parameters and callbacks'''

    def __init__(self):
        rospy.init_node('simple_depth_register_node') #the programm is called by detect_location.py,so rospy.init_node should be in there
        self.cv_bridge = CvBridge()

        # extrinsics parameters (offset rgb cam -> depth cam), defaulting to realsense r200 values
        #dx = rospy.get_param('~x_offset', -0.0589333333)
        dx = rospy.get_param('~x_offset', 0)
        dy = rospy.get_param('~y_offset', 0)
        dz = rospy.get_param('~z_offset', 0)


        # scale in which depth values are presented relative to meters
        #depth_scale = rospy.get_param('~depth_scale', 1000.0) # 1000 is millimeters
        depth_scale = rospy.get_param('~depth_scale', 1)  # As usual, it shoule be 1000 in order to convert depth value to meter, but for this task, the depth value have already been convert to meter.

        self.dr = DepthRegisterer(dx, dy, dz, depth_scale)

        # input topics
        rgb_topic = rospy.get_param('~rgb_topic', '/thorvald_001/kinect2_front_camera/hd/image_color_rect')
        depth_topic = rospy.get_param('~depth_topic', '/thorvald_001/kinect2_front_sensor/sd/image_depth_rect')


        # read cameras' intrinsics
        #rgb_info_topic, depth_info_topic = [t[:t.rfind('/')] + '/camera_info' for t in (rgb_topic, depth_topic)]
        ##myself
        rgb_info_topic = '/thorvald_001/kinect2_front_camera/hd/camera_info'
        depth_info_topic = '/thorvald_001/kinect2_front_sensor/sd/camera_info'

        self.camera_info_callback(rospy.wait_for_message(rgb_info_topic, CameraInfo), 'rgb')
        self.camera_info_callback(rospy.wait_for_message(depth_info_topic, CameraInfo), 'depth')
        rospy.loginfo('camera calibration data OK')

        # subscribe to input topics
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.sub = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1) #it id very important for moving robot to keep time consistence
        self.sub.registerCallback(self.image_pair_callback)
        rospy.loginfo('synchronized subscriber OK')

        # registered depth image output topic name
        registered_topic = rospy.get_param('~registered_topic', '~depth_registered')

        # announce output topics
        self.pub = rospy.Publisher(registered_topic, Image, queue_size=1)

        self.rgb_image = None
        self.depth_image = None
        self.registered_depth_image = None


    def camera_info_callback(self, msg_camera_info, cam_id):
        '''passes the intrinsics of a camera_info message to the depth registerer'''
        fx, _, cx, _, fy, cy, _, _, _ = msg_camera_info.K
        self.dr.set_intrinsics(cam_id, fx, fy, cx, cy)

    def camera_info_callback(self, msg_camera_info, cam_id):
        '''passes the intrinsics of a camera_info message to the depth registerer'''
        fx, _, cx, _, fy, cy, _, _, _ = msg_camera_info.K
        self.dr.set_intrinsics(cam_id, fx, fy, cx, cy)


    def image_pair_callback(self, msg_rgb_image, msg_depth_image):

        '''makes the depth registerer process the image pair and produces output images'''
        # convert images ROS -> OpenCV
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg_rgb_image, "rgb8")
        except CvBridgeError as e:
            rospy.logwarn('error converting rgb image: %s' % e)
            return
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg_depth_image,"32FC1")
        except CvBridgeError as e:
            rospy.logwarn('error converting depth image: %s' % e)
            return

        # be lazy, check subscriber count first
        has_depth_subs = self.pub.get_num_connections() > 0
        has_subs = has_depth_subs
        #print('self.pub.get_num_connections()',self.pub.get_num_connections())
        #print('has_info_subs', has_info_subs)


        #if has_subs:
        if self.pub:
            # processing
            self.registered_depth_image = self.dr.register(self.rgb_image, self.depth_image)
                # convert image OpenCV -> ROS and send out
            msg = self.cv_bridge.cv2_to_imgmsg(self.registered_depth_image)
            msg.header.stamp = msg_depth_image.header.stamp
            self.pub.publish(msg)

        #registered_depth_image = self.registered_depth_image



        return self.registered_depth_image



    """
    def get_info_images(self):
        '''generates one registered and one unregistered blended image of rgb and depth image'''
        # clip depth values to a certain maximum and scale constantly wrt. this value to avoid jitter
        clipping_distance = 5. # meters
        # clip
        depth_image = self.depth_image / (clipping_distance * self.dr.depth_scale) * 255.
        depth_image[depth_image > 255] = 255
        # resize to fit rgb image and convert to 24-bit bgr
        depth_image = cv2.cvtColor(cv2.resize(depth_image.astype('uint8'), (self.rgb_image.shape[1], self.rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

        # clip
        registered_depth_image = self.registered_depth_image.astype(float) * 255. / clipping_distance
        registered_depth_image[registered_depth_image > 255] = 255
        # convert to 24-bit bgr
        registered_depth_image = cv2.cvtColor(registered_depth_image.astype('uint8'), cv2.COLOR_GRAY2BGR)

        # blend
        return cv2.addWeighted(self.rgb_image, 0.5, depth_image, 0.5, 0), cv2.addWeighted(self.rgb_image, 0.5, registered_depth_image, 0.5, 0)
    """
    
    def spin(self):
        '''stayin' alive'''
        rospy.spin()


if __name__ == '__main__':
    node = DepthRegisterNode()
    node.spin()
