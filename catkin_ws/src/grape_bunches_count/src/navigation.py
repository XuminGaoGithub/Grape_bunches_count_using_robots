#! /usr/bin/env python
import rospy
import actionlib
from topological_navigation.msg import GotoNodeAction, GotoNodeGoal
from std_msgs.msg import String

class Navigation:
    '''
    Navigation:
    The robot can autonomous navigation around the grape trellis using the pre-established
    topology map and movebase package in ROS. Due to the kinect has a limited perceptual range
    and best perceptual range is 0.5 ~ 4.5 m. So there are two type of waypoints: 'Execute_count' and 'No_count'.
    when robot receives 'Execute_count' from navigation node,it executes count. Conversely,it doesn't execute count.
    '''

    def __init__(self):
        rospy.init_node('topological_navigation_client')  # initialize ROS node
        self.client = actionlib.SimpleActionClient('/thorvald_001/topological_navigation', GotoNodeAction) # create an action client
        self.client.wait_for_server() # waits until the action server has started up
        self.goal = GotoNodeGoal()
        self.pub = rospy.Publisher('/thorvald_001/nav_points', String, queue_size=1)  # initialize publisher for String

    def execute(self):
        # Define waypoints of topology navigation,
        # the specific pose of every waypoint can be find in ../maps/map.yaml.
        waypoints = ['StartPoint', 'DetectionPoint_0', 'DetectionPoint_1', 'DetectionPoint_2', 'MiddlePoint',
                     'TurningPoint', 'DetectionPoint_3', 'DetectionPoint_4', 'DetectionPoint_5','MiddlePoint','StartPoint']


        '''
        Set 'DetectionPoint_1','DetectionPoint_2','DetectionPoint_4','DetectionPoint_5' as the type of
        'Execute_count' waypoints, these points are close to grape trellis. So when robot moves 
        'DetectionPoint_0'-> 'DetectionPoint_1' -> 'DetectionPoint_2' and 
        'DetectionPoint_3'-> 'DetectionPoint_4' -> 'DetectionPoint_5', it executes count task.
        Conversely, it doesn't execute count task.
        '''
        for waypoint in waypoints:
            self.goal.target = waypoint
            if (self.goal.target=='DetectionPoint_1' or self.goal.target=='DetectionPoint_2'
                    or self.goal.target=='DetectionPoint_4' or self.goal.target=='DetectionPoint_5'):
                print('Execute_count')
                self.pub.publish('Execute_count')
            else:
                print('No_count')
                self.pub.publish('No_count')

            self.client.send_goal(self.goal)  # send the goal to the action server
            status = self.client.wait_for_result() # wait for the server to finish the action.
            result = self.client.get_result() # get the result

if __name__ == '__main__':
    print('-----Navigation starting-----')
    Navigation().execute() # execute the navigation according to the waypoins
    print('-----Navigation exiting-----')
