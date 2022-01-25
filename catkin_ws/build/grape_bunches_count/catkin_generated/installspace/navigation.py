#!/usr/bin/env python2
import rospy
import actionlib
from topological_navigation.msg import GotoNodeAction, GotoNodeGoal
from std_msgs.msg import String
from colorama import Fore, Back, Style, init # this allows for color print to console
init(autoreset=True) # dont save color to memory


class Navigation:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('/thorvald_001/topological_navigation', GotoNodeAction) # subscribe to movement master
        self.client.wait_for_server() 
        self.goal = GotoNodeGoal()
        self.pub = rospy.Publisher('/thorvald_001/nav_points', String, queue_size=1)

    def execute(self):
        #define nav_points
        waypoints = ['StartPoint', 'DetectionPoint_0', 'DetectionPoint_1', 'DetectionPoint_2', 'MiddlePoint',
            'DetectionPoint_3', 'DetectionPoint_4', 'DetectionPoint_5','MiddlePoint','StartPoint']

        # loop through waypoints
        for waypoint in waypoints:
            self.goal.target = waypoint
            if (self.goal.target=='DetectionPoint_1' or self.goal.target=='DetectionPoint_2'
                    or self.goal.target=='DetectionPoint_4' or self.goal.target=='DetectionPoint_5'):
                print('detect')
                self.pub.publish('detect')
            else:
                print('no detect')
                self.pub.publish('no_detect')

            self.client.send_goal(self.goal)
            status = self.client.wait_for_result()
            result = self.client.get_result()
            
            ''' '''


if __name__ == '__main__':
    rospy.init_node('topological_navigation_client')
    print('-----Navigation started-----')
    Navigation().execute()
    print('-----Navigation exiting-----')
