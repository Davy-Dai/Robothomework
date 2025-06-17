#!/usr/bin/env python2
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped

class TFBroadcaster:
    def __init__(self):
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.last_time = rospy.Time(0)
        self.rate = rospy.Rate(30) 
    def callback(self, msg):
        try:
            model_name = rospy.get_param('~model_name', 'turtlebot3_waffle')
            idx = msg.name.index(model_name)
            current_time = rospy.Time.now()
            
            # 防止重复时间戳
            if current_time == self.last_time:
                current_time += rospy.Duration(0.001)
            
            tf_msg = TransformStamped()
            tf_msg.header.stamp = current_time
            tf_msg.header.frame_id = "map"
            tf_msg.child_frame_id = "base_footprint"
            tf_msg.transform.translation = msg.pose[idx].position
            tf_msg.transform.rotation = msg.pose[idx].orientation
            
            self.broadcaster.sendTransform(tf_msg)
            self.last_time = current_time
            
        except ValueError as e:
            rospy.logerr_throttle(1.0, f"Model {model_name} not found: {str(e)}")

if __name__ == '__main__':
    rospy.init_node('gazebo_to_tf')
    node = TFBroadcaster()
    rospy.Subscriber("/gazebo/model_states", ModelStates, node.callback)
    rospy.spin()