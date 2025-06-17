#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
import math
import time
from geometry_msgs.msg import Twist
import numpy as np
from nav_msgs.msg import Odometry

def calc_distance(p_1, p_2):
    return math.sqrt((p_2[0]-p_1[0])**2 + (p_2[1]-p_1[1])**2)

def calculate_angle(p_1, p_2, p_3):        
    '''
    p_1 is the point it is coming from 

    p_2 is the point it is currently at

    p_3 is the point it is going to
    '''
    
    angle_1 = math.atan((p_2[1]-p_1[1])/(p_2[0]-p_1[0]))
    angle_2 = math.atan((p_3[1]-p_2[1])/(p_3[0]-p_2[0]))

    # return (angle_1 - angle_2)*180/math.pi
    return angle_2 - angle_1

def quat_2_euler(ox, oy, oz, ow):
    t3 = +2.0 * (ow * oz + ox * oy)
    t4 = +1.0 - 2.0 * (oy * oy + oz * oz)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def get_sign(current, goal, dir):
    
    vect1 = np.array([goal[0] - current[0], goal[1] - current[1]])/ np.linalg.norm([goal[0] - current[0], goal[1] - current[1]])
    vect2 = np.array(dir)/np.linalg.norm(dir)

    if round(np.dot(vect1, vect2), 2) < 0:
            return -1
    else:
        return 1


    


class controller():

    def __init__(self, path):
        self.prev_error = 0
        self.error_integral = 0
	self.error_lin_integral = 0
	self.error_lin_prev = 0
        
        self.path = path
        self.max_speed = 1.0
        self.translating = False
        self.rotating = False
        self.num_of_points = len(path)

        self.vec = 0

        self.prev_orient = 0
        self.current_pos = (0,0)
        self.current_orient = 0
        self.trans_speed = 0.1
        self.rot_speed = 1
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.get_pos)

        self.kp_lin = 0.5
        self.ki_lin = 0
        self.kd_lin = 0.1

        self.kp_ang = 0.3
        self.ki_ang = 0
        self.kd_ang = 0.05

        self.error = 0
        self.error_last = 0
        self.integral_error = 0
        self.angle_integral = 0
        self.derivative_error = 0
        self.output = 0

    def get_pos(self, data):
        self.current_pos = (data.pose.pose.position.x, data.pose.pose.position.y)
        ox, oy, oz, ow = data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w
        angle_rad = quat_2_euler(ox, oy, oz, ow)
        self.current_orient = self.normalize_angle(angle_rad)
        # self.current_orient = quat_2_euler(ox, oy, oz, ow)*180/math.pi

    def mover(self):
        print('Number of nodes in path: ', len(self.path))
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        move_cmd = Twist()
        rate = rospy.Rate(50)  # 控制频率
        lookahead_distance = 0.8  # 前视距离，可调参数
        
        goal_reached = False
        current_idx = 0
        
        self.prev_error = 0
        self.error_integral = 0
        dt = 1.0 / 50  # 控制周期 50Hz
        
        while not rospy.is_shutdown() and not goal_reached:
            robot_x, robot_y = self.current_pos
            robot_yaw = self.current_orient
            
            lookahead_point = None
            for i in range(current_idx, len(self.path)):
                px, py = self.path[i]
                dist = math.hypot(px - robot_x, py - robot_y)
                if dist >= lookahead_distance:
                    lookahead_point = (px, py)
                    current_idx = i
                    break
                
            if lookahead_point is None:
                final_point = self.path[-1]
                dist_to_goal = calc_distance(self.current_pos, final_point)
                if dist_to_goal < 0.1:
                    goal_reached = True
                    break
                else:
                    lookahead_point = final_point
                    
            # 坐标系转换
            dx = lookahead_point[0] - robot_x
            dy = lookahead_point[1] - robot_y
            x_r = math.cos(-robot_yaw) * dx - math.sin(-robot_yaw) * dy
            y_r = math.sin(-robot_yaw) * dx + math.cos(-robot_yaw) * dy
            
            # 计算横向误差并进行 PID 控制
            error = y_r
            self.error_integral += error * dt
            error_derivative = (error - self.prev_error) / dt
            self.prev_error = error
            
            angular_speed = self.kp_ang * error + self.ki_ang * self.error_integral + self.kd_ang * error_derivative
            angular_speed = max(min(angular_speed, 2.0), -2.0)
            
            
            dist_error = math.hypot(x_r, y_r)  # 与前视点距离，也可用路径剩余距离
            self.error_lin_integral += dist_error * 0.02
            error_lin_derivative = (dist_error - self.error_lin_prev) / 0.02
            self.error_lin_prev = dist_error
            
            raw_linear_speed = self.kp_lin * dist_error + self.ki_lin * self.error_lin_integral + self.kd_lin * error_lin_derivative
            
            # 非线性函数平滑，例如 tanh 或 sigmoid，可选
            raw_linear_speed *= math.tanh(dist_error)  # 可加权
            
            # 限制最大速度
            linear_speed = max(min(raw_linear_speed, self.max_speed), 0.0)

            move_cmd.linear.x = linear_speed
            move_cmd.angular.z = angular_speed
            pub.publish(move_cmd)
            
            rospy.loginfo(
                "[PID] Pos: ({:.2f}, {:.2f}) | Yaw: {:.2f}° | Lookahead: ({:.2f}, {:.2f}) | Error: {:.3f} | Angular: {:.3f}".format(
                    robot_x, robot_y, math.degrees(robot_yaw),
                    lookahead_point[0], lookahead_point[1], error, angular_speed
                )
            )
            
            rate.sleep()
            
        # 停止机器人
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.0
        pub.publish(move_cmd)
        print("All points reached!")
    def normalize_angle(self, angle):
	"""将角度标准化到 [-π, π] 区间"""
	while angle > math.pi:
    	    angle -= 2.0 * math.pi
	while angle < -math.pi:
    	    angle += 2.0 * math.pi
	return angle
        
    


def callback(data):
    path_nodes = []
    published_data = data.data
    
    for i in range(len(published_data)):
        if i%2 == 0:
            path_nodes.append((round(published_data[i], 3), round(published_data[i+1], 3)))
        else:
            continue
    print(path_nodes)
    control = controller(path_nodes)
    control.mover()

def listener():
    rospy.init_node('controller', anonymous=True)
    data = rospy.wait_for_message('path', Float32MultiArray, timeout=None)
    callback(data)

    rospy.spin()
    
if __name__ == '__main__':
    listener()
    
    
    
    
