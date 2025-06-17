#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import math
import random
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

def calc_distance(p_1, p_2):
    return math.sqrt((p_2[0]-p_1[0])**2 + (p_2[1]-p_1[1])**2)

def point_to_line(p_1, p_2, p_3):
    dist = math.sqrt((p_2[0] - p_1[0])**2 + (p_2[1] - p_1[1])**2)

    # determine intersection ratio u
    # for three points A, B with a line between them and a third point C, the tangent to the line AB
    # passing through C intersects the line AB a distance along its length equal to u*|AB|
    r_u = ((p_3[0] - p_1[0])*(p_2[0] - p_1[0]) + (p_3[1] - p_1[1])*(p_2[1] - p_1[1]))/(dist**2)

    # intersection point
    p_i = (p_1[0] + r_u*(p_2[0] - p_1[0]), p_1[1] + r_u*(p_2[1] - p_1[1]))

    # distance from P3 to intersection point
    tan_len = calc_distance(p_i, p_3)

    return r_u, tan_len

class MapProcessor:
    def __init__(self):
        self.map_data = None
        self.map_info = None
        self.obstacles = []
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.extract_obstacles()
        rospy.loginfo("Map loaded with resolution %.3f at origin (%.2f, %.2f)" % (
            self.map_info.resolution, 
            self.map_info.origin.position.x,
            self.map_info.origin.position.y))

    def extract_obstacles(self):
        self.obstacles = []
        for y in range(self.map_info.height):
            for x in range(self.map_info.width):
                if self.map_data[y][x] > 50:  # 障碍物阈值
                    world_x = x * self.map_info.resolution + self.map_info.origin.position.x
                    world_y = y * self.map_info.resolution + self.map_info.origin.position.y
                    self.obstacles.append((world_x, world_y))

    def world_to_map(self, point):
        x = int((point.x - self.map_info.origin.position.x) / self.map_info.resolution)
        y = int((point.y - self.map_info.origin.position.y) / self.map_info.resolution)
        return (x, y)

    def is_collision(self, p1, p2, safe_radius=0.25):  # 添加安全距离参数
        if self.map_data is None or self.map_info is None:
            return True  # 没有地图信息，默认认为碰撞
        
        x1, y1 = self.world_to_map(p1)
        x2, y2 = self.world_to_map(p2)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        
        # 安全半径对应的格子范围
        safe_range = int(safe_radius / self.map_info.resolution)
        
        def is_near_obstacle(cx, cy):
            for dx in range(-safe_range, safe_range + 1):
                for dy in range(-safe_range, safe_range + 1):
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                        if self.map_data[ny][nx] > 50:
                            return True
            return False
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if not (0 <= x < self.map_info.width and 0 <= y < self.map_info.height):
                    return True
                if is_near_obstacle(x, y):
                    return True
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if not (0 <= x < self.map_info.width and 0 <= y < self.map_info.height):
                    return True
                if is_near_obstacle(x, y):
                    return True
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        return False  # 路径段无碰撞
    
    

class RRTStarPlanner:
    def __init__(self):
        self.mp = MapProcessor()
        self.start = None
        self.goal = None
        self.nodes = []
        
        # 双路径发布器
        self.vis_path_pub = rospy.Publisher('/rrt_path', Path, queue_size=10)  # 可视化用
        self.ctrl_path_pub = rospy.Publisher('/path', Float32MultiArray, queue_size=10)  # 控制用
        self.tree_pub = rospy.Publisher('/rrt_tree', MarkerArray, queue_size=10)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        rospy.wait_for_message('/map', OccupancyGrid)
        rospy.loginfo("Planner initialized, waiting for goal...")

    class Node:
        def __init__(self, point, parent=None):
            self.point = point
            self.parent = parent
            self.cost = 0.0 if parent is None else parent.cost + math.hypot(
                point.x - parent.point.x, point.y - parent.point.y)

    def goal_callback(self, msg):
        self.goal = msg.pose.position
        if self.mp.map_info is not None:
            self.start = Point(0, 0, 0)  # 强制起点为(0,0)以匹配控制器
            self.plan_path()

    def calc_cost(self, node):
        cost = 0.0
        while node.parent is not None:
            cost += calc_distance(
                (node.point.x, node.point.y),
                (node.parent.point.x, node.parent.point.y))
            node = node.parent
        return cost
    
    def plan_path(self):
        self.nodes = [self.Node(self.start)]  # 初始化树
        max_iter = 5000
        step_size = 0.2
        goal_threshold = 0.5
        neighbor_radius = 0.5  # RRT*的邻居搜索半径
        
        for i in range(max_iter):
            # Goal bias 引导采样
            goal_bias = 0.1
            if random.random() < goal_bias:
                rand_point = self.goal
            else:
                rand_x = random.uniform(
                    self.mp.map_info.origin.position.x,
                    self.mp.map_info.origin.position.x + self.mp.map_info.width * self.mp.map_info.resolution)
                rand_y = random.uniform(
                    self.mp.map_info.origin.position.y,
                    self.mp.map_info.origin.position.y + self.mp.map_info.height * self.mp.map_info.resolution)
                rand_point = Point(rand_x, rand_y, 0)
                
            # 找到最近的节点
            nearest_node = min(self.nodes, key=lambda node: calc_distance(
                (node.point.x, node.point.y), (rand_point.x, rand_point.y)))
            
            # 扩展新点
            theta = math.atan2(rand_point.y - nearest_node.point.y,
                                rand_point.x - nearest_node.point.x)
            new_x = nearest_node.point.x + step_size * math.cos(theta)
            new_y = nearest_node.point.y + step_size * math.sin(theta)
            new_point = Point(new_x, new_y, 0)
            
            # 碰撞检测
            if self.mp.is_collision(nearest_node.point, new_point):
                continue
            
            # 搜索所有邻居（在邻域范围内）
            neighbors = [node for node in self.nodes if calc_distance(
                (node.point.x, node.point.y), (new_point.x, new_point.y)) < neighbor_radius]
            
            # 初始化：父节点 = nearest，最小cost = 它的路径代价 + 距离
            min_cost = self.calc_cost(nearest_node) + calc_distance(
                (nearest_node.point.x, nearest_node.point.y), (new_point.x, new_point.y))
            best_parent = nearest_node
            
            # 在所有邻居中找 cost 最小的可连接父节点
            for neighbor in neighbors:
                if not self.mp.is_collision(neighbor.point, new_point):
                    cost = self.calc_cost(neighbor) + calc_distance(
                        (neighbor.point.x, neighbor.point.y), (new_point.x, new_point.y))
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = neighbor
                        
            # 加入新节点
            new_node = self.Node(new_point, best_parent)
            self.nodes.append(new_node)
            
            # RRT* rewiring：看新节点是否能优化邻居的路径
            for neighbor in neighbors:
                if neighbor == best_parent:
                    continue
                if not self.mp.is_collision(new_node.point, neighbor.point):
                    new_cost = self.calc_cost(new_node) + calc_distance(
                        (new_node.point.x, new_node.point.y), (neighbor.point.x, neighbor.point.y))
                    if new_cost < self.calc_cost(neighbor):
                        neighbor.parent = new_node  # 重接 parent
                        
            # 到达目标检测
            if calc_distance((new_point.x, new_point.y), (self.goal.x, self.goal.y)) < goal_threshold:
                goal_node = self.Node(self.goal, new_node)
                self.nodes.append(goal_node)
                self.publish_path(goal_node)
                return
            
        rospy.logwarn("RRT* planning failed: could not find path to goal.")
                

    def publish_path(self, goal_node):
        # 可视化Path构建
        vis_path = Path()
        vis_path.header.frame_id = "map"
        vis_path.header.stamp = rospy.Time.now()
        
        # 控制用路径数据构建
        ctrl_path = Float32MultiArray()
        path_points = []
        
        # 收集路径点（从终点到起点）
        current = goal_node
        while current is not None:
            # 可视化Path的点
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position = current.point
            vis_path.poses.append(pose)
            
            # 控制用路径点
            path_points.append((current.point.x, current.point.y))
            current = current.parent
        
        # 反转路径顺序（起点到终点）
        vis_path.poses.reverse()
        path_points.reverse()
        
        # 填充控制路径数据
        for point in path_points:
            ctrl_path.data.extend([point[0], point[1]])
        
        # 同时发布两种格式
        self.vis_path_pub.publish(vis_path)
        self.ctrl_path_pub.publish(ctrl_path)
	rospy.loginfo("Published both path formats:\n" +
		"- Visual Path ({} poses)\n".format(len(vis_path.poses)) +
		"- Control Path ({} points)".format(len(path_points)))
        
        self.publish_tree()

    def publish_tree(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rrt_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        
        for node in self.nodes:
            if node.parent:
                marker.points.append(node.parent.point)
                marker.points.append(node.point)
        
        marker_array.markers.append(marker)
        self.tree_pub.publish(marker_array)

if __name__ == '__main__':
    rospy.init_node('rrt_star_planner')
    planner = RRTStarPlanner()
    rospy.spin()
    
    