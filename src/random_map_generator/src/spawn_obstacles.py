#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import random
import numpy as np
from PIL import Image
import yaml
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from geometry_msgs.msg import Pose, Point, Quaternion

# ================== 配置参数 ==================
MAP_WIDTH = 50.0    # 地图宽度（米）
MAP_HEIGHT = 50.0   # 地图高度（米）
RESOLUTION = 0.1     # 地图分辨率（米/像素）
OCCUPANCY_THRESHOLD = 0.65  # 占据阈值
ppgm=rospy.get_param('pgm_path')
pyaml=rospy.get_param('yaml_path')
# ================== 障碍物生成模块 ==================
class ObstacleGenerator:
    def __init__(self):
        self.obstacles = []  # 存储障碍物信息 (x, y, size_x, size_y)
        
    def generate_in_gazebo(self, num_obstacles=10):
        """在 Gazebo 中生成障碍物并记录参数"""
        rospy.init_node('spawn_random_obstacles')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        for i in range(num_obstacles):
            # 随机生成障碍物参数
            x = random.uniform(-MAP_WIDTH/2, MAP_WIDTH/2)
            y = random.uniform(-MAP_HEIGHT/2, MAP_HEIGHT/2)
            size_x = random.uniform(0.5, 3.0)
            size_y = random.uniform(0.5, 3.0)
            size_z = random.uniform(0.5, 2.0)
            half_x = size_x / 2
            half_y = size_y / 2
            x_min_check=x - half_x
            y_min_check=y - half_y
            x_max_check=x + half_x
            y_max_check=y + half_y
            near_threshold = 3

            # 检查障碍物是否包含原点或在其附近
            contains_origin = (x_min_check <= 0 <= x_max_check) and (y_min_check <= 0 <= y_max_check)
            near_origin = (abs(x) < half_x + near_threshold) and (abs(y) < half_y + near_threshold)
            if((contains_origin) or (near_origin)):
              continue
            self.obstacles.append((x, y, size_x, size_y))

            # 生成 Gazebo 模型
            model_name = "obstacle_{}".format(i)
            model_xml = self._create_sdf_model(size_x, size_y, size_z)
            pose = Pose(position=Point(x, y, size_z/2), orientation=Quaternion(0,0,0,1))
            
            try:
                req = SpawnModelRequest()
                req.model_name = model_name
                req.model_xml = model_xml
                req.initial_pose = pose
                req.reference_frame = "world"
                spawn_model(req)
                # rospy.loginfo(f"Spawned {model_name} at ({x:.2f}, {y:.2f})")
            except rospy.ServiceException as e:
                cgh=1
                # rospy.logerr(f"Failed to spawn model: {e}")

    def _create_sdf_model(self, size_x, size_y, size_z):
        """生成障碍物 SDF 模型"""
        r=random.random()
        g=random.random()
        b=random.random()
        return """
        <sdf version="1.6">
          <model>
            <static>true</static>
            <link name="link">
              <collision name="collision">
                <geometry>
                  <box>
                    <size>{:.3f} {:.3f} {:.3f}</size>
                  </box>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <box>
                    <size>{:.3f} {:.3f} {:.3f}</size>
                  </box>
                </geometry>
                <material>
                  <ambient>{:.3f} {:.3f} {:.3f} 1</ambient>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """.format(size_x,size_y,size_z,size_x,size_y,size_z,r,g,b)

# ================== PGM 地图生成模块 ==================
class PGMMapGenerator:
    def __init__(self):
        self.grid = np.ones((
            int(MAP_HEIGHT / RESOLUTION), 
            int(MAP_WIDTH / RESOLUTION)
        ), dtype=np.uint8) * 255

    def world_to_grid(self, x, y):
        """世界坐标转像素坐标"""
        grid_x = int((x + MAP_WIDTH/2) / RESOLUTION)
        grid_y = int((y + MAP_HEIGHT/2) / RESOLUTION)
        return grid_x, grid_y

    def add_obstacle(self, x, y, size_x, size_y):
        
        """添加矩形障碍物到地图"""
        # 计算障碍物覆盖的像素范围
        half_x = size_x / 2
        half_y = size_y / 2
        x_min_check=x - half_x
        y_min_check=y - half_y
        x_max_check=x + half_x
        y_max_check=y + half_y
        x_min, y_min = self.world_to_grid(x - half_x, y - half_y)
        x_max, y_max = self.world_to_grid(x + half_x, y + half_y)
        near_threshold = 3
  
        # 检查障碍物是否包含原点或在其附近
        contains_origin = (x_min_check <= 0 <= x_max_check) and (y_min_check <= 0 <= y_max_check)
        near_origin = (abs(x) < half_x + near_threshold) and (abs(y) < half_y + near_threshold)
        if((contains_origin) or (near_origin)):
          return

       
        
        # 确保索引不越界
        x_min = max(0, x_min)
        y_min_0 = max(0, y_min)
        x_max = min(self.grid.shape[1], x_max)
        y_max_0 = min(self.grid.shape[0], y_max)
        y_min=-y_min_0
        y_max=-y_max_0
        if(y_min>y_max):
            y_mid=y_max
            y_max=y_min
            y_min=y_mid
        self.grid[y_min:y_max, x_min:x_max] = 0  # 黑色代表障碍物

    def save(self, pgm_path=ppgm, yaml_path=pyaml):
  
        """保存PGM和YAML文件"""
        ros_grid = np.where(self.grid == 0, 0, 100).astype(np.uint8)
        Image.fromarray(ros_grid).save(pgm_path)
        
        # 生成YAML元数据
        yaml_data = {
            "image": pgm_path,
            "resolution": RESOLUTION,
            "origin": [-MAP_WIDTH/2, -MAP_HEIGHT/2, 0.0],
            "occupied_thresh": OCCUPANCY_THRESHOLD,
            "free_thresh": 0.25,
            "negate": 0
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
     

# ================== 主流程 ==================
if __name__ == "__main__":
    # 步骤1: 生成 Gazebo 障碍物
    obstacle_gen = ObstacleGenerator()
    obstacle_gen.generate_in_gazebo(num_obstacles=75)  # 生成15个障碍物
    
    # 步骤2: 生成 PGM 地图
    map_gen = PGMMapGenerator()
    for obs in obstacle_gen.obstacles:
        x, y, size_x, size_y = obs
        map_gen.add_obstacle(x, y, size_x, size_y)
    # 保存文件
    print(ppgm)
    print(pyaml)
    map_gen.save()
    print("PGM地图已生成: gazebo_map.pgm 和 gazebo_map.yaml")
