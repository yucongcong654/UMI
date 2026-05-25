"""
Vive Tracker模块 - 基于pysurvive库
提供对Vive Tracker设备位姿数据的访问接口
"""

import sys
import time
import threading
import queue
import logging
import math
import numpy as np
import pysurvive

from src.pose_utils import normalize_quaternion

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('umi.vive_tracker')

class PoseData:
    """位姿数据结构，用于存储和格式化位姿信息"""
    def __init__(self, device_name, timestamp, position, rotation):
        self.device_name = device_name
        self.timestamp = timestamp
        self.position = position  # [x, y, z]
        self.rotation = rotation  # [x, y, z, w] 四元数

    def __str__(self):
        """格式化输出位姿信息"""
        return f"{self.device_name}: T: {self.timestamp:.6f} P: {self.position[0]:9.6f}, {self.position[1]:9.6f}, {self.position[2]:9.6f} R: {self.rotation[0]:9.6f}, {self.rotation[1]:9.6f}, {self.rotation[2]:9.6f}, {self.rotation[3]:9.6f}"

class ViveTracker:
    """
    Vive Tracker设备类，提供对Vive Tracker设备位姿数据的访问接口
    
    参数:
        config_path (str, optional): 配置文件路径
        lh_config (str, optional): 灯塔配置
        args (list, optional): 其他pysurvive参数
    """
    
    def __init__(self, config_path=None, lh_config=None, args=None):
        self.config_path = config_path
        self.lh_config = lh_config
        self.args = args if args else []
        
        # 初始化状态变量
        self.running = False
        self.context = None
        self.pose_queue = queue.Queue(maxsize=100)  # 用于存储最新位姿的队列
        self.devices_info = {}  # 存储设备信息的字典
        self.data_lock = threading.Lock()
        self.latest_poses = {}  # 存储每个设备的最新位姿
        
        # 线程对象
        self.collector_thread = None
        self.processor_thread = None
        self.device_monitor_thread = None

    @property
    def is_connected(self):
        if not self.running or self.context is None:
            return False
        try:
            return bool(self.context.Running())
        except Exception:
            return False
    
    def connect(self):
        """
        初始化并连接到Vive Tracker设备
        
        返回:
            bool: 连接是否成功
        """
        if self.running:
            logger.warning("Vive Tracker已经连接")
            return True
        
        try:
            logger.info("正在初始化pysurvive...")
            
            # 构建pysurvive参数
            survive_args = sys.argv[:1]  # 保留程序名
            
            # 添加配置文件参数
            if self.config_path:
                survive_args.extend(['--config', self.config_path])
            
            # 添加灯塔配置参数
            if self.lh_config:
                survive_args.extend(['--lh', self.lh_config])
            
            # 添加其他参数
            survive_args.extend(self.args)
            
            # 初始化pysurvive上下文
            self.context = pysurvive.SimpleContext(survive_args)
            if not self.context:
                logger.error("错误: 无法初始化pysurvive上下文")
                return False
            
            logger.info("pysurvive初始化成功")
            
            # 标记为运行状态
            self.running = True
            
            # 创建并启动位姿收集线程
            self.collector_thread = threading.Thread(target=self._pose_collector)
            self.collector_thread.daemon = True
            self.collector_thread.start()
            
            # 创建并启动位姿处理线程
            self.processor_thread = threading.Thread(target=self._pose_processor)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            
            # 创建并启动设备监控线程
            self.device_monitor_thread = threading.Thread(target=self._device_monitor)
            self.device_monitor_thread.daemon = True
            self.device_monitor_thread.start()
            
            logger.info("Vive Tracker位姿追踪已启动")
            
            # 等待初始数据
            time.sleep(0.5)
            return True
            
        except Exception as e:
            logger.error(f"连接Vive Tracker时发生错误: {e}")
            self.running = False
            return False

    def disconnect(self):
        """
        断开Vive Tracker设备连接
        """
        if not self.running:
            return
        
        logger.info("正在停止Vive Tracker位姿追踪...")
        self.running = False
        
        # 等待线程结束
        if self.collector_thread:
            self.collector_thread.join(timeout=2.0)
        
        if self.processor_thread:
            self.processor_thread.join(timeout=2.0)
            
        if self.device_monitor_thread:
            self.device_monitor_thread.join(timeout=2.0)
        
        # 清理资源
        self.context = None
        self.pose_queue = queue.Queue(maxsize=100)
        self.collector_thread = None
        self.processor_thread = None
        self.device_monitor_thread = None
        with self.data_lock:
            self.latest_poses = {}
        
        # 打印统计信息
        logger.info("设备统计信息:")
        for device_name, info in self.devices_info.items():
            logger.info(f"  - {device_name}: 更新次数 {info['updates']}")
        
        logger.info("Vive Tracker已断开连接")
    
    def _device_monitor(self):
        """
        设备监控线程函数
        定期检查新设备并更新设备列表
        """
        logger.info("设备监控线程已启动")
        
        # 初始化设备列表
        self._update_device_list()
        
        # 定期检查新设备
        while self.running and self.context.Running():
            # 更新设备列表
            self._update_device_list()
            
            # 每秒检查一次
            time.sleep(1.0)
    
    def _update_device_list(self):
        """
        更新设备列表
        """
        if not self.is_connected:
            return
        try:
            # 获取当前所有设备
            devices = list(self.context.Objects())
            
            # 更新设备信息字典
            with self.data_lock:
                for device in devices:
                    device_name = str(device.Name(), 'utf-8')
                    if device_name not in self.devices_info:
                        logger.info(f"检测到新设备: {device_name}")
                        self.devices_info[device_name] = {"updates": 0, "last_update": 0}
        except Exception as e:
            logger.error(f"更新设备列表时出错: {e}")
    
    def _pose_collector(self):
        """
        位姿收集线程函数
        持续从pysurvive获取最新位姿数据并放入队列
        """
        logger.info("位姿收集线程已启动")
        
        # 获取并打印所有可用设备
        devices = list(self.context.Objects())
        if not devices:
            logger.warning("警告: 未检测到任何设备")
        else:
            logger.info(f"检测到 {len(devices)} 个设备:")
            for device in devices:
                device_name = str(device.Name(), 'utf-8')
                logger.info(f"  - {device_name}")
                self.devices_info[device_name] = {"updates": 0, "last_update": 0}
        
        # 持续获取最新位姿
        while self.running and self.context.Running():
            updated = self.context.NextUpdated()
            if updated:
                # 获取设备名称
                device_name = str(updated.Name(), 'utf-8')
                
                # 如果是新设备，添加到设备信息字典
                with self.data_lock:
                    if device_name not in self.devices_info:
                        logger.info(f"检测到新设备更新: {device_name}")
                        self.devices_info[device_name] = {"updates": 0, "last_update": 0}
                
                # 获取位姿数据
                pose_obj = updated.Pose()
                pose_data = pose_obj[0]  # 位姿数据
                timestamp = pose_obj[1]  # 时间戳
                
                x, y, z, qx, qy, qz, qw = pose_data.Pos[0], pose_data.Pos[1], pose_data.Pos[2], pose_data.Rot[1], pose_data.Rot[2], pose_data.Rot[3], pose_data.Rot[0]
                position = np.array([x, y, z], dtype=np.float64)
                rotation = normalize_quaternion(np.array([qx, qy, qz, qw], dtype=np.float64))

                # 创建位姿数据对象
                pose = PoseData(
                    device_name,
                    timestamp,
                    position.tolist(),
                    rotation.tolist(),
                )
                
                # 更新设备信息
                with self.data_lock:
                    if device_name in self.devices_info:
                        self.devices_info[device_name]["updates"] += 1
                        self.devices_info[device_name]["last_update"] = time.time()
                
                # 将位姿数据放入队列，如果队列满则丢弃旧数据
                try:
                    self.pose_queue.put_nowait(pose)
                except queue.Full:
                    try:
                        self.pose_queue.get_nowait()  # 丢弃最旧的数据
                        self.pose_queue.put_nowait(pose)
                    except:
                        pass  # 忽略可能的错误
    
    def _pose_processor(self):
        """
        位姿处理线程函数
        从队列中获取并处理位姿数据，更新最新位姿字典
        """
        logger.info("位姿处理线程已启动")
        
        while self.running:
            try:
                # 尝试从队列获取位姿数据，设置超时以便定期检查running状态
                pose = self.pose_queue.get(timeout=0.1)
                
                # 更新最新位姿字典
                with self.data_lock:
                    self.latest_poses[pose.device_name] = pose
                
                # 在这里可以添加自定义的位姿处理逻辑
                # 例如: 发送到其他应用程序、记录到文件、进行分析等
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"处理位姿数据时出错: {e}")

    def read_once(self, device_name=None):
        if not self.is_connected:
            logger.warning("Vive Tracker未连接，返回空位姿数据")
            return None if device_name else {}
        
        # 强制更新一次设备列表，确保能获取到最新添加的设备
        self._update_device_list()
        
        with self.data_lock:
            if device_name:
                return self.latest_poses.get(device_name)
            else:
                return self.latest_poses.copy()
    
    def get_devices(self):
        """
        获取所有已检测到的设备列表
        
        返回:
            list: 设备名称列表
        """
        if not self.is_connected:
            return []

        self._update_device_list()
        
        with self.data_lock:
            return list(self.devices_info.keys())
    
    def get_device_info(self, device_name=None):
        """
        获取设备信息
        
        参数:
            device_name (str, optional): 设备名称，如果为None则返回所有设备的信息
        
        返回:
            dict: 设备信息字典
        """
        if not self.is_connected:
            return None if device_name else {}

        self._update_device_list()
        
        with self.data_lock:
            if device_name:
                return self.devices_info.get(device_name)
            else:
                return self.devices_info.copy()

    def close(self):
        self.disconnect()
    
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        self.disconnect()
