#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import numpy as np

# ---- CONSTANTS from main7.py ----
CAMERA_FOV_DEG = 60.0
CAMERA_FOV_RAD = math.radians(CAMERA_FOV_DEG)

TURN_SPEED_TARGET    = 25.0
FORWARD_SPEED_TARGET = 30.0
CENTER_DEADZONE      = 0.10

ACCEL_LINEAR = 100.0
ACCEL_TURN   = 100.0

FOLLOW_NEAR = 0.8
FOLLOW_FAR  = 1.5

RANGE_MIN      = 0.15
RANGE_MAX      = 5.00
CONE_HALF_W    = math.radians(6.0)
PLOT_MAX_RANGE = 4.0


def smooth(prev, target, accel_per_sec, dt):
    max_step = accel_per_sec * dt
    delta = np.clip(target - prev, -max_step, +max_step)
    return prev + delta


class FollowerNode(Node):
    def __init__(self):
        super().__init__("follower_node")

        # Parameters (image width for center_x normalisation)
        self.declare_parameter("image_width", 640)
        self.imW = int(self.get_parameter("image_width").value)

        # State
        self.robot_enabled = False
        self.center_x = -1.0  # -1 = no lock
        self.forward_cmd = 0.0
        self.turn_cmd    = 0.0
        self.last_time   = time.time()

        self.front_distance = None
        self.aim_angle      = 0.0

        # ROS I/O
        self.create_subscription(Float32, "/person/center_x", self.center_cb, 10)
        self.create_subscription(Bool, "/robot_enabled", self.rf_cb, 10)

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, "/scan", self.scan_cb, scan_qos)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Timer for control loop (~33 Hz, same spirit as while loop)
        self.timer = self.create_timer(0.03, self.control_loop)
        self.get_logger().info("FollowerNode (EXACT main7.py motion) started")

        self.last_log_time = time.time()
        self.log_period = 1.0 

    # ---------- Callbacks ----------
    def rf_cb(self, msg: Bool):
        self.robot_enabled = msg.data

    def center_cb(self, msg: Float32):
        self.center_x = msg.data

    def scan_cb(self, msg: LaserScan):
        """Equivalent of read_lidar_cone using LaserScan instead of ydlidar directly."""
        ang_min = msg.angle_min
        ang_inc = msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)
        angs = ang_min + np.arange(len(ranges)) * ang_inc

        def valid(mask):
            vals = ranges[mask]
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals > RANGE_MIN) & (vals <= RANGE_MAX)]
            return vals

        # front cone around current aim_angle
        front_mask = (angs >= (self.aim_angle - CONE_HALF_W)) & (angs <= (self.aim_angle + CONE_HALF_W))
        front_vals = valid(front_mask)
        if front_vals.size > 0:
            self.front_distance = float(np.median(front_vals))
        else:
            self.front_distance = None

    # ---------- Control loop (EXACT logic of follow part) ----------
    def control_loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        dt = max(0.0, min(dt, 0.2))

        # has_lock_and_bbox equivalent: center_x >= 0 from detector
        has_lock = (self.center_x >= 0.0)

        target_forward_cmd = 0.0
        target_turn_cmd = 0.0

        if has_lock:
            frame_center_x = self.imW / 2.0
            x_center = self.center_x
            offset_norm = (x_center - frame_center_x) / frame_center_x
            self.aim_angle = offset_norm * (CAMERA_FOV_RAD / 2.0)

            if offset_norm < -CENTER_DEADZONE:
                target_turn_cmd = +TURN_SPEED_TARGET
            elif offset_norm > +CENTER_DEADZONE:
                target_turn_cmd = -TURN_SPEED_TARGET
            else:
                target_turn_cmd = 0.0
        else:
            offset_norm = 0.0

        # front_distance is updated in scan_cb with current aim_angle
        if has_lock and (self.front_distance is not None):
            if self.front_distance > FOLLOW_FAR:
                target_forward_cmd = +FORWARD_SPEED_TARGET
            elif self.front_distance < FOLLOW_NEAR:
                target_forward_cmd = -FORWARD_SPEED_TARGET
            else:
                target_forward_cmd = 0.0
        else:
            target_forward_cmd = 0.0

        if not has_lock:
            target_forward_cmd = 0.0
            target_turn_cmd = 0.0

        # RF gating and smoothing (mirrors main7 while loop)
        if self.robot_enabled:
            self.forward_cmd = smooth(self.forward_cmd, target_forward_cmd, ACCEL_LINEAR, dt)
            self.turn_cmd    = smooth(self.turn_cmd,    target_turn_cmd,   ACCEL_TURN,   dt)

            left  = np.clip(self.forward_cmd - self.turn_cmd, -100, 100)
            right = np.clip(self.forward_cmd + self.turn_cmd, -100, 100)

            twist = Twist()
            twist.linear.x  = float(self.forward_cmd / 100.0)
            twist.angular.z = float(self.turn_cmd    / 100.0)
            self.cmd_pub.publish(twist)

            motion_str = ""
            if self.forward_cmd > 1.0:
                if self.turn_cmd > 1.0:
                    motion_str = "FORWARD + TURN LEFT"
                elif self.turn_cmd < -1.0:
                    motion_str = "FORWARD + TURN RIGHT"
                else:
                    motion_str = "FORWARD STRAIGHT"
            elif self.forward_cmd < -1.0:
                if self.turn_cmd > 1.0:
                    motion_str = "BACKWARD + TURN LEFT"
                elif self.turn_cmd < -1.0:
                    motion_str = "BACKWARD + TURN RIGHT"
                else:
                    motion_str = "BACKWARD STRAIGHT"
            else:
                if self.turn_cmd > 1.0:
                    motion_str = "TURN LEFT IN PLACE"
                elif self.turn_cmd < -1.0:
                    motion_str = "TURN RIGHT IN PLACE"
                else:
                    motion_str = "STOP"

            # now = time.time()
            # if (now - self.last_log_time) >= self.log_period:
            #     self.last_log_time = now
            #     self.get_logger().info(
            #         f"[MOTION] {motion_str} | "
            #         f"dist={self.front_distance if self.front_distance is not None else 'None'} | "
            #         f"L={left:.1f} R={right:.1f}"
            #     )
        else:
            # RF STOP: same as main7 (forward_cmd, turn_cmd → 0 and motors stopped via MotorNode)
            self.forward_cmd = 0.0
            self.turn_cmd    = 0.0
            twist = Twist()
            self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
