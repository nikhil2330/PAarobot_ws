#!/usr/bin/env python3
import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class RandomWalkNode(Node):
    def __init__(self):
        super().__init__("random_walk_node")

        # Publish velocity to diff drive
        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        # Not used yet, but kept for future
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # Lidar
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_cb,
            10,
        )

        self.last_scan = None

        # Motion params
        self.forward_speed = 0.3
        self.turn_speed = 0.8

        # Random wandering params
        self.change_prob = 0.03  # chance to change wandering direction each tick
        self.wander_state = "forward"  # forward / turn_left / turn_right / stop

        # Obstacle avoidance params
        self.front_obstacle_threshold = 0.4  # m: start avoiding when closer than this
        self.avoid_turn_ticks = 10          # how many timer ticks to keep turning ~ 1.5s @ 0.1s
        self.mode = "wander"                # "wander" or "avoid"
        self.avoid_direction = None         # "left" or "right"
        self.avoid_ticks_left = 0

        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info(
            "RandomWalkNode started (random wander + lidar avoidance with side choice)"
        )

    # ---------- Callbacks ----------

    def odom_cb(self, msg: Odometry):
        # Placeholder if you want to use odometry later
        pass

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    # ---------- Lidar helpers ----------

    def min_distance_in_sector(self, scan: LaserScan, angle_start: float, angle_end: float):
        """
        angle_start/end in radians, in the scan frame.
        Returns min distance in that sector or None if no valid readings.
        """
        if scan is None:
            return None

        if angle_start > angle_end:
            angle_start, angle_end = angle_end, angle_start

        i_start = int((angle_start - scan.angle_min) / scan.angle_increment)
        i_end = int((angle_end - scan.angle_min) / scan.angle_increment)

        i_start = max(0, min(len(scan.ranges) - 1, i_start))
        i_end = max(0, min(len(scan.ranges) - 1, i_end))

        if i_start > i_end:
            i_start, i_end = i_end, i_start

        vals = []
        for r in scan.ranges[i_start: i_end + 1]:
            if scan.range_min < r < scan.range_max and not math.isinf(r) and not math.isnan(r):
                vals.append(r)

        return min(vals) if vals else None

    # ---------- Wandering logic ----------

    def choose_new_wander_state(self):
        r = random.random()
        if r < 0.6:
            self.wander_state = "forward"
        elif r < 0.8:
            self.wander_state = "turn_left"
        elif r < 0.95:
            self.wander_state = "turn_right"
        else:
            self.wander_state = "stop"
        self.get_logger().debug(f"[wander] New state: {self.wander_state}")

    def apply_wander(self, twist: Twist):
        if random.random() < self.change_prob:
            self.choose_new_wander_state()

        if self.wander_state == "forward":
            twist.linear.x = self.forward_speed
        elif self.wander_state == "turn_left":
            twist.angular.z = self.turn_speed
        elif self.wander_state == "turn_right":
            twist.angular.z = -self.turn_speed
        # "stop" -> leave twist zero

    # ---------- Avoidance logic ----------

    def start_avoidance(self, left_min, right_min, front_min):
        """
        Decide which way to turn and set avoidance state for a fixed duration.
        """
        # Choose side with more space (larger min distance wins).
        # If right is None or left has more room, turn left. Otherwise turn right.
        if right_min is None or (left_min is not None and left_min > right_min):
            self.avoid_direction = "left"
        else:
            self.avoid_direction = "right"

        self.mode = "avoid"
        self.avoid_ticks_left = self.avoid_turn_ticks

        self.get_logger().info(
            f"[avoid] Obstacle ahead at {front_min:.2f} m, "
            f"turning {self.avoid_direction.upper()} "
            f"(left_min={left_min}, right_min={right_min})"
        )

    def apply_avoidance(self, twist: Twist):
        """
        Keep turning in the chosen direction for avoid_ticks_left steps.
        """
        if self.avoid_direction == "left":
            twist.angular.z = self.turn_speed
        elif self.avoid_direction == "right":
            twist.angular.z = -self.turn_speed

        # Optional: back up a bit while turning
        # twist.linear.x = -0.05

        self.avoid_ticks_left -= 1
        if self.avoid_ticks_left <= 0:
            self.mode = "wander"
            self.avoid_direction = None
            self.get_logger().debug("[avoid] Done, switching back to wander")

    # ---------- Main loop ----------

    def tick(self):
        twist = Twist()

        # Use lidar to decide whether to enter avoidance mode
        front_min = None
        left_min = None
        right_min = None

        if self.last_scan is not None:
            # Front: about [-30°, +30°] = [-0.52, +0.52] rad
            front_min = self.min_distance_in_sector(self.last_scan, -0.52, 0.52)

            # Left: [30°, 90°] = [0.52, 1.57]
            left_min = self.min_distance_in_sector(self.last_scan, 0.52, 1.57)

            # Right: [-90°, -30°] = [-1.57, -0.52]
            right_min = self.min_distance_in_sector(self.last_scan, -1.57, -0.52)

        # --- State transitions ---
        if self.mode == "wander":
            # Check if we need to switch to avoidance
            if front_min is not None and front_min < self.front_obstacle_threshold:
                self.start_avoidance(left_min, right_min, front_min)
                # fall-through: this tick will use avoidance
        # If already in "avoid", we stay until avoid_ticks_left runs out

        # --- Apply behavior ---
        if self.mode == "avoid":
            self.apply_avoidance(twist)
        else:
            self.apply_wander(twist)

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RandomWalkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
