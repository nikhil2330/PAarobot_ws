#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')

        # Subscriptions (listen to system state)
        self.create_subscription(Float32, '/person/center_x', self.person_center_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)

        # Example system status info
        self.person_visible = False
        self.last_distance = None
        self.last_cmd = Twist()

        # Timer to print health info
        self.timer = self.create_timer(2.0, self.status_timer)

        self.get_logger().info("🧩 Main node online — supervising system...")

    def person_center_cb(self, msg: Float32):
        self.person_visible = msg.data >= 0.0

    def lidar_cb(self, msg: LaserScan):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            self.last_distance = min(valid)

    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg

    def status_timer(self):
        """Prints periodic health updates — replace later with state logic."""
        status = f"Person: {'✅' if self.person_visible else '❌'} | "
        status += f"Min Dist: {self.last_distance:.2f}m | " if self.last_distance else "Min Dist: N/A | "
        status += f"Vel: {self.last_cmd.linear.x:.2f}, Turn: {self.last_cmd.angular.z:.2f}"
        self.get_logger().info(status)


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
