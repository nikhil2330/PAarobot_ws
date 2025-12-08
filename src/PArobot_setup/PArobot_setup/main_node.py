import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')

        # Subscriptions
        self.create_subscription(Float32, '/person/center_x', self.person_center_cb, 10)
        self.create_subscription(Bool,   '/person/locked',    self.lock_cb,       10)
        self.create_subscription(Float32, '/person/det_score', self.det_cb,       10)
        self.create_subscription(Float32, '/person/sim_score', self.sim_cb,       10)

        self.create_subscription(LaserScan, '/scan',    self.lidar_cb, 10)
        self.create_subscription(Twist,     '/cmd_vel', self.cmd_cb,   10)

        # State
        self.person_visible = False
        self.person_locked  = False
        self.det_score      = -1.0
        self.sim_score      = -1.0

        self.last_distance = None
        self.last_cmd      = Twist()

        # Timer to print health info
        self.timer = self.create_timer(1.0, self.status_timer)

        self.get_logger().info("🧩 Main node online — supervising system...")

    # ---------- Callbacks ----------
    def person_center_cb(self, msg: Float32):
        # person visible if detector publishes valid center_x
        self.person_visible = (msg.data >= 0.0)

    def lock_cb(self, msg: Bool):
        self.person_locked = msg.data

    def det_cb(self, msg: Float32):
        self.det_score = msg.data

    def sim_cb(self, msg: Float32):
        self.sim_score = msg.data

    def lidar_cb(self, msg: LaserScan):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            self.last_distance = min(valid)
        else:
            self.last_distance = None

    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg

    # ---------- Periodic status print ----------
    def status_timer(self):
        # Person and lock status
        person_str = '✅' if self.person_visible else '❌'
        locked_str = '✅' if self.person_locked  else '❌'

        # Distance
        if self.last_distance is not None:
            dist_str = f"{self.last_distance:.2f}m"
        else:
            dist_str = "N/A"

        # Motion string (same logic as follower_node)
        v = self.last_cmd.linear.x
        w = self.last_cmd.angular.z

        if v > 0.01:
            if w > 0.01:
                motion_str = "FORWARD + TURN LEFT"
            elif w < -0.01:
                motion_str = "FORWARD + TURN RIGHT"
            else:
                motion_str = "FORWARD STRAIGHT"
        elif v < -0.01:
            if w > 0.01:
                motion_str = "BACKWARD + TURN LEFT"
            elif w < -0.01:
                motion_str = "BACKWARD + TURN RIGHT"
            else:
                motion_str = "BACKWARD STRAIGHT"
        else:
            if w > 0.01:
                motion_str = "TURN LEFT IN PLACE"
            elif w < -0.01:
                motion_str = "TURN RIGHT IN PLACE"
            else:
                motion_str = "STOP"

        # Scores
        det_str = f"{self.det_score:.2f}" if self.det_score >= 0.0 else "N/A"
        sim_str = f"{self.sim_score:.2f}" if self.sim_score >= 0.0 else "N/A"

        log = (
            f"Person: {person_str} | "
            f"Locked: {locked_str} | "
            f"Det: {det_str} | Sim: {sim_str} | "
            f"Min Dist: {dist_str} | "
            f"Motion: {motion_str} | "
            f"Vel: {v:.2f}, Turn: {w:.2f}"
        )
        self.get_logger().info(log)


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
