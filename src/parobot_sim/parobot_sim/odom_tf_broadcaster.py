#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__("odom_tf_broadcaster")

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_cb,
            10
        )

        self.get_logger().info("OdomTFBroadcaster started (publishing odom -> base_link TF)")

    def odom_cb(self, msg: Odometry):
        t = TransformStamped()

        # Use the odom message frames directly
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id       # usually "odom"
        t.child_frame_id = msg.child_frame_id         # usually "base_link"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTFBroadcaster()
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
