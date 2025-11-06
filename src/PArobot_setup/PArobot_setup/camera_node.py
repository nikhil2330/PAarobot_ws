#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from picamera2 import Picamera2
import cv2
import numpy as np


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # --- Parameters ---
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('show_preview', False)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.show_preview = self.get_parameter('show_preview').value

        # --- ROS setup ---
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # --- Initialize Picamera2 ---
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.get_logger().info(
            f"📷 Picamera2 started at {self.width}x{self.height} @ {self.fps} fps"
        )

        # --- Timer to capture and publish frames ---
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)

    def publish_frame(self):
        try:
            frame = self.picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            self.publisher_.publish(msg)

            if self.show_preview:
                cv2.imshow("Camera Preview", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Camera capture error: {e}")

    def destroy_node(self):
        self.get_logger().info("Stopping camera node...")
        try:
            self.picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Camera node interrupted, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
