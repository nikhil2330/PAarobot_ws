import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from picamera2 import Picamera2
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # --- Parameters (match main7.py resolution) ---
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)

        # --- ROS setup ---
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # --- Initialize Picamera2 (like VideoStream in main7.py) ---
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
            frame_rgb = self.picam2.capture_array("main")  # same as VideoStream.read()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Camera capture error: {e}")

    def destroy_node(self):
        self.get_logger().info("Stopping camera node...")
        try:
            self.picam2.stop()
        except Exception:
            pass
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
