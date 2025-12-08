#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, Bool
from cv_bridge import CvBridge

import numpy as np
import cv2


class FakePersonDetectorNode(Node):
    """
    Fake person detector for simulation.

    - Subscribes: /camera/image_raw
    - Looks for a RED blob (our "person" box in the world)
    - Publishes same topics as real detector:
        /person/center_x   (Float32, -1.0 = no detection)
        /person/bbox       (Float32MultiArray: xmin,ymin,xmax,ymax,score)
        /person/locked     (Bool)
        /person/det_score  (Float32)
        /person/sim_score  (Float32)
    """

    def __init__(self):
        super().__init__('fake_person_detector_node')

        self.bridge = CvBridge()

        # parameters
        self.declare_parameter('min_area', 300.0)
        self.min_area = float(self.get_parameter('min_area').value)

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)

        # Publishers
        self.center_pub = self.create_publisher(Float32, '/person/center_x', 10)
        self.bbox_pub   = self.create_publisher(Float32MultiArray, '/person/bbox', 10)
        self.lock_pub   = self.create_publisher(Bool,    '/person/locked',    10)
        self.det_pub    = self.create_publisher(Float32, '/person/det_score', 10)
        self.sim_pub    = self.create_publisher(Float32, '/person/sim_score', 10)

        self.last_log_time = 0.0
        self.log_period = 1.0  # seconds

        self.get_logger().info(
            "FakePersonDetectorNode started (HSV red blob, interface compatible with real detector)."
        )

    def image_cb(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        h, w = frame_bgr.shape[:2]

        # BGR -> HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # RED in HSV: two ranges (0-10) and (160-179)
        lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # largest blob
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area >= self.min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                xmin, ymin, xmax, ymax = x, y, x + bw, y + bh
                cx = (xmin + xmax) / 2.0

                # Normal interface: like real detector when locked & valid bbox
                self.center_pub.publish(Float32(data=float(cx)))
                bbox_msg = Float32MultiArray()
                bbox_msg.data = [
                    float(xmin), float(ymin), float(xmax), float(ymax), 1.0
                ]
                self.bbox_pub.publish(bbox_msg)
                self.lock_pub.publish(Bool(data=True))
                self.det_pub.publish(Float32(data=1.0))
                self.sim_pub.publish(Float32(data=1.0))

                now = time.time()
                if now - self.last_log_time >= self.log_period:
                    self.last_log_time = now
                    self.get_logger().info(
                        f"[FAKE DETECT] area={area:.1f} bbox=({xmin},{ymin})-({xmax},{ymax}) cx={cx:.1f}"
                    )
                return

        # If we reach here → no valid detection
        self.center_pub.publish(Float32(data=-1.0))
        self.bbox_pub.publish(Float32MultiArray(data=[]))
        self.lock_pub.publish(Bool(data=False))
        self.det_pub.publish(Float32(data=-1.0))
        self.sim_pub.publish(Float32(data=-1.0))

    # end class


def main(args=None):
    rclpy.init(args=args)
    node = FakePersonDetectorNode()
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
