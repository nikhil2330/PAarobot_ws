#!/usr/bin/env python3
import threading
import serial
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE   = 9600


class RFRemoteNode(Node):
    def __init__(self):
        super().__init__("rf_remote_node")

        self.pub = self.create_publisher(Bool, "/robot_enabled", 10)
        self.robot_enabled = False
        self._ser = None

        self.get_logger().info(f"Opening RF serial on {SERIAL_PORT} @ {BAUD_RATE}")
        try:
            self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.get_logger().info("[RF] Serial port opened")
        except Exception as e:
            self.get_logger().error(f"[RF] Could not open serial port: {e}")
            self.get_logger().error("RF node idle (no serial).")
            return

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _publish_state(self, enabled: bool):
        if enabled == self.robot_enabled:
            return
        self.robot_enabled = enabled
        msg = Bool()
        msg.data = enabled
        self.pub.publish(msg)
        self.get_logger().info(f"[RF] Robot {'ENABLED' if enabled else 'DISABLED'}")

    def _loop(self):
        while rclpy.ok() and self._ser is not None:
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    self.get_logger().info(f"[RF] {line}")
                    if line == "START":
                        self._publish_state(True)
                    elif line == "STOP":
                        self._publish_state(False)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"[RF] Serial error: {e}")
                time.sleep(0.5)

    def destroy_node(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RFRemoteNode()
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
