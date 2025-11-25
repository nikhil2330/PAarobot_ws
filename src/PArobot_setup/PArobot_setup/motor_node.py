#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from gpiozero import LED, PWMLED

# Motor inversion flags (match your globals)
INV_LEFT  = False
INV_RIGHT = False


class MotorNode(Node):
    def __init__(self):
        super().__init__("motor_node")
        self.get_logger().info("Motor Node Started (BTS7960 tank drive, EXACT main7.py mapping)")

        # Subscribe to velocity commands
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)

        # Subscribe to RF enable for hardware-level stop
        self.robot_enabled = False
        self.create_subscription(Bool, "/robot_enabled", self.rf_cb, 10)

        # ----------------- BTS7960 PIN MAP (BOARD numbering) -----------------
        # Left BTS7960
        self.l_rpwm = PWMLED("BOARD33", frequency=200)  # GPIO13 -> RPWM (Left)
        self.l_lpwm = PWMLED("BOARD31", frequency=200)  # GPIO6  -> LPWM (Left)
        self.l_en   = LED("BOARD29")                    # GPIO5  -> R_EN & L_EN

        # Right BTS7960
        self.r_rpwm = PWMLED("BOARD35", frequency=200)  # GPIO19 -> LPWM (Right)
        self.r_lpwm = PWMLED("BOARD37", frequency=200)  # GPIO26 -> RPWM (Right)
        self.r_en   = LED("BOARD32")                    # GPIO12 -> R_EN & L_EN

        self.enable_all()

    # ---------- BTS7960 helpers ----------
    def enable_all(self):
        self.l_en.on()
        self.r_en.on()
        self.get_logger().info("BTS7960 drivers ENABLED")

    def disable_all(self):
        self.l_en.off()
        self.r_en.off()
        self.get_logger().info("BTS7960 drivers DISABLED")

    def _drive_side(self, rpwm: PWMLED, lpwm: PWMLED, val: float):
        """val in [-1..1]; >0 uses rpwm, <0 uses lpwm, 0 stops. EXACTLY like your script."""
        if val > 0:
            lpwm.value = 0
            rpwm.value = val
        elif val < 0:
            rpwm.value = 0
            lpwm.value = -val
        else:
            rpwm.value = 0
            lpwm.value = 0

    def tank(self, left_cmd: float, right_cmd: float):
        """
        EXACT port of your tank():
            left_cmd, right_cmd in [-100..100]
            -> scale to [-1..1], apply inversion, drive BTS7960.
        """
        left  = max(-100.0, min(100.0, float(left_cmd)))  / 100.0
        right = max(-100.0, min(100.0, float(right_cmd))) / 100.0

        if INV_LEFT:
            left = -left
        if INV_RIGHT:
            right = -right

        self._drive_side(self.l_rpwm, self.l_lpwm, left)
        self._drive_side(self.r_rpwm, self.r_lpwm, right)

    # ---------- RF gating ----------
    def rf_cb(self, msg: Bool):
        if msg.data == self.robot_enabled:
            return
        self.robot_enabled = msg.data
        if not self.robot_enabled:
            # hardware-level immediate stop
            self.tank(0.0, 0.0)
            self.get_logger().warn("[RF] Robot disabled → HARD STOP")
        else:
            self.get_logger().info("[RF] Robot enabled")

    # ---------- ROS callback ----------
    def cmd_callback(self, msg: Twist):
        """
        follower_node publishes:
            linear.x  = forward_cmd / 100
            angular.z = turn_cmd    / 100

        Exactly like your script:
            left  = forward_cmd - turn_cmd
            right = forward_cmd + turn_cmd
        """
        if not self.robot_enabled:
            # Ignore commands if disabled; keep motors stopped
            self.tank(0.0, 0.0)
            return

        forward_cmd = msg.linear.x * 100.0
        turn_cmd    = msg.angular.z * 100.0

        left  = forward_cmd - turn_cmd
        right = forward_cmd + turn_cmd

        left  = max(-100.0, min(100.0, left))
        right = max(-100.0, min(100.0, right))

        self.tank(left, right)

    def destroy_node(self):
        try:
            self.tank(0.0, 0.0)
            self.disable_all()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
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
