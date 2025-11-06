#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gpiozero import LED, PWMLED


class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.get_logger().info('Motor Node Started (BTS7960 tank drive)')

        # Subscribe to velocity commands
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        # ----------------- BTS7960 PIN MAP (BOARD numbering) -----------------
        # Left BTS7960
        self.l_rpwm = PWMLED("BOARD33", frequency=200)  # GPIO13 -> RPWM (Left)
        self.l_lpwm = PWMLED("BOARD31", frequency=200)  # GPIO6  -> LPWM (Left)
        self.l_en   = LED("BOARD29")                    # GPIO5  -> R_EN & L_EN (tie both to this)

        # Right BTS7960
        self.r_rpwm = PWMLED("BOARD35", frequency=200)  # GPIO19 -> RPWM (Right)
        self.r_lpwm = PWMLED("BOARD37", frequency=200)  # GPIO26 -> LPWM (Right)
        self.r_en   = LED("BOARD32")                    # GPIO12 -> R_EN & L_EN (tie both to this)

        # Optional inversion flags if a side spins backward due to wiring
        self.INV_LEFT  = False
        self.INV_RIGHT = False

        # Deadband
        self.MIN_PWM = 0.32  # tune per motor: 0.28–0.40 typical

        # Enable drivers
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

    def _apply_deadband(self, val: float) -> float:
        if val == 0.0:
            return 0.0
        s = 1 if val > 0 else -1
        a = abs(val)
        return s * (self.MIN_PWM + (1.0 - self.MIN_PWM) * a)

    def _drive_side(self, rpwm: PWMLED, lpwm: PWMLED, val: float):
        """val in [-1..1]; >0 uses RPWM, <0 uses LPWM, 0 coasts."""
        if val > 0:
            lpwm.value = 0
            rpwm.value = val
        elif val < 0:
            rpwm.value = 0
            lpwm.value = -val
        else:
            rpwm.value = 0
            lpwm.value = 0

    def tank(self, l: float, r: float):
        """
        Tank command in [-100..100] per side.
        Exactly matches main3.py mapping:
          - First map [-100..100] -> [-1..1]
          - Apply inversion + deadband
          - Then drive BTS7960 with RPWM/LPWM
        """
        # map from [-100..100] to [-1..1]
        l = max(-100.0, min(100.0, float(l))) / 100.0
        r = max(-100.0, min(100.0, float(r))) / 100.0

        if self.INV_LEFT:
            l = -l
        if self.INV_RIGHT:
            r = -r

        l = self._apply_deadband(l)
        r = self._apply_deadband(r)

        self._drive_side(self.l_rpwm, self.l_lpwm, l)
        self._drive_side(self.r_rpwm, self.r_lpwm, r)

    # ---------- ROS callback ----------
    def cmd_callback(self, msg: Twist):
        """
        Convert Twist to tank left/right exactly like main3.py’s:
            left  = current_forward - current_turn
            right = current_forward + current_turn

        Here:
            current_forward = msg.linear.x * 100
            current_turn    = msg.angular.z * 100
        """
        current_forward = msg.linear.x * 100.0
        current_turn    = msg.angular.z * 100.0

        left  = current_forward - current_turn
        right = current_forward + current_turn

        # clamp to [-100,100]
        left  = max(min(left, 100.0), -100.0)
        right = max(min(right, 100.0), -100.0)

        self.get_logger().info(f"Tank drive -> L:{left:.1f}, R:{right:.1f}")
        self.tank(left, right)

    def destroy_node(self):
        # Stop motors and disable drivers on shutdown
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
