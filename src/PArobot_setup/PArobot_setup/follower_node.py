#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import numpy as np
import math
import time

# Optional plotting (like main3.py) – leave commented until you want it:
# import matplotlib.pyplot as plt


class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')

        # ==================== PARAMETERS (basic) ====================
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('cam_fov_deg', 60.0)

        self.imW = int(self.get_parameter('image_width').value)
        self.cam_fov = math.radians(float(self.get_parameter('cam_fov_deg').value))

        # ==================== CONTROL CONSTANTS (from main3.py) ====================

        # --- Motion & turning ---
        self.turn_pwm       = 30.0   # max turning PWM (0..100)
        self.forward_pwm    = 38.0   # max forward PWM (0..100)

        # --- Behavior tuning ---
        self.accel_rate_fwd       = 36.0   # forward ramp (PWM units/sec)
        self.accel_rate_turn_up   = 22.0   # ramp when |target| is growing (gentle)
        self.accel_rate_turn_down = 150.0  # ramp when braking / sign change (fast)

        self.alpha           = 0.30       # smoothing factor for x-position
        self.dead_zone_ratio = 0.18
        self.turn_sensitivity = 0.80
        self.turn_D          = 0.22
        self.zero_cross_band = 0.08

        # Bias knobs
        self.CENTER_BIAS_PX = 0.0   # +ve shifts neutral point to the RIGHT
        self.TURN_BIAS      = 0.0   # adds constant bias to turn output
        self.LEFT_GAIN      = 1.00  # per-side scaling (if robot pulls)
        self.RIGHT_GAIN     = 1.00

        self.lost_timeout   = 1.0   # seconds

        # --- Distance thresholds (m) ---
        self.FOLLOW_FAR   = 1.20
        self.FOLLOW_NEAR  = 0.80
        self.SIDE_WARN    = 0.50
        self.SIDE_STOP    = 0.30
        self.RANGE_MIN    = 0.15
        self.RANGE_MAX    = 4.00
        self.CLOSE_RATE   = 0.30

        # --- LiDAR cone geometry ---
        self.CENTER_FOV   = self.cam_fov  # 60 deg
        self.FRONT_MARGIN = math.radians(6.0)
        self.SIDE_OFFSET  = math.radians(20.0)
        self.SIDE_SPREAD  = math.radians(40.0)

        # --- Search mode (replaces recovery) ---
        self.SEARCH_SPIN_TURN = 8.0    # slow spin
        self.SEARCH_FLIP_SEC  = 6.0    # flip spin direction every N sec
        self.STOP_PAUSE_SEC   = 1.5    # stand still before spinning

        # ==================== STATE ====================
        self.last_update = time.time()
        self.last_ctrl_time = self.last_update

        self.current_forward = 0.0   # current forward PWM
        self.current_turn    = 0.0   # current turn PWM

        self.prev_dist      = None
        self.front_distance = None
        self.left_min       = None
        self.right_min      = None

        self.smooth_x        = None
        self.center_x        = -1.0
        self.center_angle    = 0.0
        self.person_visible  = False
        self.last_seen_time  = 0.0
        self.last_seen_angle = 0.0
        self.last_seen_dist  = 1.0

        self.search_mode  = False
        self.search_stage = 0   # 0 = pause, 1 = spin
        self.search_start = 0.0
        self.scan_dir     = 1.0
        self.prev_error   = 0.0

        # ==================== OPTIONAL PLOT SETUP (commented) ====================
        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # self.ax.set_aspect('equal')
        # self.ax.set_xlim(-1, 1)
        # self.ax.set_ylim(0, 3)
        # self.ax.grid(True)
        # self.lidar_points_plot, = self.ax.plot([], [], 'b.', markersize=3, label="LIDAR points")
        # self.robot_dot, = self.ax.plot(0, 0, 'ro', markersize=6, label="Robot")
        # self.fov_fill = self.ax.fill([], [], 'orange', alpha=0.2, label="Camera FOV")[0]
        # self.fov_left_line,  = self.ax.plot([], [], 'orange', linewidth=1.5)
        # self.fov_right_line, = self.ax.plot([], [], 'orange', linewidth=1.5)
        # self.ax.legend(loc="upper right")

        # ==================== ROS I/O ====================
        self.create_subscription(Float32, '/person/center_x', self.center_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.03, self.control_loop)  # ~33 Hz

        self.get_logger().info("FollowerNode (main3-style behavior) initialized")

    # ==================== CALLBACKS ====================
    def center_cb(self, msg: Float32):
        """
        Gets center_x in pixels from person_detector_node.
        - >=0 means person (or short-term memory) present
        - -1 means lost
        """
        x = msg.data
        now = time.time()

        if x < 0:
            self.person_visible = False
            self.center_x = -1.0
            # don't change last_seen_time here – we use it for lost_timeout
            return

        self.center_x = x
        self.person_visible = True
        self.last_seen_time = now

        # Compute angle relative to camera center (like main3.py)
        frame_center = (self.imW / 2.0) + self.CENTER_BIAS_PX
        x_offset = (x - frame_center) / frame_center  # [-1..1]
        center_angle_temp = x_offset * (self.CENTER_FOV / 2.0)

        self.center_angle    = center_angle_temp
        self.last_seen_angle = center_angle_temp

        # If we were in search mode but see a person again, exit search
        if self.search_mode:
            self.search_mode  = False
            self.search_stage = 0

    def scan_cb(self, msg: LaserScan):
        """
        Equivalent to main3.py's read_lidar(aim_angle).
        Uses current center_angle (or last_seen_angle) to compute:
         - front_distance
         - left_min / right_min
        """
        ang_min = msg.angle_min
        ang_inc = msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)

        # Build angle array of same length as ranges
        angs = ang_min + np.arange(len(ranges)) * ang_inc

        # Use last known aim angle
        aim_angle = self.center_angle if self.person_visible else self.last_seen_angle

        def valid_vals(mask):
            vals = ranges[mask]
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals > self.RANGE_MIN) & (vals < self.RANGE_MAX)]
            return vals

        # FRONT window
        front_mask = (angs >= (aim_angle - self.FRONT_MARGIN)) & \
                     (angs <= (aim_angle + self.FRONT_MARGIN))
        front_vals = valid_vals(front_mask)
        if front_vals.size > 0:
            s = np.sort(front_vals)
            self.front_distance = float(np.median(s[:3])) if s.size >= 3 else float(np.median(s))
        else:
            self.front_distance = None

        # LEFT window
        left_mask = (angs > (aim_angle + self.SIDE_OFFSET)) & \
                    (angs < (aim_angle + self.SIDE_OFFSET + self.SIDE_SPREAD))
        left_vals = valid_vals(left_mask)
        self.left_min = float(np.min(left_vals)) if left_vals.size > 0 else None

        # RIGHT window
        right_mask = (angs > (aim_angle - self.SIDE_OFFSET - self.SIDE_SPREAD)) & \
                     (angs < (aim_angle - self.SIDE_OFFSET))
        right_vals = valid_vals(right_mask)
        self.right_min = float(np.min(right_vals)) if right_vals.size > 0 else None

        # -------- OPTIONAL PLOT UPDATE (commented) --------
        # try:
        #     xs, ys = [], []
        #     cone_width = self.FRONT_MARGIN
        #     fov_range = 3.0
        #     for angle, r in zip(angs, ranges):
        #         if ((aim_angle - cone_width) <= angle <= (aim_angle + cone_width)
        #                 and self.RANGE_MIN < r <= fov_range):
        #             xs.append(r * math.sin(angle))
        #             ys.append(r * math.cos(angle))
        #     self.lidar_points_plot.set_data(xs, ys)
        #     fov_color = 'green' if self.person_visible else 'red'
        #     left_x = [0, fov_range * math.sin(aim_angle - cone_width)]
        #     left_y = [0, fov_range * math.cos(aim_angle - cone_width)]
        #     right_x = [0, fov_range * math.sin(aim_angle + cone_width)]
        #     right_y = [0, fov_range * math.cos(aim_angle + cone_width)]
        #     cone_fill_x = [0,
        #                    fov_range * math.sin(aim_angle - cone_width),
        #                    fov_range * math.sin(aim_angle + cone_width)]
        #     cone_fill_y = [0,
        #                    fov_range * math.cos(aim_angle - cone_width),
        #                    fov_range * math.cos(aim_angle + cone_width)]
        #     self.fov_fill.set_xy(np.column_stack((cone_fill_x, cone_fill_y)))
        #     self.fov_fill.set_facecolor(fov_color)
        #     self.fov_left_line.set_data(left_x, left_y)
        #     self.fov_right_line.set_data(right_x, right_y)
        #     self.fov_left_line.set_color(fov_color)
        #     self.fov_right_line.set_color(fov_color)
        #     self.fig.canvas.draw()
        #     self.fig.canvas.flush_events()
        # except Exception:
        #     pass

    # ==================== CONTROL LOOP ====================
    def control_loop(self):
        now = time.time()
        dt_loop = now - self.last_update
        self.last_update = now

        # ---------- LOST / SEARCH LOGIC ----------
        if (not self.person_visible) and \
           ((now - self.last_seen_time) > self.lost_timeout) and \
           (not self.search_mode):
            self.search_mode  = True
            self.search_stage = 0
            self.search_start = now
            self.scan_dir = 1.0 if self.last_seen_angle >= 0 else -1.0
            self.get_logger().warn("Person lost → SIMPLE SEARCH: pause then spin")

        detected = self.person_visible
        target_forward = 0.0
        target_turn    = 0.0

        # =====================================================================
        # NORMAL FOLLOW (person visible and NOT searching)
        # =====================================================================
        if detected and not self.search_mode and self.center_x >= 0:
            x = self.center_x

            # Smooth x tracking
            self.smooth_x = x if (self.smooth_x is None) else \
                               (self.alpha * x + (1.0 - self.alpha) * self.smooth_x)

            frame_center = (self.imW / 2.0) + self.CENTER_BIAS_PX
            error_ratio = (self.smooth_x - frame_center) / frame_center  # [-1..1]

            # modest dead-zone: do nothing very near center
            if abs(error_ratio) < self.dead_zone_ratio:
                target_turn = 0.0
                self.prev_error = 0.0
            else:
                # zero-cross short brake: if crossing near center, command zero
                dt_ctrl = max(1e-3, now - self.last_ctrl_time)
                self.last_ctrl_time = now

                if (np.sign(error_ratio) != np.sign(self.prev_error)) and \
                   (abs(error_ratio) < self.zero_cross_band):
                    target_turn = 0.0
                else:
                    # PD turn control + nonlinearity
                    d_error = (error_ratio - self.prev_error) / dt_ctrl
                    d_error = float(np.clip(d_error, -6.0, 6.0))
                    self.prev_error = error_ratio

                    near_center_scale = 0.6 + 0.4 * abs(error_ratio)  # 0.6..1.0
                    kP = self.turn_sensitivity * near_center_scale

                    target_turn = np.clip(
                        -self.turn_pwm * (kP * error_ratio + self.turn_D * d_error),
                        -self.turn_pwm,
                        self.turn_pwm
                    )

            # Distance band (front_distance)
            if self.front_distance is not None:
                self.last_seen_dist = self.front_distance
                if self.front_distance > self.FOLLOW_FAR:
                    target_forward = self.forward_pwm
                elif self.front_distance < self.FOLLOW_NEAR:
                    target_forward = -self.forward_pwm
                else:
                    target_forward = 0.0

            # Obstacle shaping
            avoid_turn = 0.0
            allow_forward = 1.0

            if self.left_min is not None and self.left_min < self.SIDE_WARN:
                avoid_turn += 0.5
            if self.right_min is not None and self.right_min < self.SIDE_WARN:
                avoid_turn -= 0.5
            if (self.left_min is not None and self.left_min < self.SIDE_STOP) or \
               (self.right_min is not None and self.right_min < self.SIDE_STOP):
                allow_forward = 0.0

            target_turn    += avoid_turn * self.turn_pwm
            target_turn    += self.TURN_BIAS
            target_forward *= allow_forward

            # Predictive braking
            if (self.prev_dist is not None) and (self.front_distance is not None):
                rate = (self.prev_dist - self.front_distance) / max(1e-3, dt_loop)
                if rate > self.CLOSE_RATE:
                    target_forward *= max(0.0, 1.0 - rate)
            self.prev_dist = self.front_distance

        # =====================================================================
        # SIMPLE SEARCH (no recovery state machine)
        # =====================================================================
        elif self.search_mode:
            elapsed = now - self.search_start
            if self.search_stage == 0:
                # stop & wait briefly
                target_forward = 0.0
                target_turn = 0.0
                if elapsed >= self.STOP_PAUSE_SEC:
                    self.search_stage = 1
                    self.search_start = now
            elif self.search_stage == 1:
                # slow spin; flip direction periodically
                target_forward = 0.0
                target_turn = self.scan_dir * self.SEARCH_SPIN_TURN
                if elapsed >= self.SEARCH_FLIP_SEC:
                    self.scan_dir *= -1.0
                    self.search_start = now

        # =====================================================================
        # SMOOTH ACCEL + DRIVE (with asymmetric turn ramps)
        # =====================================================================
        ramp_step_fwd       = self.accel_rate_fwd       * dt_loop
        ramp_step_turn_up   = self.accel_rate_turn_up   * dt_loop
        ramp_step_turn_down = self.accel_rate_turn_down * dt_loop

        def ramp(current, target, step):
            if current < target:
                return min(current + step, target)
            if current > target:
                return max(current - step, target)
            return current

        def ramp_asym_turn(current, target, step_up, step_down):
            same_sign = (np.sign(current) == np.sign(target)) or (current == 0) or (target == 0)
            if same_sign and abs(target) > abs(current):
                step = step_up
            else:
                step = step_down
            if current < target:
                return min(current + step, target)
            if current > target:
                return max(current - step, target)
            return current

        self.current_forward = ramp(self.current_forward, target_forward, ramp_step_fwd)
        self.current_turn    = ramp_asym_turn(
            self.current_turn, target_turn,
            ramp_step_turn_up, ramp_step_turn_down
        )

        # Compute final left/right PWM (for logging only; MotorNode reconstructs this)
        left  = np.clip(self.current_forward - self.current_turn, -100.0, 100.0)
        right = np.clip(self.current_forward + self.current_turn, -100.0, 100.0)

        # Per-side gain correction
        left  = float(np.clip(left * self.LEFT_GAIN,  -100.0, 100.0))
        right = float(np.clip(right * self.RIGHT_GAIN, -100.0, 100.0))

        # Publish cmd_vel where:
        #   linear.x  = current_forward / 100
        #   angular.z = current_turn    / 100
        # so MotorNode yields the same left/right as main3.py
        twist = Twist()
        twist.linear.x  = float(self.current_forward / 100.0)
        twist.angular.z = float(self.current_turn    / 100.0)
        self.cmd_pub.publish(twist)

        mode_str = 'SEARCH' if self.search_mode else 'FOLLOW'
        self.get_logger().info(
            f"L:{left:.1f} R:{right:.1f} | mode={mode_str} "
            f"| fd={self.front_distance if self.front_distance is not None else -1:.2f} "
            f"| lm={self.left_min if self.left_min is not None else -1:.2f} "
            f"| rm={self.right_min if self.right_min is not None else -1:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
