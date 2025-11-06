import matplotlib.pyplot as plt
import numpy as np
import math
import time

# --- Tuned Parameters (matching improved follower_node behavior) ---
TURN_GAIN = 3.5
FORWARD_SPEED = 1.0
DEAD_ZONE_RATIO = 0.08
TURN_SENSITIVITY = 0.9
ACCEL_RATE = 2.0
FOLLOW_DIST_NEAR = 0.9
FOLLOW_DIST_FAR = 1.1
dt = 0.05  # 20 Hz
FOV_DEG = 60.0  # LiDAR + camera FOV (±30°)

# --- State ---
robot = np.array([0.0, 0.0])
heading = 0.0
person = np.array([2.0, 0.0])
prev_dist = None

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# --- Visualization setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)

# Elements
robot_dot, = ax.plot([], [], 'ro', label='Robot')
person_dot, = ax.plot([], [], 'go', label='Person')
trace, = ax.plot([], [], 'r--', alpha=0.5)
# Field of view cone (lines + fill)
fov_fill = ax.fill([], [], 'orange', alpha=0.2, label='FOV')[0]
fov_left, = ax.plot([], [], 'orange', linewidth=1.5)
fov_right, = ax.plot([], [], 'orange', linewidth=1.5)
# Simulated person bounding box
bbox_rect = plt.Rectangle((0,0), 0.0, 0.0, fill=False, color='lime', lw=2)
ax.add_patch(bbox_rect)
ax.legend(loc='upper right')

path_x, path_y = [], []

# --- Simulation loop ---
for step in range(10000):
    # Move person in a smooth, wavy motion
    person[0] = 1.5 * math.cos(0.4 * step * dt)
    person[1] = 1.2 * math.sin(0.3 * step * dt)

    dx, dy = person - robot
    dist = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    angle_error = wrap_angle(target_angle - heading)

    # --- Turn control ---
    err_ratio = angle_error / (math.pi / 2)
    if abs(err_ratio) > DEAD_ZONE_RATIO:
        w = TURN_GAIN * err_ratio * TURN_SENSITIVITY
    else:
        w = 0.0
    w = np.clip(w, -1.5, 1.5)

    # --- Forward control ---
    if dist > FOLLOW_DIST_FAR:
        v = FORWARD_SPEED
    elif dist < FOLLOW_DIST_NEAR:
        v = -FORWARD_SPEED * 0.6
    else:
        v = 0.0

    # --- Adaptive braking ---
    if prev_dist is not None:
        dist_rate = (prev_dist - dist) / dt
        if dist_rate > 0.5:
            v *= 0.5
    prev_dist = dist

    # --- Motion update ---
    heading += w * dt
    robot += v * dt * np.array([math.cos(heading), math.sin(heading)])

    # --- Visualization ---
    path_x.append(robot[0])
    path_y.append(robot[1])
    robot_dot.set_data([robot[0]], [robot[1]])
    person_dot.set_data([person[0]], [person[1]])
    trace.set_data(path_x, path_y)

    # Draw FOV cone
    fov_half = math.radians(FOV_DEG / 2)
    cone_range = 1.0
    left_vec = np.array([
        cone_range * math.cos(heading + fov_half),
        cone_range * math.sin(heading + fov_half)
    ])
    right_vec = np.array([
        cone_range * math.cos(heading - fov_half),
        cone_range * math.sin(heading - fov_half)
    ])
    fov_fill.set_xy(np.array([
        [robot[0], robot[1]],
        robot + left_vec,
        robot + right_vec
    ]))
    fov_left.set_data([robot[0], robot[0] + left_vec[0]],
                      [robot[1], robot[1] + left_vec[1]])
    fov_right.set_data([robot[0], robot[0] + right_vec[0]],
                       [robot[1], robot[1] + right_vec[1]])

    # Draw bounding box around person (simulated detection)
    box_w, box_h = 0.3, 0.5
    bbox_rect.set_bounds(person[0] - box_w/2, person[1] - box_h/2, box_w, box_h)

    ax.set_title(f"Step {step} | Dist: {dist:.2f} m | Heading: {math.degrees(heading):.1f}° | Turn: {w:.2f}")
    plt.pause(dt)

plt.ioff()
plt.show()
