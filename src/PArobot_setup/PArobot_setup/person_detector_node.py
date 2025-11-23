#!/usr/bin/env python3
import os
import time
import importlib.util

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, Bool
from cv_bridge import CvBridge

import numpy as np
import cv2

# ---- CONSTANTS (from your script) ----
DETECTION_THRESHOLD   = 0.59
LOCK_VISIBLE_TIME_SEC = 2.0
MIN_HIST_FRAMES       = 10
COLOR_SIM_THRESH      = 0.5
HIST_BINS             = 16


class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__("person_detector_node")

        # ------------------- PARAMETERS -------------------
        self.declare_parameter("model_dir", "./")
        self.declare_parameter("graph", "detect.tflite")
        self.declare_parameter("labels", "labelmap.txt")
        self.declare_parameter("score_thresh", DETECTION_THRESHOLD)

        self.model_dir = self.get_parameter("model_dir").value
        self.graph = self.get_parameter("graph").value
        labels_file = self.get_parameter("labels").value
        self.labels_path = os.path.join(self.model_dir, labels_file)
        self.score_thresh = float(self.get_parameter("score_thresh").value)

        # ------------------- ROS I/O -------------------
        self.bridge = CvBridge()
        self.create_subscription(Image, "/camera/image_raw", self.image_cb, 10)
        self.create_subscription(Bool, "/robot_enabled", self.rf_cb, 10)

        self.center_pub = self.create_publisher(Float32, "/person/center_x", 10)
        self.bbox_pub   = self.create_publisher(Float32MultiArray, "/person/bbox", 10)
        self.debug_img_pub = self.create_publisher(Image, "/person/debug_image", 10)

        # ------------------- LABELS -------------------
        self.labels = []
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                self.labels = [l.strip() for l in f.readlines()]
            if self.labels and self.labels[0] == "???":
                self.labels = self.labels[1:]
        else:
            self.get_logger().warn(f"Label file not found: {self.labels_path}")

        # ------------------- LOAD TFLITE -------------------
        try:
            pkg = importlib.util.find_spec("tflite_runtime")
            if pkg is not None:
                from tflite_runtime.interpreter import Interpreter
                self.get_logger().info("Using tflite_runtime.Interpreter")
            else:
                from tensorflow.lite.python.interpreter import Interpreter
                self.get_logger().info("Using tensorflow.lite Interpreter")

            model_path = os.path.join(self.model_dir, self.graph)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"TFLite model not found at: {model_path}")

            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.h = self.input_details[0]["shape"][1]
            self.w = self.input_details[0]["shape"][2]
            self.float_input = (self.input_details[0]["dtype"] == np.float32)
            self.input_mean, self.input_std = 127.5, 127.5

            outname = self.output_details[0]["name"]
            if "StatefulPartitionedCall" in outname:
                self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
            else:
                self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

            self.get_logger().info(f"✅ TFLite model loaded: {model_path}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load TFLite model: {e}")
            rclpy.shutdown()
            raise

        # ------------------- RF STATE -------------------
        self.robot_enabled = False

        # ------------------- LOCK STATE (as in script) -------------------
        self.candidate_active          = False
        self.candidate_bbox            = None
        self.candidate_hist_accum      = None
        self.candidate_hist_count      = 0
        self.candidate_visible_time    = 0.0
        self.candidate_visible_start_t = None

        self.tracked_active      = False
        self.tracked_bbox        = None
        self.tracked_hist        = None
        self.tracked_score       = 0.0
        self.tracked_lost_frames = 0
        self.max_tracked_lost_frames = 30

    # ------------------- RF (reset on START) -------------------
    def rf_cb(self, msg: Bool):
        prev = self.robot_enabled
        self.robot_enabled = msg.data
        if self.robot_enabled and not prev:
            # mimic your "new start" reset of candidate + tracking
            self.candidate_active          = False
            self.candidate_bbox            = None
            self.candidate_hist_accum      = None
            self.candidate_hist_count      = 0
            self.candidate_visible_time    = 0.0
            self.candidate_visible_start_t = None

            self.tracked_active      = False
            self.tracked_bbox        = None
            self.tracked_hist        = None
            self.tracked_score       = 0.0
            self.tracked_lost_frames = 0
            self.get_logger().info("[RF] New Start → resetting lock state")

    # ------------------- HISTOGRAM HELPERS -------------------
    def compute_color_hist(self, frame_rgb, bbox):
        xmin, ymin, xmax, ymax = bbox
        h, w, _ = frame_rgb.shape

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w - 1, xmax)
        ymax = min(h - 1, ymax)

        if xmax <= xmin or ymax <= ymin:
            return None

        bw = xmax - xmin
        bh = ymax - ymin
        if bw < 10 or bh < 10:
            return None

        inner_xmin = int(xmin + 0.15 * bw)
        inner_xmax = int(xmax - 0.15 * bw)
        inner_ymin = int(ymin + 0.20 * bh)
        inner_ymax = int(ymax - 0.05 * bh)

        inner_xmin = max(xmin, inner_xmin)
        inner_ymin = max(ymin, inner_ymin)
        inner_xmax = min(xmax, inner_xmax)
        inner_ymax = min(ymax, inner_ymax)

        if inner_xmax <= inner_xmin or inner_ymax <= inner_ymin:
            return None

        roi = frame_rgb[inner_ymin:inner_ymax, inner_xmin:inner_xmax]
        if roi.size == 0:
            return None

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(roi_blur, cv2.COLOR_RGB2HSV)

        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [HIST_BINS, HIST_BINS],
            [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.astype(np.float32)

    @staticmethod
    def hist_correlation(hist1, hist2):
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    # ------------------- TFLITE + LOCK + PUBLISH -------------------
    def image_cb(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imH, imW = frame_bgr.shape[:2]
        now = time.time()

        # 1) TFLite inference → list of people
        people = []
        try:
            inp = cv2.resize(frame_rgb, (self.w, self.h))
            inp = np.expand_dims(inp, axis=0)
            if self.float_input:
                inp = (np.float32(inp) - self.input_mean) / self.input_std

            self.interpreter.set_tensor(self.input_details[0]["index"], inp)
            self.interpreter.invoke()

            boxes   = self.interpreter.get_tensor(self.output_details[self.boxes_idx]["index"])[0]
            classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]["index"])[0]
            scores  = self.interpreter.get_tensor(self.output_details[self.scores_idx]["index"])[0]

            for i, s in enumerate(scores):
                if self.score_thresh < s <= 1.0:
                    cls_id = int(classes[i])
                    cls_name = self.labels[cls_id] if (self.labels and cls_id < len(self.labels)) else "person"
                    if cls_name == "person":
                        ymin = int(max(0,   boxes[i][0] * imH))
                        xmin = int(max(0,   boxes[i][1] * imW))
                        ymax = int(min(imH, boxes[i][2] * imH))
                        xmax = int(min(imW, boxes[i][3] * imW))
                        people.append({"bbox": (xmin, ymin, xmax, ymax), "score": float(s)})
        except Exception as e:
            self.get_logger().error(f"TFLite inference error: {e}")

        for det in people:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)

        state_str = "NO_PERSON"
        candidate_visible_this_frame = False

        # 2) Candidate / Lock logic (EXACT same as your script)
        if not self.tracked_active:
            if not self.candidate_active:
                if people:
                    best = max(people, key=lambda d: d["score"])
                    self.candidate_bbox = best["bbox"]
                    self.candidate_hist_accum = None
                    self.candidate_hist_count = 0
                    self.candidate_visible_time = 0.0
                    self.candidate_visible_start_t = None
                    self.candidate_active = True
                    state_str = "CANDIDATE_STARTED"
                else:
                    state_str = "WAITING_FOR_PERSON"
            else:
                if people:
                    best = max(people, key=lambda d: d["score"])
                    self.candidate_bbox = best["bbox"]

                    hist = self.compute_color_hist(frame_rgb, self.candidate_bbox)
                    if hist is not None:
                        if self.candidate_hist_accum is None:
                            self.candidate_hist_accum = hist.copy()
                            self.candidate_hist_count = 1
                        else:
                            self.candidate_hist_accum += hist
                            self.candidate_hist_count += 1

                    candidate_visible_this_frame = True
                    state_str = "CANDIDATE_VISIBLE"
                else:
                    candidate_visible_this_frame = False
                    self.candidate_visible_start_t = None
                    self.candidate_visible_time = 0.0
                    state_str = "CANDIDATE_NOT_VISIBLE"

                if candidate_visible_this_frame:
                    if self.candidate_visible_start_t is None:
                        self.candidate_visible_start_t = now
                        self.candidate_visible_time = 0.0
                    else:
                        self.candidate_visible_time = now - self.candidate_visible_start_t

                if self.candidate_bbox is not None:
                    x1, y1, x2, y2 = self.candidate_bbox
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if (
                    candidate_visible_this_frame
                    and self.candidate_hist_accum is not None
                    and self.candidate_hist_count >= MIN_HIST_FRAMES
                    and self.candidate_visible_time >= LOCK_VISIBLE_TIME_SEC
                ):
                    avg_hist = self.candidate_hist_accum / max(1, self.candidate_hist_count)
                    cv2.normalize(avg_hist, avg_hist)
                    self.tracked_hist = avg_hist.astype(np.float32)
                    self.tracked_bbox = self.candidate_bbox
                    self.tracked_score = 0.0
                    self.tracked_active = True
                    self.tracked_lost_frames = 0
                    state_str = "LOCKED"
        else:
            best_match = None
            best_sim = None

            if people and self.tracked_hist is not None:
                for det in people:
                    bbox = det["bbox"]
                    hist = self.compute_color_hist(frame_rgb, bbox)
                    if hist is None:
                        continue
                    color_sim = self.hist_correlation(self.tracked_hist, hist)
                    if color_sim < COLOR_SIM_THRESH:
                        continue
                    if (best_sim is None) or (color_sim > best_sim):
                        best_sim = color_sim
                        best_match = det

            if best_match is not None:
                self.tracked_bbox = best_match["bbox"]
                self.tracked_score = best_match["score"]
                self.tracked_lost_frames = 0
                state_str = f"LOCKED (sim={best_sim:.2f})"
            else:
                self.tracked_lost_frames += 1
                state_str = f"LOCKED_LOST ({self.tracked_lost_frames})"
                if self.tracked_lost_frames > self.max_tracked_lost_frames:
                    self.get_logger().warn("Tracked person lost – resetting lock.")
                    self.tracked_active = False
                    self.tracked_bbox = None
                    self.tracked_hist = None
                    self.candidate_active = False
                    self.candidate_bbox = None

            if self.tracked_bbox is not None:
                x1, y1, x2, y2 = self.tracked_bbox
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"LOCKED score={self.tracked_score:.2f}"
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        # 3) Publish center_x and bbox ONLY when lock is active
        if self.tracked_active and self.tracked_bbox is not None:
            xmin, ymin, xmax, ymax = self.tracked_bbox
            cx = (xmin + xmax) / 2.0

            self.center_pub.publish(Float32(data=float(cx)))
            arr = Float32MultiArray()
            arr.data = [float(xmin), float(ymin), float(xmax), float(ymax), float(self.tracked_score)]
            self.bbox_pub.publish(arr)
        else:
            self.center_pub.publish(Float32(data=-1.0))
            self.bbox_pub.publish(Float32MultiArray(data=[]))

        cv2.putText(
            frame_bgr,
            f"STATE: {state_str}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        dbg_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        self.debug_img_pub.publish(dbg_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
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
