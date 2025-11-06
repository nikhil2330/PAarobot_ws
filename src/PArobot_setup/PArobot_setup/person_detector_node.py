#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import os, time, importlib.util

def try_load_tflite():
    try:
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter, None
        # fallback to TF’s bundled tflite
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter, None
    except Exception as e:
        return None, e

class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')
        self.declare_parameter('model_dir', './')
        self.declare_parameter('graph', 'detect.tflite')
        self.declare_parameter('labels', 'labelmap.txt')
        self.declare_parameter('score_thresh', 0.59)

        self.model_dir = self.get_parameter('model_dir').value
        self.graph = self.get_parameter('graph').value
        self.labels_path = os.path.join(self.model_dir, self.get_parameter('labels').value)
        self.score_thresh = float(self.get_parameter('score_thresh').value)

        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)

        # Publishes: center_x (pixels), bbox [xmin,ymin,xmax,ymax,score]
        self.center_pub = self.create_publisher(Float32, '/person/center_x', 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, '/person/bbox', 10)

        # Load labels if exist
        self.labels = []
        if os.path.exists(self.labels_path):
            with open(self.labels_path, 'r') as f:
                self.labels = [l.strip() for l in f.readlines()]
            if self.labels and self.labels[0] == '???':
                self.labels = self.labels[1:]

        # Try TFLite
        Interpreter, err = try_load_tflite()
        self.using_tflite = False
        self.interpreter = None

        if Interpreter is not None:
            try:
                model_path = os.path.join(self.model_dir, self.graph)
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.h = self.input_details[0]['shape'][1]
                self.w = self.input_details[0]['shape'][2]
                self.float_input = (self.input_details[0]['dtype'] == np.float32)
                self.input_mean, self.input_std = 127.5, 127.5
                self.using_tflite = True
                self.get_logger().info(f"✅ Using TFLite model: {model_path}")
            except Exception as e:
                self.get_logger().warn(f"TFLite load failed ({e}). Falling back to OpenCV HOG person detector.")
        else:
            self.get_logger().warn(f"No TFLite interpreter available ({err}). Falling back to OpenCV HOG.")

        # OpenCV HOG person detector fallback (works right now without extra installs)
        if not self.using_tflite:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.get_logger().info("👟 OpenCV HOG person detector ready (fallback).")

        self.last_bbox = None
        self.last_seen_time = 0.0
        self.lost_timeout = 1.0

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        imH, imW = frame.shape[:2]

        xmin = ymin = xmax = ymax = None
        score = 0.0
        found = False

        if self.using_tflite:
            # Preprocess to RGB and model size
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(rgb, (self.w, self.h))
            inp = np.expand_dims(inp, axis=0)
            if self.float_input:
                inp = (np.float32(inp) - self.input_mean) / self.input_std

            self.interpreter.set_tensor(self.input_details[0]['index'], inp)
            self.interpreter.invoke()

            # TF1 vs TF2 output ordering handling
            outname = self.output_details[0]['name']
            if 'StatefulPartitionedCall' in outname:
                boxes_idx, classes_idx, scores_idx = 1, 3, 0
            else:
                boxes_idx, classes_idx, scores_idx = 0, 1, 2

            boxes = self.interpreter.get_tensor(self.output_details[boxes_idx]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[classes_idx]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[scores_idx]['index'])[0]

            # find person with score > threshold
            for i in range(len(scores)):
                if self.score_thresh < scores[i] <= 1.0:
                    cls_name = self.labels[int(classes[i])] if self.labels else 'person'
                    if cls_name == 'person':
                        y1 = int(max(0, boxes[i][0] * imH))
                        x1 = int(max(0, boxes[i][1] * imW))
                        y2 = int(min(imH, boxes[i][2] * imH))
                        x2 = int(min(imW, boxes[i][3] * imW))
                        xmin, ymin, xmax, ymax = x1, y1, x2, y2
                        score = float(scores[i])
                        found = True
                        break

        else:
            # OpenCV HOG fallback
            # returns list of rectangles (x,y,w,h)
            rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
            if len(rects) > 0:
                # pick the largest (as a proxy for closest)
                idx = np.argmax([w*h for (x,y,w,h) in rects])
                x, y, w, h = rects[idx]
                xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)
                score = float(weights[idx]) if len(weights) > idx else 0.0
                found = True

        if not found and self.last_bbox and (time.time() - self.last_seen_time) < self.lost_timeout:
            xmin, ymin, xmax, ymax = self.last_bbox
            found = True  # short-term memory

        if found and xmin is not None:
            cx = (xmin + xmax) / 2.0
            # publish center_x
            self.center_pub.publish(Float32(data=float(cx)))

            # publish bbox
            arr = Float32MultiArray()
            arr.data = [float(xmin), float(ymin), float(xmax), float(ymax), float(score)]
            self.bbox_pub.publish(arr)

            self.last_bbox = (xmin, ymin, xmax, ymax)
            self.last_seen_time = time.time()
        else:
            # publish a sentinel (e.g., -1) so follower knows target is lost
            self.center_pub.publish(Float32(data=-1.0))
            self.bbox_pub.publish(Float32MultiArray(data=[]))


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
