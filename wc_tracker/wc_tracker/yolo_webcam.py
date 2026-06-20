import json
import csv
import time
import math
import os
import shutil
import sys
from ultralytics import YOLO
from pathlib import Path
import cv2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


class YOLOWebcamProcessor(Node):
    def __init__(self, model, output_dir):
        super().__init__('yolo_webcam_processor')

        self.model = model
        self.output_dir = output_dir
        self.csv_output = []
        self.confidences = []
        self.max_object_count = 0
        self.classNames = model.names
        self.should_shutdown = False

        # class 0 객체 중심 좌표 publish
        self.car_pub = self.create_publisher(
            Point,
            'car_position_webcam',
            10
        )

    def run(self):
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            self.get_logger().error("Failed to open webcam.")
            return

        while rclpy.ok() and not self.should_shutdown:
            ret, img = cap.read()
            if not ret:
                continue

            results = self.model(img, stream=True)
            object_count = 0
            fontScale = 1

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    label = self.classNames.get(cls, f"class_{cls}")

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        img,
                        f"{label}: {confidence}",
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        (255, 0, 0),
                        2
                    )

                    # 바운딩박스 중심 좌표 계산
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # class 0 객체만 publish
                    if cls == 0:
                        msg = Point()
                        msg.x = cx
                        msg.y = cy
                        msg.z = 0.0
                        self.car_pub.publish(msg)

                        self.get_logger().info(
                            f"Published car center: x={cx:.1f}, y={cy:.1f}"
                        )

                        cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 0), -1)

                    self.confidences.append(confidence)
                    self.csv_output.append([x1, y1, x2, y2, confidence, label, cx, cy])
                    object_count += 1

            self.max_object_count = max(self.max_object_count, object_count)
            cv2.putText(
                img,
                f"Objects_count: {object_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 255, 0),
                1
            )

            if object_count > 0:
                filename = f'output_{int(time.time())}.jpg'
                cv2.imwrite(os.path.join(self.output_dir, filename), img)

            display_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
            cv2.imshow("Detection", display_img)

            key = cv2.waitKey(10)
            if key == ord('q'):
                self.get_logger().info("Shutting down...")
                self.should_shutdown = True

            rclpy.spin_once(self, timeout_sec=0.001)

        cap.release()
        cv2.destroyAllWindows()

    def save_output(self):
        with open(os.path.join(self.output_dir, 'output.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X1', 'Y1', 'X2', 'Y2', 'Confidence', 'Class', 'CenterX', 'CenterY'])
            writer.writerows(self.csv_output)

        with open(os.path.join(self.output_dir, 'output.json'), 'w') as f:
            json.dump(self.csv_output, f)

        with open(os.path.join(self.output_dir, 'statistics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Max Object Count', 'Average Confidence'])
            avg_conf = sum(self.confidences) / len(self.confidences) if self.confidences else 0
            writer.writerow([self.max_object_count, avg_conf])


def main(args=None):
    rclpy.init(args=args)

    model_path = "/home/sangcheol/rokey_ws/src/wc_tracker/models/best.pt"

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        rclpy.shutdown()
        sys.exit(1)

    suffix = Path(model_path).suffix.lower()

    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"Unsupported model format: {suffix}")
        rclpy.shutdown()
        sys.exit(1)

    output_dir = './output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    processor = YOLOWebcamProcessor(model, output_dir)
    processor.run()
    processor.save_output()

    processor.destroy_node()
    rclpy.shutdown()
    print("Shutdown complete.")


if __name__ == '__main__':
    main()
