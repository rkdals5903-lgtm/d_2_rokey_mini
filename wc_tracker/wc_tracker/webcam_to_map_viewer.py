import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
import numpy as np
import yaml


class WebcamToMapViewer(Node):
    def __init__(self):
        super().__init__('webcam_to_map_viewer')

        self.sub = self.create_subscription(Point, 'car_position_webcam', self.position_callback, 10)
        self.pub = self.create_publisher(Point, 'car_position_map_pixel', 10)

        self.map_yaml_path = '/home/sangcheol/rokey_ws/src/wc_tracker/maps/real_map_2.yaml'
        with open(self.map_yaml_path, 'r') as f:
            yaml.safe_load(f)

        map_image_path = self.map_yaml_path.replace('.yaml', '.pgm')
        gray = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f'Failed to load map image: {map_image_path}')

        self.map_img_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.map_h, self.map_w = self.map_img_base.shape[:2]

        self.src_points = np.float32([
        [278.5, 82.0],    # 좌상
        [502.5, 136.0],   # 우상
        [526.5, 343.5],   # 우하
        [140.0, 329.5],   # 좌하
        ])

        self.dst_points = np.float32([
            [27.0, 33.0],
            [46.0, 64.0],
            [30.0, 93.0],
            [8.0, 87.0],
        ])

        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        self.latest_point = None

        cv2.namedWindow('Car Position on Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Car Position on Map', 600, 600)

        self.timer = self.create_timer(0.05, self.update_display)

        self.get_logger().info('webcam_to_map_viewer started.')

    def position_callback(self, msg):
        image_x = msg.x
        image_y = msg.y

        src = np.array([[[image_x, image_y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H)

        map_px = float(dst[0][0][0])
        map_py = float(dst[0][0][1])

        self.latest_point = (map_px, map_py)

        out = Point()
        out.x = map_px
        out.y = map_py
        out.z = 0.0
        self.pub.publish(out)

        self.get_logger().info(
            f'image=({image_x:.1f}, {image_y:.1f}) -> map_pixel=({map_px:.1f}, {map_py:.1f})'
        )

    def update_display(self):
        display = self.map_img_base.copy()

        if self.latest_point is None:
            cv2.putText(
                display,
                'No detection yet',
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        else:
            map_px, map_py = self.latest_point
            px = int(map_px)
            py = int(map_py)

            if 0 <= px < self.map_w and 0 <= py < self.map_h:
                cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f'({map_px:.1f}, {map_py:.1f})',
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1
                )
            else:
                cv2.putText(
                    display,
                    f'Outside map: ({map_px:.1f}, {map_py:.1f})',
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

        cv2.imshow('Car Position on Map', display)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = WebcamToMapViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()