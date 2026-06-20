import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
import cv2


class WebcamToMapNode(Node):
    def __init__(self):
        super().__init__('webcam_to_map_node')

        self.sub = self.create_subscription(
            Point,
            'car_position_webcam',
            self.position_callback,
            10
        )

        self.pub = self.create_publisher(
            Point,
            'car_position_map',
            10
        )

        # 변환 전: 704x704 이미지의 네 꼭짓점
        self.src_points = np.float32([
            [0.0, 0.0],       # 좌상
            [704.0, 0.0],     # 우상
            [704.0, 704.0],   # 우하
            [0.0, 704.0],     # 좌하
        ])

        # 변환 후: map 좌표계 네 점
        # 일단 real_map_2.yaml 기준 전체 맵 영역으로 설정
        # 나중에 실험하면서 이 값만 수정하면 됨
        dst_points = np.float32([
        [27.0, 33.0],  # 맵 이미지 픽셀 좌표
        [55.0, 42.0],
        [47.0, 65.0],
        [19.0, 54.0],
        ])
        
        self.H = cv2.getPerspectiveTransform(
            self.src_points,
            self.dst_points
        )

        self.get_logger().info('webcam_to_map_node started.')

    def position_callback(self, msg):
        image_x = msg.x
        image_y = msg.y

        src = np.array([[[image_x, image_y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H)

        map_x = float(dst[0][0][0])
        map_y = float(dst[0][0][1])

        out = Point()
        out.x = map_x
        out.y = map_y
        out.z = 0.0

        self.pub.publish(out)

        self.get_logger().info(
            f'image=({image_x:.1f}, {image_y:.1f}) -> map=({map_x:.2f}, {map_y:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = WebcamToMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()