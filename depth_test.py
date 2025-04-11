import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

hfov, vfov = 87.0, 58.0
flatness, final_alt = 0.2, 1.0  # Define thresholds
disparity_to_depth_scale = 0.0010000000474974513  # mm to meters
MAX_DISTANCE = 10.0  # Max detection distance (meters)

rng_alt = 0.0  # Global variable for altitude


class SafeLander(Node):
    def __init__(self):
        super().__init__('safe_lander')
        self.bridge = CvBridge()

        # Subscriptions
        self.subscription = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        self.last_valid_altitude = 2.0  # Initialize with a default value

    def depth_callback(self, msg):
        """Callback for processing depth images and finding safe landing spots."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

        mask = (frame <= (MAX_DISTANCE * 1000)).astype(np.uint16)  # Convert meters to mm
        frame = frame * mask

        height, width = frame.shape

        # Get depth at the center pixel (convert mm to meters)
        center_x, center_y = width // 2, height // 2
        center_depth_mm = frame[center_y, center_x]
        center_depth_m = center_depth_mm * disparity_to_depth_scale

        # Validate depth (ignore 0 or invalid depth values)
        if 0.1 < center_depth_m < MAX_DISTANCE:
            altitude = center_depth_m
            self.last_valid_altitude = altitude  # Update last valid altitude
        else:
            altitude = self.last_valid_altitude  # Use last valid altitude if current is invalid

        self.get_logger().info(f"[Altitude: {altitude:.2f} m.]")

        # Grid partitioning (3x3)
        grid_rows, grid_cols = 3, 3
        third_width, third_height = width // grid_cols, height // grid_rows

        segments = [
            frame[i * third_height:(i + 1) * third_height, j * third_width:(j + 1) * third_width]
            for i in range(grid_rows)
            for j in range(grid_cols)
        ]

        min_diffs = [np.max(seg) - np.min(seg) for seg in segments]
        differences_in_meters = [diff * disparity_to_depth_scale for diff in min_diffs]

        min_diff = min(differences_in_meters)
        min_diff_index = differences_in_meters.index(min_diff)

        # Normalize depth for visualization
        depth_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Draw grid lines
        for i in range(1, grid_cols):
            cv2.line(frame_colored, (i * third_width, 0), (i * third_width, height), (255, 255, 255), 2)
        for i in range(1, grid_rows):
            cv2.line(frame_colored, (0, i * third_height), (width, i * third_height), (255, 255, 255), 2)

        # Annotate each segment
        for i, diff in enumerate(differences_in_meters):
            x, y = (i % grid_cols) * third_width, (i // grid_cols) * third_height
            cv2.putText(frame_colored, f"{diff:.2f}m", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if min_diff <= flatness and altitude >= final_alt:
            top_left_x = (min_diff_index % 3) * third_width
            top_left_y = (min_diff_index // 3) * third_height
            bottom_right_x = top_left_x + third_width
            bottom_right_y = top_left_y + third_height

            x_avg = (top_left_x + bottom_right_x) / 2
            y_avg = (top_left_y + bottom_right_y) / 2

            # Compute landing coordinates
            x_ang = (x_avg - width / 2) * (hfov / width)
            y_ang = ((height / 2) - y_avg) * (vfov / height)
            x_dist = altitude * np.tan(np.radians(x_ang))
            y_dist = altitude * np.tan(np.radians(y_ang))

            self.get_logger().info(f"[Landing at x: {x_dist:.2f} m., y: {y_dist:.2f} m.]")

            if altitude <= final_alt:
                self.get_logger().info("Landing Final Altitude Reached")
                self.destroy_node()

            cv2.rectangle(frame_colored, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)

        cv2.imshow("Depth Grid", frame_colored)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SafeLander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
