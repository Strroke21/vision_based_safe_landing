#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import math

hfov, vfov = np.radians(87.0), np.radians(58.0)
flatness, final_alt = 0.4, 0.5
disparity_to_depth_scale = 0.0010000000474974513
MAX_DISTANCE = 10.0


def compute_angle_scale_5x5(altitude, hfov, vfov,
                           base_gain=0.8,
                           min_scale=0.15,
                           max_scale=0.9,
                           smooth_factor=0.75,
                           prev_scale=[0.5]):

    if altitude <= 0:
        return min_scale

    width_m = 2 * altitude * math.tan(hfov / 2)
    height_m = 2 * altitude * math.tan(vfov / 2)

    cell_w = width_m / 5.0
    cell_h = height_m / 5.0
    avg_cell = 0.5 * (cell_w + cell_h)

    scale = base_gain * avg_cell
    scale = scale / (1.0 + scale)

    scale = max(min_scale, min(scale, max_scale))

    scale = smooth_factor * prev_scale[0] + (1 - smooth_factor) * scale
    prev_scale[0] = scale

    return scale

class SafeLander(Node):
    def __init__(self):
        super().__init__('safe_lander')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )

        self.last_valid_altitude = 2.0

        # Persistent state
        self.current_cell = None
        self.current_diff = flatness + 1.0
        self.bad_frame_count = 0
        self.hyst_threshold = 30

    def depth_callback(self, msg):
        st = time.time()

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        mask = (frame <= (MAX_DISTANCE * 1000)).astype(np.uint16)
        frame = frame * mask

        height, width = frame.shape

        # ---- ALTITUDE ----
        cx, cy = width // 2, height // 2
        center_depth_m = frame[cy, cx] * disparity_to_depth_scale

        if 0.1 < center_depth_m < MAX_DISTANCE:
            altitude = center_depth_m
            self.last_valid_altitude = altitude
        else:
            altitude = self.last_valid_altitude

        self.get_logger().info(f"[Altitude: {altitude:.2f} m]")

        # ---- GRID ----
        grid_rows, grid_cols = 5, 5
        cell_w, cell_h = width // grid_cols, height // grid_rows

        differences = []
        valid_indices = []

        for i in range(grid_rows):
            for j in range(grid_cols):

                seg = frame[
                    i * cell_h:(i + 1) * cell_h,
                    j * cell_w:(j + 1) * cell_w
                ]

                seg = seg[seg > 0]

                if seg.size == 0:
                    scaled_diff = flatness + 1.0
                else:
                    diff = np.max(seg) - np.min(seg)
                    scaled_diff = diff * disparity_to_depth_scale

                # Central priority
                if 1 <= i <= 3 and 1 <= j <= 3:
                    weight = 1.0
                else:
                    weight = 3.0

                differences.append(scaled_diff * weight)
                valid_indices.append((i, j))

        # ---- BEST CELL ----
        min_diff = min(differences)
        best_idx = differences.index(min_diff)
        best_cell = valid_indices[best_idx]

        # ---- CENTRAL CELL ----
        central_cell = (2, 2)
        if central_cell in valid_indices:
            central_idx = valid_indices.index(central_cell)
            central_diff = differences[central_idx]
        else:
            central_diff = flatness + 1.0

        # ---- HYSTERESIS + PRIORITY ----
        if self.current_cell is None:

            if central_diff <= flatness:
                self.current_cell = central_cell
                self.current_diff = central_diff
            else:
                self.current_cell = best_cell
                self.current_diff = min_diff

            self.bad_frame_count = 0

        else:
            # Update current diff
            if self.current_cell in valid_indices:
                idx = valid_indices.index(self.current_cell)
                self.current_diff = differences[idx]
            else:
                self.current_diff = flatness + 1.0

            # CENTRAL PRIORITY LOCK
            if central_diff <= flatness:
                self.current_cell = central_cell
                self.current_diff = central_diff
                self.bad_frame_count = 0

            else:
                # Check if current is bad
                if self.current_diff > flatness:
                    self.bad_frame_count += 1
                else:
                    self.bad_frame_count = 0

                # Switch condition
                if (self.bad_frame_count >= self.hyst_threshold and
                        min_diff < self.current_diff):

                    self.current_cell = best_cell
                    self.current_diff = min_diff
                    self.bad_frame_count = 0

        grid_i, grid_j = self.current_cell

        # ---- VISUALIZATION ----
        depth_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        for i in range(1, grid_cols):
            cv2.line(frame_colored, (i * cell_w, 0), (i * cell_w, height), (255, 255, 255), 1)
        for i in range(1, grid_rows):
            cv2.line(frame_colored, (0, i * cell_h), (width, i * cell_h), (255, 255, 255), 1)

        for idx, diff in enumerate(differences):
            i, j = valid_indices[idx]
            x, y = j * cell_w, i * cell_h
            cv2.putText(frame_colored, f"{diff:.2f}",
                        (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

        # ---- LANDING ----
        if self.current_diff <= flatness and altitude >= final_alt:

            scale_factor = compute_angle_scale_5x5(altitude, hfov, vfov)

            top_left_x = grid_j * cell_w
            top_left_y = grid_i * cell_h
            bottom_right_x = top_left_x + cell_w
            bottom_right_y = top_left_y + cell_h

            x_avg = (top_left_x + bottom_right_x) / 2
            y_avg = (top_left_y + bottom_right_y) / 2

            x_ang = (x_avg - width / 2) * (hfov / width)
            y_ang = (y_avg - height / 2) * (vfov / height)

            x_dist = altitude * np.tan(-y_ang)
            y_dist = altitude * np.tan(x_ang)

            self.get_logger().info(f"[Landing → x: {x_dist:.2f}m, y: {y_dist:.2f}m]")
            self.get_logger().info(f"[Cell: ({grid_i}, {grid_j}), Flatness: {self.current_diff:.2f}]")

            if altitude <= final_alt:
                self.get_logger().info("Landing Complete")
                self.destroy_node()

            cv2.rectangle(
                frame_colored,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                (0, 255, 0),
                2
            )

        cv2.imshow("Depth Grid", frame_colored)
        cv2.waitKey(1)

        self.get_logger().info(f"Solver Time: {time.time() - st:.3f}s")


def main(args=None):
    rclpy.init(args=args)
    node = SafeLander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()