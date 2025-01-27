#!/usr/bin/env python3

import cv2
import pyrealsense2 as rs
import numpy as np
import time
from dronekit import connect, VehicleMode, APIException, LocationGlobalRelative
import argparse
import math
from pymavlink import mavutil
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

MAX_DISTANCE = 8.0

hfov = 87 * (math.pi/180)
vfov = 58 * (math.pi/180)

###### functions #######

def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect

    if not connection_string:
        connection_string = '/dev/ttyACM0'

    vehicle = connect(connection_string, wait_ready=True)

    return vehicle

def send_land_message(x,y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0,
        0,
        0,)
    vehicle.send_mavlink(msg)
    vehicle.flush()

vehicle = connectMyCopter()
# SETUP PARAMETERS TO ENABLE PRECISION LANDING
vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1  # 1 for companion computer
vehicle.parameters['PLND_EST_TYPE'] = 0  # 0 for raw sensor, 1 for Kalman filter pos estimation
vehicle.parameters['LAND_SPEED'] = 25  # Descent speed of 25cm/s



# Start streaming
pipeline.start(config)

status = vehicle.system_status.state #input("Enter the vehicle status: ")
print("Vehicle State: ",status)

if ('CRITICAL' in status):
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        frame = np.asanyarray(depth_frame.get_data())
         # Mask out values greater than MAX_DISTANCE
        mask = frame * depth_frame.get_units() <= MAX_DISTANCE
        frame = frame * mask

        height, width = frame.shape
        third_height = height // 3
        third_width = width // 3

        # Split the frame into 3x3 grid
        segments = [
            frame[:third_height, :third_width], frame[:third_height, third_width:2*third_width], frame[:third_height, 2*third_width:],
            frame[third_height:2*third_height, :third_width], frame[third_height:2*third_height, third_width:2*third_width], frame[third_height:2*third_height, 2*third_width:],
            frame[2*third_height:, :third_width], frame[2*third_height:, third_width:2*third_width], frame[2*third_height:, 2*third_width:]
        ]

        # Calculate min and max disparity in each segment
        min_diffs = []
        for segment in segments:
            min_disp = np.min(segment)
            max_disp = np.max(segment)
            min_diffs.append(max_disp - min_disp)

        # Convert disparity differences to meters
        disparity_to_depth_scale = depth_frame.get_units()
        differences_in_meters = [diff * disparity_to_depth_scale for diff in min_diffs]

        # Find the segment with the lowest difference
        min_diff = min(differences_in_meters)
        min_diff_index = differences_in_meters.index(min_diff)

        # Apply color map to the original frame for visualization
        frame_colored = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.03), cv2.COLORMAP_JET)

        # Draw grid lines
        cv2.line(frame_colored, (third_width, 0), (third_width, height), (255, 255, 255), 2)
        cv2.line(frame_colored, (2*third_width, 0), (2*third_width, height), (255, 255, 255), 2)
        cv2.line(frame_colored, (0, third_height), (width, third_height), (255, 255, 255), 2)
        cv2.line(frame_colored, (0, 2*third_height), (width, 2*third_height), (255, 255, 255), 2)

        # Annotate each segment with the difference in meters
        for i, diff in enumerate(differences_in_meters):
            top_left_x = (i % 3) * third_width
            top_left_y = (i // 3) * third_height
            cv2.putText(frame_colored, f"{diff:.2f}m", (top_left_x + 5, top_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Check if the lowest difference is less than or equal to 0.2m and draw bounding box
        altitude = vehicle.rangefinder.distance
        if (min_diff <= 0.2) and (altitude>=1):
            top_left_x = (min_diff_index % 3) * third_width
            top_left_y = (min_diff_index // 3) * third_height
            bottom_right_x = top_left_x + third_width
            bottom_right_y = top_left_y + third_height
            cv2.rectangle(frame_colored, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

            top_left_x = (min_diff_index % 3) * third_width
            top_left_y = (min_diff_index // 3) * third_height
            bottom_right_x = top_left_x + third_width
            bottom_right_y = top_left_y + third_height
                
            x_sum = top_left_x + bottom_right_x
            y_sum = top_left_y + bottom_right_y
                
            x_avg = x_sum * 0.5 
            y_avg = y_sum * 0.5
                
            x_ang = ((x_avg - width * 0.5) * (hfov / width)) + 0.2  # 0.2 rad offset in x-direction
            y_ang = ((y_avg - height * 0.5) * (vfov / height)) + 0.2 # 0.2 rad offset in y-direction
                
            if vehicle.mode != 'LAND':
                vehicle.mode = VehicleMode('LAND')
                while vehicle.mode != 'LAND':
                    time.sleep(1)
                print("------------------------")
                print("Vehicle now in LAND mode")
                print("------------------------")
                send_land_message(x_ang, y_ang)
                #break                                 #break the loop if safe place is found
                time.sleep(1/10)
                    
            else:
                send_land_message(x_ang, y_ang)
                time.sleep(1/10)
                pass
                        
            cv2.rectangle(frame_colored, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            print(f"x angle: {x_ang} y angle: {y_ang}")

        else:
            send_land_message(0,0.5) #move to another location this should be respective of direction with no obstacles.


        # Display the frame with grid and bounding box (if any)
        cv2.imshow("Disparity Grid", frame_colored)
        # time.sleep(2)

        if cv2.waitKey(1) == ord('q'):
            break


