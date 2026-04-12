#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python extract_from_bag.py D:\GraduationProjectCode\dataset\real_data\wheelchock.bag -o extracted_data --step 18 原1800帧，步长改成18，总共提取100帧

import pyrealsense2 as rs
import cv2
import numpy as np
import os
import argparse

def extract_bag(bag_path, output_dir, step=1, max_frames=None):
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale (meters per unit): {depth_scale}")

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    print(f"Camera intrinsics:")
    print(f"  fx = {color_intr.fx}, fy = {color_intr.fy}")
    print(f"  cx = {color_intr.ppx}, cy = {color_intr.ppy}")
    print(f"  width = {color_intr.width}, height = {color_intr.height}")

    with open(os.path.join(output_dir, 'camera_intrinsics.txt'), 'w') as f:
        f.write(f"fx {color_intr.fx}\n")
        f.write(f"fy {color_intr.fy}\n")
        f.write(f"cx {color_intr.ppx}\n")
        f.write(f"cy {color_intr.ppy}\n")
        f.write(f"width {color_intr.width}\n")
        f.write(f"height {color_intr.height}\n")
        f.write(f"depth_scale {depth_scale}\n")

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    frame_count = 0
    saved_count = 0
    print("开始提取帧...")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 按步长保存
            if frame_count % step == 0:
                rgb_image = np.asanyarray(color_frame.get_data())
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                depth_image = np.asanyarray(depth_frame.get_data())

                rgb_filename = os.path.join(rgb_dir, f"frame_{saved_count:06d}.png")
                depth_filename = os.path.join(depth_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(rgb_filename, bgr_image)
                cv2.imwrite(depth_filename, depth_image)
                saved_count += 1

                if saved_count % 100 == 0:
                    print(f"已保存 {saved_count} 帧")

                if max_frames is not None and saved_count >= max_frames:
                    print(f"达到最大保存帧数 {max_frames}，停止提取。")
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        pipeline.stop()
        print(f"共读取 {frame_count} 帧，实际保存 {saved_count} 帧到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 RealSense .bag 文件中提取对齐的 RGB 和深度图")
    parser.add_argument("bag_file", help="输入 .bag 文件路径")
    parser.add_argument("-o", "--output", default="extracted_data", help="输出目录")
    parser.add_argument("--step", type=int, default=1, help="每隔step帧保存一帧")
    parser.add_argument("--max_frames", type=int, default=None, help="最多保存的帧数（默认 None 表示不限制）")
    args = parser.parse_args()

    extract_bag(args.bag_file, args.output, step=args.step, max_frames=args.max_frames)