import pyrealsense2 as rs

bag_path = r"D:\GraduationProjectCode\dataset\real_data\wheelchock.bag"  # 修改为你的实际路径

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bag_path)
profile = pipeline.start(config)

# 获取设备信息
device = profile.get_device()
playback = device.as_playback()
print("Playback device created")

# 获取所有传感器
for sensor in device.sensors:
    print(f"Sensor: {sensor.get_info(rs.camera_info.name)}")
    for stream_profile in sensor.get_stream_profiles():
        stream = stream_profile.stream_type()
        format = stream_profile.format()
        fps = stream_profile.fps()
        print(f"  Stream: {stream}, Format: {format}, FPS: {fps}")

pipeline.stop()