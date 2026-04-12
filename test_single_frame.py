import cv2
import numpy as np
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
MODEL_PATH = r"D:/GraduationProjectCode/kaggle/download/trained_weights/wheelchock4_lab_grass_best.pt"  # 确保是 YOLOv8-seg 模型
RGB_PATH = r"extracted_data2/rgb/frame_000040.png"
DEPTH_PATH = r"extracted_data2/depth/frame_000040.png"

# 相机内参（从 camera_intrinsics.txt 读取）
fx = 913.0030517578125
fy = 911.9224243164062
cx = 635.4708251953125
cy = 377.90032958984375
depth_scale = 0.0010000000474974513  # 深度图单位：毫米 → 米

# ==================== 加载模型 ====================
model = YOLO(MODEL_PATH)

# ==================== 读取图像 ====================
rgb = cv2.imread(RGB_PATH)
depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)  # 16位，单位毫米

if rgb is None:
    raise FileNotFoundError(f"RGB 图像未找到: {RGB_PATH}")
if depth_raw is None:
    raise FileNotFoundError(f"深度图像未找到: {DEPTH_PATH}")

# 确保深度图是二维 (H, W)，如果存在单维度则移除
if depth_raw.ndim == 3:
    depth_raw = depth_raw.squeeze()

print(f"RGB 尺寸: {rgb.shape}")
print(f"深度图尺寸: {depth_raw.shape}, 数据类型: {depth_raw.dtype}")

# ==================== YOLOv8-seg 推理 ====================
results = model(rgb)

if results[0].masks is None:
    print("未检测到任何目标！")
    exit()

# 取第一个检测到的目标
mask = results[0].masks.data[0].cpu().numpy()   # 浮点型，范围 0~1
mask = (mask > 0.5).astype(np.uint8)            # 二值化

print("目标检测成功，正在计算位姿...")

# ==================== 提取目标区域的深度值 ====================
ys, xs = np.where(mask == 1)
if len(xs) == 0:
    print("掩码为空！")
    exit()

depths_mm = depth_raw[ys, xs]                   # 一维数组，单位毫米
valid = depths_mm > 0
xs = xs[valid]
ys = ys[valid]
depths_mm = depths_mm[valid]

if len(xs) == 0:
    print("有效深度点为空！")
    exit()

# 转换为米
z = depths_mm * depth_scale
x = (xs - cx) * z / fx
y = (ys - cy) * z / fy
points = np.stack((x, y, z), axis=-1)   # (N, 3)

print(f"有效点云点数: {len(points)}")

# ==================== 计算质心（位置） ====================
centroid = np.mean(points, axis=0)
print(f"三维位置: {centroid}")


# ==================== PCA 姿态估计 ====================
centered = points - centroid
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
axes = Vt.T  # 3x3，列向量为主方向

# 构造旋转矩阵（从物体坐标系到相机坐标系）
rot_mat = axes
# 确保旋转矩阵是正交且行列式为 +1（避免反射）
if np.linalg.det(rot_mat) < 0:
    rot_mat[:, 2] = -rot_mat[:, 2]   # 翻转第三轴

# 将旋转矩阵转换为欧拉角（ZYX 顺序，即 yaw, pitch, roll）
from scipy.spatial.transform import Rotation as R
r = R.from_matrix(rot_mat)
yaw, pitch, roll = r.as_euler('zyx')   # 弧度
roll_deg = np.degrees(roll)
pitch_deg = np.degrees(pitch)
yaw_deg = np.degrees(yaw)

print("姿态:")
print(f"Roll : {roll_deg:.2f}°")  #  (绕X轴)
print(f"Pitch: {pitch_deg:.2f}°")   #  (绕Y轴)
print(f"Yaw  : {yaw_deg:.2f}°")   #  (绕Z轴)

# ==================== 可选：保存点云（用于可视化） ====================
# try:当前base环境的python版本不支持open3d，换用下面的matplotlib
#     import open3d as o3d
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.visualization.draw_geometries([pcd])
# except ImportError:
#     print("未安装 open3d，跳过点云可视化。")
# 点云可视化方式2：matplotlib
# def visualize_points(points):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points[:,0], points[:,1], points[:,2], s=2, c='b')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()
#
# # 直接调用
# visualize_points(points)


# ==================== 可视化（用于论文） ====================
# 将 mask 缩放到原始图像尺寸 (720, 1280)
mask_original = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
mask_original = (mask_original > 0.5).astype(np.uint8)   # 确保二值

# 创建输出文件夹
vis_dir = "visualization"
os.makedirs(vis_dir, exist_ok=True)

# 1. RGB + 掩码叠加
mask_overlay = rgb.copy()
color_mask = np.zeros_like(rgb)
color_mask[:, :, 2] = (mask_original * 255).astype(np.uint8)   # 红色通道
mask_overlay = cv2.addWeighted(rgb, 0.6, color_mask, 0.4, 0)
contours, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_overlay, contours, -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(vis_dir, "1_rgb_mask.png"), mask_overlay)

# 2. 深度伪彩色图
depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
cv2.drawContours(depth_color, contours, -1, (255, 255, 255), 2)
cv2.imwrite(os.path.join(vis_dir, "2_depth_colored.png"), depth_color)

# 3. 点云可视化（采样后）
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sample_step = max(1, len(points) // 3000)
points_sample = points[::sample_step]
sc = ax.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                c=points_sample[:, 2], cmap='viridis', s=1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.colorbar(sc, ax=ax, label='Depth (m)')
plt.title('Point Cloud of Wheel Chock')
plt.savefig(os.path.join(vis_dir, "3_pointcloud.png"), dpi=300)
plt.close()

# 4. 点云 + 质心 + 主轴方向
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
           c='gray', s=1, alpha=0.5)
ax.scatter(centroid[0], centroid[1], centroid[2], c='red', s=50, marker='o', label='Centroid')
axis_len = 0.1
for i, color in enumerate(['r', 'g', 'b']):
    vec = axes[:, i] * axis_len
    ax.quiver(centroid[0], centroid[1], centroid[2], vec[0], vec[1], vec[2],
              color=color, arrow_length_ratio=0.1, linewidth=2, label=f'Axis {i+1}')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.title('3D Position and Orientation')
plt.savefig(os.path.join(vis_dir, "4_pose.png"), dpi=300)
plt.close()

print(f"可视化图片已保存到 {vis_dir}/")
