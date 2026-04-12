import cv2
import numpy as np

rgb = cv2.imread("extracted_data/rgb/frame_000000.png")
depth = cv2.imread("extracted_data/depth/frame_000000.png", cv2.IMREAD_UNCHANGED)
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

# 叠加半透明
blended = cv2.addWeighted(rgb, 0.5, depth_color, 0.5, 0)
cv2.imshow("Overlay", blended)
cv2.waitKey(0)