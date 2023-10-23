# -*- coding: utf-8 -*-
import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头，可以尝试不同的索引

# 初始化KNN背景建模器
bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用KNN背景建模器进行前景-背景分割
    fg_mask = bg_subtractor.apply(frame)

    # 去除阴影部分（可选）
    fg_mask[fg_mask == 127] = 0

    # 显示前景分割结果
    cv2.imshow('Foreground Mask', fg_mask)

    # 显示原始帧
    cv2.imshow('Original Frame', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# 释放摄像头资源和关闭窗口
cap.release()
cv2.destroyAllWindows()
