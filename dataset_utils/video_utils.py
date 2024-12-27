import os
import numpy as np
import cv2


BBOX_COLORS = [
    (255, 0, 0),  # right hand
    (0, 255, 0),  # left hand
    (0, 0, 255),  # tool
    (255, 255, 0),  # target object
    (0, 128, 128),
    (255, 0, 255),
]


def imgs2mp4(imgs, video_path, fps=30, W=1920, H=1080):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    for img in imgs:
        assert img.shape == (H, W, 3)
        videoWriter.write(img)
    videoWriter.release()


def mp42imgs(video_path, return_rgb=False, max_cnt=None):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    imgs = []

    suc = cap.isOpened()
    frame_cnt = -1
    while True:
        frame_cnt += 1
        suc, img = cap.read()
        if not suc:
            break
        if return_rgb:
            img = img[:, :, ::-1].astype(np.uint8)  # bgr2rgb
        imgs.append(img)
        if (not max_cnt is None) and (frame_cnt + 1 >= max_cnt):
            break
    cap.release()
    
    return imgs


def overlay_two_imgs(img1, img2):
    overlay = (img1 * 0.5 + img2 * 0.5).astype(np.uint8)
    return overlay


def overlay(img, rgb_img):
    overlay = rgb_img.copy()
    mask_idx = np.where(img.sum(axis=-1) < 255*3)
    overlay[mask_idx[0], mask_idx[1]] = ((rgb_img[mask_idx[0], mask_idx[1]] * 0.15) + (img[mask_idx[0], mask_idx[1]] * 0.85)).astype(np.uint8)

    return overlay


def overlay_bboxes(img, bbox_list):
    overlay = img.copy()

    for bbox, color in zip(bbox_list, BBOX_COLORS):
        row_min, row_max, col_min, col_max = bbox
        cv2.line(overlay, (col_min, row_min), (col_max, row_min), color, thickness=1)
        cv2.line(overlay, (col_max, row_min), (col_max, row_max), color, thickness=1)
        cv2.line(overlay, (col_max, row_max), (col_min, row_max), color, thickness=1)
        cv2.line(overlay, (col_min, row_max), (col_min, row_min), color, thickness=1)

    return overlay
