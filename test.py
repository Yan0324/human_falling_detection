import logging
import mimetypes
import os
import time
from argparse import Namespace
import argparse
from collections import deque
import numpy as np
import cv2
import mmcv
import mmengine
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class FallDetector:
    """ 综合跌倒检测器，包括基于角度和关键点下降比例的检测 """
    def __init__(self):
        self.history = deque(maxlen=10)  
        self.angle_history = deque(maxlen=15)
        self.hip_history = deque(maxlen=20)
        self.knee_history = deque(maxlen=20)
        self.nose_history = deque(maxlen=15)

        # 可调参数
        self.ANGLE_THRESH = 135         # 身体倾斜角度阈值
        self.CONF_THRESH = 0.4          # 关键点置信度阈值
        self.HIP_THRESH = 0.20         # 髋部下降阈值
        self.KNEE_THRESH = 0.15        # 膝盖下降阈值
        self.NOSE_THRESH = 0.30       # 鼻子下降阈值
        self.RATIO_THRESH = 1.0        # 宽高比阈值

    def analyze_posture(self, keypoints, bbox):
        """通过角度和关键点位置判断是否跌倒"""
        # 计算宽高比
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height
        
        # 计算角度
        LS, RS = 5, 6  # 左肩和右肩
        LH, RH = 11, 12  # 左髋和右髋

        # 获取肩部和髋部关键点数据
        l_shoulder = keypoints[LS]
        r_shoulder = keypoints[RS]
        l_hip = keypoints[LH]
        r_hip = keypoints[RH]
        
        # 计算身体中心点（脖部和髋部之间的连线）
        neck = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        
        # 计算倾斜角度
        body_vector = np.array([neck[0] - hip_center[0], neck[1] - hip_center[1]])  # 脖部到髋部的向量
        vertical = np.array([0, 1])  # 垂直向量（正上方）
        
        cos_theta = np.dot(body_vector, vertical) / (np.linalg.norm(body_vector) + 1e-6)  # 余弦值
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))  # 计算角度

        # 将计算出的角度加入历史记录
        self.angle_history.append(angle)

        # 计算下降比例和角度等其他因素
        hip_center_y = (l_hip[1] + r_hip[1]) / 2  # 髋部中心y坐标
        knee_center_y = (keypoints[13][1] + keypoints[14][1]) / 2  # 膝盖中心y坐标
        nose_y = keypoints[0][1]  # 鼻子y坐标

        # 更新历史记录
        # if len(self.hip_history) == 0:
        #     self.hip_history.append(hip_center_y)
        # if len(self.knee_history) == 0:
        #     self.knee_history.append(knee_center_y)
        # if len(self.nose_history) == 0:
        #     self.nose_history.append(nose_y)
        self.hip_history.append(hip_center_y)
        self.knee_history.append(knee_center_y)
        self.nose_history.append(nose_y)
        
        # 判断各个条件
        hip_fall_ratio = (hip_center_y - self.hip_history[0]) / self.hip_history[0] if self.hip_history else 0
        knee_fall_ratio = (knee_center_y - self.knee_history[0]) / self.knee_history[0] if self.knee_history else 0
        nose_fall_ratio = (nose_y - self.nose_history[0]) / self.nose_history[0] if self.nose_history else 0

        # 判断各个条件
        cond_hip = hip_fall_ratio > self.HIP_THRESH
        cond_knee = knee_fall_ratio > self.KNEE_THRESH
        cond_nose = nose_fall_ratio > self.NOSE_THRESH

        # 处理角度历史
        if not self.angle_history:
            self.angle_history.append(0)  # 初始角度填充为0
        
        cond_angle = np.nanmean(self.angle_history) < self.ANGLE_THRESH  # 使用角度历史均值

        # 宽高比判断
        cond_ratio = aspect_ratio > self.RATIO_THRESH

        # 打印各个判断条件
        print(f"Hip fall ratio: {hip_fall_ratio:.4f}, Knee fall ratio: {knee_fall_ratio:.4f}, Nose fall ratio: {nose_fall_ratio:.4f}")
        print(f"Body angle: {np.nanmean(self.angle_history):.2f} degrees")
        print(f"Aspect ratio condition: {aspect_ratio}")

        # 总结满足条件的数量
        conditions_met = sum([cond_hip, cond_knee, cond_nose, cond_angle, cond_ratio])

        # 如果满足的条件数量 >= 3，判定为跌倒
        fall_status = conditions_met >= 3

        # 使用历史记录来判断是否跌倒
        self.history.append(fall_status)
        if len(self.history) >= 10:
            return np.mean(self.history) > 0.7  # 如果超过70%的历史记录表示跌倒，则判定为跌倒

        return fall_status


def process_one_image(args, img, detector, pose_estimator, visualizer, fall_detectors):
    """基于角度和关键点位置的单帧处理"""
    det_result = inference_detector(detector, img)
    pred_instances = det_result.pred_instances.cpu().numpy()
    
    # 检测框处理
    bboxes = np.concatenate((pred_instances.bboxes, pred_instances.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instances.labels == args.det_cat_id,
                                   pred_instances.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # 姿态估计
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)
    
    # 可视化处理
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)
    
    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=0.001,
            kpt_thr=args.kpt_thr)
    
    frame_vis = visualizer.get_image()
    pred_instances = data_samples.get('pred_instances', None)
    
    # 跌倒检测与标注
    if pred_instances:
        # 维持检测器实例
        while len(fall_detectors) < len(pred_instances):
            fall_detectors.append(FallDetector())
        fall_detectors = fall_detectors[:len(pred_instances)]
        
        for i, (detector, instance) in enumerate(zip(fall_detectors, pred_instances)):
            # 重塑关键点维度
            try:
                keypoints = instance.keypoints.reshape(-1, 2)  # 确保(N,2)形状
                scores = instance.keypoint_scores.reshape(-1, 1)
                if keypoints.shape[0] != scores.shape[0]:
                    continue
            except Exception as e:
                print(f"关键点重塑错误: {e}")
                continue
            
            kpts = np.concatenate([keypoints, scores], axis=1)
            # 获取检测框坐标
            bbox = instance.bboxes[0]
            is_fallen = detector.analyze_posture(kpts, bbox)  # 添加bbox参数
            
            # 获取检测框坐标
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # 绘制带状态的边界框
            color = (255, 0, 0) if is_fallen else (0, 255, 0)
            label = "FALL!" if is_fallen else "SAFE"
            
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame_vis, label, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
            cv2.putText(frame_vis, label, (x1, y1-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame_vis, fall_detectors


def main():
    parser = argparse.ArgumentParser(description='基于角度的跌倒检测系统')
    parser.add_argument('--input', type=str, required=True,
                       help='输入源（视频文件/摄像头ID/图片路径）')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录路径')
    parser.add_argument('--draw-bbox', action='store_true', help='是否绘制检测框')
    parser.add_argument('--radius', type=int, default=5, help='关键点绘制半径')
    parser.add_argument('--thickness', type=int, default=4, help='连线粗细')
    parser.add_argument('--alpha', type=float, default=0.8, help='透明度')
    args_cmd = parser.parse_args()

    # 配置参数
    args = Namespace(
        det_config='mmdetection/configs/yolox/yolox_l_8xb8-300e_coco.py',
        det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        pose_config='/home/yc/yan/mywork/human_falling_detection/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py',
        pose_checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth',
        device='cuda:0',
        det_cat_id=0,
        bbox_thr=0.35,
        nms_thr=0.4,
        kpt_thr=0.3,
        show=False,
        draw_heatmap=False,
        draw_bbox=True,      # 控制是否绘制检测框
        show_kpt_idx=False,
        skeleton_style='mmpose',
        radius=6,            # 关键点半径
        thickness=5,         # 连线粗细
        alpha=0.8            # 透明度
    )


    assert has_mmdet, '需要安装mmdet'

    # 初始化模型
    detector = init_detector(args.det_config, args.det_checkpoint, args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap)))
    )

    # 初始化可视化工具
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, 
                              skeleton_style=args.skeleton_style)

    # 输入类型判断
    input_type = "video"
    if args_cmd.input.isdigit():
        input_type = "webcam"
    elif os.path.isfile(args_cmd.input):
        mime_type = mimetypes.guess_type(args_cmd.input)[0]
        if mime_type and mime_type.startswith('image'):
            input_type = "image"

    # 处理图片输入
    if input_type == "image":
        img = cv2.imread(args_cmd.input)
        frame_vis, _ = process_one_image(args, img, detector, pose_estimator, visualizer, [])
        output_path = os.path.join(args_cmd.output, 'result.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))
        print(f'结果保存至: {output_path}')
        return

    # 视频/摄像头处理
    cap = cv2.VideoCapture(int(args_cmd.input) if input_type == "webcam" else args_cmd.input)
    video_writer = None
    fall_detectors = []
    alert_counter = 0
    
    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    preset_fps = cap.get(cv2.CAP_PROP_FPS)

    # 添加帧率计算变量
    prev_time = time.time()
    actual_fps = 0
    
    # 创建输出目录
    os.makedirs(args_cmd.output, exist_ok=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 处理当前帧
        frame_vis, fall_detectors = process_one_image(
            args, frame, detector, pose_estimator, visualizer, fall_detectors
        )
        
        # 计算实际帧率
        current_time = time.time()
        delta = current_time - prev_time
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / (delta + 1e-6))  # 平滑处理
        prev_time = current_time

        # 转换色彩空间并保存/显示
        frame_out = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)

        # 显示帧率信息（左上角）
        cv2.putText(frame_out, f"FPS: {actual_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_out, f"Preset: {preset_fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        if args_cmd.output:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = os.path.join(args_cmd.output, 'output.avi')
                video_writer = cv2.VideoWriter(output_path, fourcc, preset_fps, 
                                             (frame_width, frame_height))
            video_writer.write(frame_out)
        
        # 显示结果
        cv2.imshow('Fall Detection System', frame_out)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
