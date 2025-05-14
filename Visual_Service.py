import cv2
import numpy as np
from collections import deque
import threading
import time
import json
import websockets
import asyncio
from argparse import Namespace
from flask import Flask, Response
from flask_cors import CORS
import logging
import mimetypes
import os
import mmcv
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import threading
import socket
import email.utils
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


# MMDetection/Pose 相关导入
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model as init_pose_estimator
    from mmpose.evaluation.functional import nms
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples
    from mmpose.utils import adapt_mmdet_pipeline
    has_mmdet = True
except ImportError:
    has_mmdet = False

# ================== 全局配置 ==================
global_frame = None
frame_lock = threading.Lock()
ws_data = {}

# 配置邮箱功能
EMAIL_CONFIG = {
    'sender': '2982764154@qq.com',    # 发件邮箱
    'password': 'zsuaypfomgvtdebh',     # 邮箱密码/授权码
    'receiver': '1544178779@qq.com',       # 收件邮箱
    'smtp_server': 'smtp.qq.com',     # SMTP服务器
    'smtp_port': 465,                      # SMTP端口
    'cooldown': 300                        # 防骚扰间隔(秒)
}
last_email_time = 0
# ================== Flask 服务 ==================
app = Flask(__name__)
CORS(app)

# ================== 邮件功能增强 ==================
def send_fall_alert_with_image(frame_data):
    """发送带图片附件的报警邮件"""
    global last_email_time
    success = False  # 新增状态标志
    
    try:
        # 创建安全SSL上下文
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # 使用with语句自动管理连接
        with smtplib.SMTP_SSL(
            EMAIL_CONFIG['smtp_server'],
            EMAIL_CONFIG['smtp_port'],
            context=context,
            timeout=15
        ) as server:
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            # 构建多部分邮件
            msg = MIMEMultipart()
            msg['From'] = formataddr((
                Header('智能跌倒监测系统', 'utf-8').encode(),
                EMAIL_CONFIG['sender']
            ))
            msg['To'] = EMAIL_CONFIG['receiver']
            msg['Subject'] = Header(f'[紧急] 跌倒报警 {time.strftime("%Y-%m-%d %H:%M:%S")}', 'utf-8').encode()
            msg['Message-ID'] = make_msgid()
            msg['Date'] = email.utils.formatdate(localtime=True)

            # 文本部分
            text = MIMEText(f'''检测到人员跌倒！
    事件时间：{time.strftime("%Y-%m-%d %H:%M:%S")}
    请查看附件图片并立即处理！
    --------------------------
    系统自动发送，请勿直接回复''', 'plain', 'utf-8')
            msg.attach(text)

            # 图片附件处理
            _, buffer = cv2.imencode('.jpg', frame_data, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            img_part = MIMEImage(buffer.tobytes(), name=Header('fall_alert.jpg', 'utf-8').encode())
            img_part.add_header('Content-Disposition', 'attachment', 
                            filename=Header('跌倒现场截图.jpg', 'utf-8').encode())
            msg.attach(img_part)

            # 关键发送步骤
            server.send_message(msg)
            success = True  # 标记发送成功

    except smtplib.SMTPServerDisconnected as e:
        if success:
            print("⚠️ 连接已安全关闭（可忽略）")
        else:
            print(f"❌ 连接异常中断: {str(e)}")
    except smtplib.SMTPResponseException as e:
        # 特殊处理QUIT阶段错误
        if e.smtp_code == -1 and "QUIT" in str(e):
            print("⚠️ 安全忽略SSL关闭信号")
        else:
            print(f"❌ 协议错误 ({e.smtp_code}): {e.smtp_error.decode()}")
    except Exception as e:
        print(f"❌ 其他错误: {str(e)}")
    finally:
        if success:
            last_email_time = time.time()
            print(f"✅ 邮件确认已提交到服务器 @ {time.strftime('%H:%M:%S')}")
            
        # 静默处理二次退出
        try:
            server.quit()  # 确保连接关闭
        except:
            pass


@app.route('/')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if global_frame is None:
                    continue
                    
                # 确保颜色空间正确性
                if global_frame.shape[-1] == 3:  # 检查是否为RGB
                    # 转换回BGR进行JPEG编码
                    output_frame = cv2.cvtColor(global_frame, cv2.COLOR_RGB2BGR)
                else:
                    output_frame = global_frame
                    
                ret, buffer = cv2.imencode('.jpg', output_frame)
                
            yield (b'--frame\r\n' + 
                b'Content-Type: image/jpeg\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

# ================== WebSocket 服务 ==================
async def ws_server(websocket):
    global ws_data
    while True:
        try:
            await websocket.send(json.dumps(ws_data))
            await asyncio.sleep(0.1)
        except websockets.exceptions.ConnectionClosed:
            break

def run_websocket():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(ws_server, "0.0.0.0", 5001)
    loop.run_until_complete(start_server)
    loop.run_forever()

class FallDetector:
    """ 综合跌倒检测器，包括基于角度和关键点下降比例的检测 """
    def __init__(self):
        self.history = deque(maxlen=10)  
        self.angle_history = deque(maxlen=25)
        self.hip_history = deque(maxlen=30)
        self.knee_history = deque(maxlen=30)
        self.nose_history = deque(maxlen=25)

        # 可调参数
        self.ANGLE_THRESH = 135         # 身体倾斜角度阈值
        self.CONF_THRESH = 0.4          # 关键点置信度阈值
        self.HIP_THRESH = 0.20         # 髋部下降阈值
        self.KNEE_THRESH = 0.15        # 膝盖下降阈值
        self.NOSE_THRESH = 0.30       # 鼻子下降阈值
        self.RATIO_THRESH = 1.0        # 宽高比阈值

        self.email_sent = False  # 新增状态跟踪

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

        # 计算下降比例
        hip_center_y = (l_hip[1] + r_hip[1]) / 2  # 髋部中心y坐标
        knee_center_y = (keypoints[13][1] + keypoints[14][1]) / 2  # 膝盖中心y坐标
        nose_y = keypoints[0][1]  # 鼻子y坐标

        # 更新历史记录
        self.hip_history.append(hip_center_y)
        self.knee_history.append(knee_center_y)
        self.nose_history.append(nose_y)
        
        # 计算下降比例
        hip_fall_ratio = (hip_center_y - self.hip_history[0]) / self.hip_history[0] if self.hip_history else 0
        knee_fall_ratio = (knee_center_y - self.knee_history[0]) / self.knee_history[0] if self.knee_history else 0
        nose_fall_ratio = (nose_y - self.nose_history[0]) / self.nose_history[0] if self.nose_history else 0

        # 判断各个条件
        cond_hip = hip_fall_ratio > self.HIP_THRESH
        cond_knee = knee_fall_ratio > self.KNEE_THRESH
        cond_nose = nose_fall_ratio > self.NOSE_THRESH
        cond_angle = np.nanmean(self.angle_history) < self.ANGLE_THRESH
        cond_ratio = aspect_ratio > self.RATIO_THRESH

        # 综合判断
        conditions_met = sum([cond_hip, cond_knee, cond_nose, cond_angle, cond_ratio])
        current_status = conditions_met >= 3
        self.history.append(current_status)

        if len(self.history) >= 10:
            return np.mean(self.history) > 0.7
        return current_status

def main():
    global global_frame
    args = Namespace(
        det_config='mmdetection/configs/yolox/yolox_l_8xb8-300e_coco.py',
        det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        pose_config='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py',
        pose_checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth',
        device='cuda:0',
        det_cat_id=0,
        bbox_thr=0.45,
        nms_thr=0.35,
        kpt_thr=0.3,
        draw_bbox=True,
        skeleton_style='mmpose'
    )

    # 初始化模型
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    # 启动网络服务
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    flask_thread.start()
    ws_thread.start()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头初始化失败")
        return

    print("服务已启动：")
    print(f"HTTP流地址：http://[本机IP]:5000")
    print(f"WebSocket地址：ws://[本机IP]:5001")

    # 初始化跌倒检测器列表
    fall_detectors = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            # 更新全局帧（带锁保护）
        vis_frame = frame.copy()
        with frame_lock:
            # 转换为RGB格式供后续处理
            global_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            global_frame = global_rgb.copy()  # 保存未标注的原始帧


        # 目标检测
        det_result = inference_detector(detector, frame)
        pred_instances = det_result.pred_instances.cpu().numpy()
        
        # 处理检测框
        bboxes = np.concatenate(
            (pred_instances.bboxes, pred_instances.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instances.labels == args.det_cat_id,
                                       pred_instances.scores > args.bbox_thr)]
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

        # 姿态估计
        pose_results = inference_topdown(pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)

        # 可视化处理
        # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # visualizer.add_datasample(
        #     'result',
        #     img_rgb,
        #     data_sample=data_samples,
        #     draw_gt=False,
        #     draw_bbox=args.draw_bbox,
        #     kpt_thr=args.kpt_thr,
        #     show=False)

        # vis_frame = cv2.cvtColor(visualizer.get_image(), cv2.COLOR_RGB2BGR)

        if pose_results:  # 只在有检测结果时处理
            # 步骤1：转换输入为RGB（MMPose需要RGB输入）
            process_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 步骤2：进行可视化处理（内部保持RGB）
            visualizer.add_datasample(
                'result',
                process_frame,  # 使用RGB帧
                data_sample=data_samples,
                draw_gt=False,
                draw_bbox=args.draw_bbox,
                kpt_thr=args.kpt_thr,
                show=False
            )
            
            # 步骤3：获取结果并转回BGR
            vis_result = visualizer.get_image()
            vis_frame = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)  # 最终转BGR

        alarm_status = False
        # 跌倒检测处理
        if pose_results:
            # 动态维护检测器实例
            while len(fall_detectors) < len(pose_results):
                fall_detectors.append(FallDetector())
            fall_detectors = fall_detectors[:len(pose_results)]

            for i, pose_data in enumerate(pose_results):
                fall_detector = fall_detectors[i]
                instance = pose_data.pred_instances
                
                # 获取关键点和检测框
                keypoints = instance.keypoints[0]
                scores = instance.keypoint_scores[0]
                bbox = instance.bboxes[0]

                # 检测跌倒状态
                is_fallen = fall_detector.analyze_posture(keypoints, bbox)
                alarm_status |= is_fallen
                
                # alarm_status |= is_fallen

                # 绘制检测框和状态
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])
                color = (0, 0, 255) if is_fallen else (0, 255, 0)
                label = "FALL!" if is_fallen else "SAFE"
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, label, (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(vis_frame, label, (x1, y1-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 报警处理
                if is_fallen:
                    if not fall_detector.email_sent:
                        # 捕获报警帧（带锁保护）
                        alert_frame = vis_frame.copy()  # 关键修改点
                        
                        # 异步发送邮件（避免阻塞主线程）
                        threading.Thread(
                            target=send_fall_alert_with_image,
                            args=(alert_frame,),
                            daemon=True
                        ).start()
                        
                        fall_detector.email_sent = True
                        # fall_detector.first_fall_frame = alert_frame
                else:
                    fall_detector.email_sent = False
                    fall_detector.first_fall_frame = None

        # 更新全局帧
        with frame_lock:
            # global_frame = vis_frame.copy()
            # 将BGR的vis_frame转RGB供网络传输
            global_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

        # 更新WebSocket数据
        ws_data.update({
            'alarm_status': alarm_status,
            'timestamp': time.time()
        })

        cv2.imshow('Fall Detection System', vis_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()