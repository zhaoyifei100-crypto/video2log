#!/usr/bin/env python3
"""
Mac 摄像头推流到nanobot
使用 HTTP MJPG 格式，方便 OpenCV 读取
"""

import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import numpy as np
import sys
import socket


def get_local_ip():
    """自动获取本机 IP 地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.15"


# 自动获取本机 IP
HOST = get_local_ip()  # 监听地址（自动检测）
PORT = 8554  # 监听端口

# apple camera_index may not be 0
#   0 may be your iphone camera, 1 may be your macbook camera
CAMERA_INDEX = 0  # 摄像头编号

# 全局帧
latest_frame = None
frame_lock = threading.Lock()


class MJPGHandler(BaseHTTPRequestHandler):
    """MJPG 流处理器"""

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            while True:
                with frame_lock:
                    if latest_frame is None:
                        continue
                    frame_bytes = cv2.imencode(".jpg", latest_frame)[1].tobytes()

                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame_bytes))
                self.end_headers()
                self.wfile.write(frame_bytes)
                self.wfile.write(b"\r\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # 禁用日志


def capture_loop():
    """摄像头捕获循环"""
    global latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    print(
        f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS))}fps"
    )

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            print("Frame grab failed")


def main():
    # 启动捕获线程
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # 启动 HTTP 服务器
    server = HTTPServer((HOST, PORT), MJPGHandler)
    print(f"Stream server started: http://{HOST}:{PORT}/stream")
    print(f"在树莓派上用: ffplay http://{HOST}:{PORT}/stream")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
