"""Chạy RTMP/HLS liên tục, tự kết nối lại; lưu hoặc gửi clip/ảnh cảnh báo tới BE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from pipeline.runner import main

if __name__ == '__main__':
    main('stream')
