"""Chạy file video qua detect/tracking/hành vi; đầu ra chỉ alert, tùy chọn gửi BE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from pipeline.runner import main

if __name__ == '__main__':
    main('video')
