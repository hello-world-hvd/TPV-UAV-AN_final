"""Kiểm tra publisher viz: queue giới hạn, bỏ frame cũ, lệnh encode và dừng worker; không truy cập mạng."""
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline.publish import RTMPPublisher


def main():
    with patch.object(RTMPPublisher,'_run',return_value=None):
        publisher=RTMPPublisher('rtmp://example.invalid/live/viz',15,'ffmpeg','nvenc')
        frame=np.zeros((64,96,3),np.uint8)
        for i in range(10):
            frame[:]=i
            publisher.write(frame)
        assert publisher.frames.qsize()==2 and publisher.stats['dropped']==8
        assert int(publisher.frames.get()[0,0,0])==8
        command=publisher.command(frame.shape)
        assert 'h264_nvenc' in command and command[-1].endswith('/viz')
        assert command[command.index('-framerate')+1]=='15'
        publisher.close()
    print('Đạt: queue publish giới hạn, bỏ frame cũ, encode NVENC và shutdown; không gọi mạng.')

if __name__=='__main__':main()
