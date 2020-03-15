import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cameraIP", type=str)
parser.add_argument("--threshold", type=float)
args = parser.parse_args()

while True:
    time.sleep(1)
    print('cameraIP=%s ,threshold=%f' % (args.cameraIP, args.threshold))

