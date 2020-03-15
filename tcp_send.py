from socket import *

import json


data = {
    "messageName": "FuncControlRequest",
    "funcControStatus": 0,
    "function": "objectDetection",
    "messageId": 1,
    "cameraIP": "192.168.1.103",
    "params": {
        'threshold':0.5
    }
    
}
data = json.dumps(data)


tcp_socket = socket(AF_INET, SOCK_STREAM)

# 目的信息
server_ip = '127.0.0.1'
server_port = 7788
# 链接服务器
tcp_socket.connect((server_ip, server_port))
tcp_socket.send(data.encode("utf-8"))