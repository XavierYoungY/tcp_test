import cv2
import time
import numpy as np
import os
import socket
from timeit import time
import sys
import queue
from threading import Thread
import subprocess as sp
from socket import *
import json

IP_list=["192.168.1.101","192.168.1.102","192.168.1.103"]

def sub_process(cameraIP,threshold):
    command = ['python', os.getcwd() + '/' + 'test.py','--cameraIP',cameraIP,"--threshold",str(threshold)]
    framechild = sp.Popen(command)
    # time.sleep(5)
    # framechild.kill()
    return framechild

def tcp_receive():
    # 本机socket
    local_socket = socket(AF_INET, SOCK_STREAM)
    address = ('127.0.0.1', 7788)
    local_socket.bind(address)
    #listen里的数字表征同一时刻能连接客户端的程度.
    local_socket.listen(128)

    # 对方socket
    des_socket = socket(AF_INET, SOCK_STREAM)
    # 链接服务器
    des_socket.connect(('127.0.0.1', 8000))



    all_cams={}
    for ip in IP_list:
        all_cams[ip]={'sp':'','funcControStatus':0}

    while True:
        client_socket, clientAddr = local_socket.accept()
        data = client_socket.recv(1024).decode("utf-8")
        if data:
            data=json.loads(data)
            messageId=data['messageId']
            cameraIP=data['cameraIP']
            function=data['function']
            threshold=data['params']['threshold']
            messageName = data['messageName']
        if messageName == 'FuncControlRequest':
            funcControStatus=data['funcControStatus']
            if funcControStatus == 0 and all_cams[cameraIP]['funcControStatus']==1:
                all_cams[cameraIP]['sp'].kill()
                all_cams[cameraIP]['funcControStatus']=0
                print('kill-------------------')
                response(des_socket, data,1)

            elif funcControStatus == 1 and all_cams[cameraIP]['funcControStatus']==0:
                cameraIP_sp=sub_process(cameraIP,threshold)
                all_cams[cameraIP]['sp']=cameraIP_sp
                all_cams[cameraIP]['funcControStatus']=1
                response(des_socket, data,1)
            else:
                # 已经开启或者关闭
                response(des_socket, data,1)

        else:
            responseStatus=data['responseStatus']




def response(des_socket, data,responseStatus):
    response_data={
    "messageName": "FuncControlRequest",
    "messageId": data['messageId'],
    "cameraIP": data['cameraIP'],
    "function": data['function'],
    'responseStatus':responseStatus
    }
    response_data = json.dumps(response_data)
    des_socket.send(response_data.encode("utf-8"))

    return response_data




if __name__ == "__main__":
    tcp_receive()
    # sub_process('111',0.3)