from socket import *
import json
num = 0
# 创建socket
while True:
   tcp_client_socket = socket(AF_INET, SOCK_STREAM)

   # 目的信息
   server_ip = input("请输入服务器ip:")
   server_port = int(input("请输入服务器port:"))
   # 链接服务器
   tcp_client_socket.connect((server_ip, server_port))

   # 提示用户输入数据
   while True:
       print("第"+str(num)+"轮")
       #判断是否断开连接
       flag = input("请输入是否继续通信（yes/no）")
       if flag == 'no':
           tcp_client_socket.send('over'.encode("utf-8"))
           recvData = tcp_client_socket.recv(1024)
           if recvData.decode('utf-8') == 'over':
               print('与', server_ip, server_port, '断开连接')
               break
       else:
           tcp_client_socket.send('continue'.encode("utf-8"))
       #输入信息
       messageName = input("messageName：")
       messageId = input("messageId：")
       cameraIP = input("cameraIP：")
       function = input("function：")
       params = input("params：")
       funcControStatus = input("funcControStatus：")
       #构建字典
       dict = {
              "messageBody": {
                  "function": function,
                  "params": params,
                  "cameraIP": cameraIP,
                  "funcControStatus": funcControStatus,
              },
              "messageHead": {
                  "messageName": messageName,
                  "messageId": messageId
              }
       }
       send_data = json.dumps(dict)      #字典转字符串
       #发送信息
       tcp_client_socket.send(send_data.encode('utf-8'))
       print('数据已发送')
       #接受返回数据，字符串转换成字典
       back = tcp_client_socket.recv(1024)
       back_data = json.loads(back.decode('utf-8'))
       print(back_data)
       print('返回数据接收完成')
       num = num+1

   # 关闭套接字
   tcp_client_socket.close()
