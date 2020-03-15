from socket import *
import json
# 创建socket
tcp_server_socket = socket(AF_INET, SOCK_STREAM)

# 本地信息
address = ('127.0.0.1', 7788)

# 绑定
tcp_server_socket.bind(address)

# 使用socket创建的套接字默认的属性是主动的，使用listen将其变为被动的，这样就可以接收别人的链接了
#listen里的数字表征同一时刻能连接客户端的程度.
tcp_server_socket.listen(128)
num = 0
# 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
# client_socket用来为这个客户端服务
# tcp_server_socket就可以省下来专门等待其他新客户端的链接
# clientAddr 是元组（ip，端口）
def maketxt(data, num):
    #字典转换成字符串写入txt
    file = open(str(num) + '.txt', 'w')
    write_date = json.dumps(data)
    file.write(write_date)
    # 关闭文件
    file.close()
    return

while True:
    client_socket, clientAddr = tcp_server_socket.accept()
    print('got connected from', clientAddr)
    while True:
        # 接收对方发送过来的数据，和udp不同返回的只有数据
        # 接收flag判断是否断开连接
        flag = client_socket.recv(1024)  # 接收1024个字节
        if flag.decode('utf-8') == 'over':
            client_socket.send("over".encode('utf-8'))
            print('与', clientAddr, '连接断开')
            break

        print("第" + str(num) + "轮")
        #接受数据，并字符串转字典
        getdata = client_socket.recv(1024)
        data = json.loads(getdata.decode('utf-8'))
        print('接受完成')
        print(data)
        # #发送一些数据到客户端
        #client_socket.send(messageName+"\r\n".encode('utf-8')+messageId)
        # 写入文档
        maketxt(data, num)
        #返回数据
        back = {
              "messageBody": {
                  "function": data['messageBody']["function"],
                  "params": data['messageBody']["params"],
                  "cameraIP": data['messageBody']["cameraIP"],
                  "responseStatus": 1,
              },
              "messageHead": {
                  "messageName": data['messageHead']["messageName"],
                  "messageId": data['messageHead']["messageId"]
              }
        }
        back_data = json.dumps(back)
        client_socket.send(back_data.encode('utf-8'))
        print('返回数据发送成功')
        #计数器加一
        num = num + 1
    client_socket.close()

# 关闭为这个客户端服务的套接字，只要关闭了，就意味着为不能再为这个客户端服务了，如果还需要服务，只能再次重新连接

