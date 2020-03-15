import json
#将txt中字符串转换回字典
file = open('0.txt', 'r')
js = file.read()
dic = json.loads(js) 
print(dic['messageBody']['function'])
if dic['messageBody']['function'] =='fff':
    print('yes')
file.close() 