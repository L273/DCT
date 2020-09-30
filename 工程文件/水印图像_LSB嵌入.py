Test = open('Wait_To_Hide.bmp','rb')
dd=Test.read()

message = [1,0,0,0,
           0,0,0,0,
           0,0,0,0,
           0,0,1,1,
           0,0,0,1,
           0,0,0,1,
           0,1,1,1,
           0,0,0,1,
           0,1,0,1,
           0,1,1,0]

dd1=[]
#将bytes类型的dd进行提取进DD1，不然无法操作
for i in dd:
    dd1.append(i)
j=0
#从图片的后面倒序替换，依次将数据嵌入进去
for i in message:
    if dd[len(dd)-j-1]%2==0:
        #最低有效位是0
        if(i==1):
            dd1[len(dd)-j-1]=dd1[len(dd)-j-1]+1
        else:
            #i为0的情况
            pass
    else:
        #最低有效位是1
        if(i==1):
            pass
        else:
            dd1[len(dd)-j-1]=dd1[len(dd)-j-1]-1
            #i为0的情况
    j+=1
    


#将数据再度转化成bytes类型，然后转存进文件2.bmp
bs = bytes(dd1)

fp = open('Wait_To_Hide.bmp','wb')
fp.write(bs)
fp.close()
# 8003117156 
# 1000
# 0000
# 0000
# 0011
# 0001
# 0001
# 0111
# 0001
# 0101
# 0110


