Test = open('DCT_Out_Result.bmp','rb')
dd=Test.read()
#因为无论是学号，还是混沌映射
#操作的数据都只有4*10=40位，而且都是末尾，所以有
sum=[]
for i in range(1,41):
    if(dd[-i]%2==1):
        #最低有效位是1
        sum.append(1)
    else:
        #最低有效位是0
        sum.append(0)

print()
print("学号的提取结果：",end='')
for i in range(0,len(sum),4):
    print(hex(sum[i]*8+sum[i+1]*4+sum[i+2]*2+sum[i+3]*1)[2:],end='')
print()
