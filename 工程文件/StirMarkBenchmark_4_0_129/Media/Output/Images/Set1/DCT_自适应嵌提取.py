import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from math import ceil #向上取整
matplotlib.rcParams['font.family']='SimHei' #中文显示

#交换x，y
def Xchange_x_y(x,y):
    temp = x
    x = y
    y = temp
    return x,y

#Arnold置乱
def Arnold(block):
    N = 8
    a = 3
    b = 5
    A1 = [1,a,b,a*b+1] #置乱矩阵
    A1 = np.asarray(A1).reshape(2,2)
    
    A2 = [a*b+1,-a,-b,1] #反置乱矩阵
    A2 = np.asarray(A2).reshape(2,2) 
    #先确定a，b，以及矩阵的宽度，以及需要使用的矩阵
    
    #由于Arnold操作的是矩阵下标
    #这里，我们可以先用一个存有下标的temp数组生产一张映射表
    #然后在用Arnold打乱，之后再使用映射关系去修改block的下标


    # 与嵌入的置乱相反，这里反乱
    # te检查算法，无误
    te1=[]
    for i in range(8):
        for j in range(8):
            te1.append(np.dot(A2,np.asarray([i,j]).reshape(2,1))%8)
    te1 = np.asarray(te1).reshape(8,8,2)
    # print(te1) 
   
    
    #再用一张表存储block的映射
    te=[]
    for i in range(8):
        for j in range(8):
            te.append(block[te1[i,j][0],te1[i,j][1]])
    te = np.asarray(te).reshape(8,8)
    #进行映射
    for i in range(8):
        for j in range(8):
            block[i][j] = te[i][j]
    
    return block
    
    

#返回图像的DC系数值
def Get_Ld(block):
    #一般来说，图像的系数值，就是左上角坐标内的数值
    return block[0][0]

#返回一组数列的和
def Get_Sum(list):
    sum = 0
    for i in list:
        sum = sum + i
    return sum
    

#返回图像的方差值
def Get_Lv(block):
    #Lv = 1/(r1*r2)∑∑(D(i,j)-meanD(i,j))
    #其中两个西格玛，第一个是r1，第二个是r2
    #而其中的r1和r2是分块的宽度和高度
    #这里我们8*8的分块，所以r1=r2=8，这里统一设置为r=8
    
    r = 8
    block_mean = block.mean() #8*8方块内的算术平均值
    
    #外层r1的∑
    sum_i=0
    for i in range(r):
        #内层r2的∑
        sum_j=0
        for j in range(r):
            sum_j =  sum_j + block[i,j]-block_mean
        sum_i = sum_i+sum_j
    
    return (1/r**2)*sum_i    


#计算式：Y=Lv+Ld/a
#Y为纹理和亮度的一个特征值
#其中：Lv是图像的方差
#      Ld是图像的DC系数
#      a是要给经验值
def Get_Y(block):
    #因为Lv和Ld都有函数实现
    #这里只要取一个合适的经验值就好了
    a = 5
    return Get_Lv(block)+Get_Ld(block)/a
    
        

io = cv2.imread('Done_CONV_1.bmp')#用于显示原图
output = io #用于显示切入水印的图片

img = io[:, :, 0] # 获取rgb通道中的一个

width,high = img.shape #提取宽度和高度
img = np.float32(img) # 将数值精度调整为32位浮点型

blocks = [[]for i in range(high//8)] #存储得到的8*8数据块,一个维度就是一行

Ys = [] #存储得到的Y值

#8*8分块处理
hdata = np.vsplit(img,high/8)
#水平方向分割，用高度作为指标，分成32行
for i in range(0,high//8):
    blockdata = np.hsplit(hdata[i],width/8)
    #横向分割，用高度作为指标，分成32列
    #至此32*32的8*的小方块切割完毕
    
    #处理各个方块的数据
    #由于循环里的操作单位是列中切割出的32个行的小方块
    #所以，这里循环的尺标就是高度
    for j in range(0,width//8):
        block = blockdata[j]
        #block为待操作的8*8小方块
       
        # 先进行Arnold置乱,共计2次(后需要修改成10次，以方便抵抗攻击)
        # for z in range(2):
            # block = Arnold(block)
        #置乱后再进行DCT变换
        #只是，需要先转换成float类型
        #因为dct的操作数是float32类型
        block_dct = cv2.dct(block.astype(np.float))
        
        #计算Y值
        Y = Get_Y(block_dct)
        
        #用于测试dct，无误
        # block_rec = cv2.idct(block_dct)
        
        blocks[i].append(block_dct) 
        Ys.append(Y)


Ys=np.asarray(Ys)
blocks=np.asarray(blocks)
#转换成numpy，以便之后的计算

Y_t=Ys.mean()*2

'''
↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑前面基本上和提取一模一样
'''

#设置阈值Y_t
#选取的坐标都是经验值
#大点于前，小点于后
#小于阈值，利用(5,3),(3,5)两个点来操作，存一位
#超过阈值，利用(5,3),(3,5)|(3,4),(4,3)四个点来操作，存两位

#如何p1>p2，就返回true。反之则返回false
def Return_two_point(block,p1,p2):
    if(block[p1[0],p1[1]]>block[p2[0],p2[1]]):
        return True
    else:
        return False

def Return_p1_p2(p1,p2,i,j,seek):
    block=blocks[i,j] #取出标记的8*8小方块
    if(Return_two_point(block,[p1[0],p1[1]],[p2[0],p2[1]])):
        return 255.0
    else:
        return 0



message = []
length = 32*32
seek=0
for i in range(high//8):
    for j in range(width//8):
        message.append(Return_p1_p2([5,3],[3,5],i,j,seek))
        seek = seek+1
        # print(message)
        if(seek==length):
            break
        if(Ys[i*8+j]>Y_t):
            message.append(Return_p1_p2([4,3],[3,4],i,j,seek))
            seek = seek+1
            if(seek==length):
                break
    if(seek==length):
        break

message = np.asarray(message).reshape(32,32)
plt.imshow(message,'gray')
plt.show()
cv2.imwrite("DCT_Out_Result.bmp",message)

