while(True):
    A = "8003117156"
    B = input("请输入待比较的序列：")
    if(len(A)==len(B)):
        break
    else:
        print("请输入两串长度一致的序列")

sum = 0
for i,j in zip(A,B):
    if(i==j):
        sum = sum+1

print("两串序列的相似度位：",(sum/len(A))*100,"%")