# import this
from numpy import *
import numpy as np
import math
import time
from datetime import date
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
import sys
# print(sys.version)
# print(sys.argv[0])

# import cv2
# print(cv2.__version__)

# import tensorflow as tf
# print(tf.__version__)

# Ql = [np.random.randint(100) for i in range(10)]
# print(Ql)
# mQ = np.max(Ql)
# print(mQ)
# al = np.where(Ql==mQ)[0]
# print(al)
# a = np.random.choice(al)
# print(a)

b = np.random.choice(10,
	2, replace=False)
print(b)



# a=[0]*10
# for i in range(10):
# 	a[i]=i
# 	print(a[i])

# a=256*256*256
# b=pow(256,3)
# print(a,b)


# GP=0.5*3.0+2*3.7+2*3.3+1.5*3.3+1.5*2.7+2*3.7+1*3.7+1*2.0+1*3.7+0.5*2.7+0.5*3.3+1*4.0
# G=0.5+2+2+1.5+1.5+2+2+1+1+1+0.5+0.5+1
# GPA=GP/G
# print('%.2f' % GPA)

# zongGPA=3.33+3.89+3.70+3.66+3.39+3.50+2.93
# GPA=zongGPA/7
# print('%.2f' % GPA)


# m = 5
# # Points x-coordinate and dummy value (x0, x1).
# X0 = np.ones((m, 1))
# X1 = np.arange(1, m+1).reshape(m, 1)
# X = np.hstack((X0, X1))
# print(X)


# x = np.arange(12).reshape(3,4)
# print(x)

# x = np.mat(np.delete(x,-1,axis=1))
# print(x)

# x = np.hstack((np.ones((x.shape[0],1)),x))
# print(x)

# a=16+23+51+2+31+24+2+59
# print(a)


# a=[1,2,3]
# b=(1,2,3)
# c={1,2,3}
# print(type(a))
# print(type(b))
# print(type(c))

# try:
#     a=3/0
#     print(a)
# except BaseException:
#     print('error!')

# print("Python Version {}".format(str(sys.version).replace('\n', '')))

# a=log(4)   #ln(x)
# b=np.log(4)
# print(a,b)

# C=np.linspace(10,80,8,dtype=int)
# print(C)
# for e in range(0,100,10):
# 	print(e)


# a=np.random.random((3,4))*2-1
# print(a)
# b=np.random.normal(0,pow(200,-0.5),(3,4))
# print(b)

# for i in range(0,1000,50):
# 	print(i)


# x = np.linspace(-5,5)
# y = 1/(1+np.exp(-x))
# plt.plot(x,y)
# plt.show()


# t1=time.time()
# a=np.arange(100).reshape(10,10)
# b=np.arange(10).reshape(10,1)
# a=a*b*a*b
# print(a)
# t2=time.time()
# print('%.3f s' % (t2-t1))

# import os
# print('当前目录：',os.getcwd())

# s="apple,peach,banana,peach,pear"
# print(s.find("peach"))

# # 三维图
# fig=plt.figure()
# ax=Axes3D(fig)
# x=np.arange(-2,2,0.1)  #linspace(-2,2)
# y=np.arange(-2,2,0.1)  #linspace(-2,2)
# X, Y = np.meshgrid(x, y)
# z=X**2+abs(X)*Y+Y**2
# ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='rainbow')
# # plt.plot(z)
# plt.show()



# X=np.array([-1,-1,0,2,0],[-2,0,0,1,1])
# print(X)

# print(sum(range(1,101)))


# def dateBetween(y1,m1,d1,y2,m2,d2):
#     return (date(y2,m2,d2)-date(y1,m1,d1)).days
# y1,m1,d1=2020,1,1
# y2,m2,d2=2020,12,3
# print(dateBetween(y1, m1, d1,y2,m2,d2))

# a, b = 525, 16
# print(a//b)
# print(a%b)
# print(0.625*(1+1.99)+0.375*(1.4+0.0199*120))


# for i in range(1,10):
#     for j in range(i):
#         j += 1
#         print("%d * %d = %-2d" % (i,j,i*j))
#     print()


# from numba import jit
# @jit
# def foo(x,y):
#         tt = time.time()
#         s = 0
#         for i in range(x,y):
#                 s += i
#         print('Time used: {} sec'.format(time.time()-tt))
#         return s

# print(foo(1,100000000))



# a=list(range(10))
# print(a)
# b=a.copy()
# print(b)
# print(len(b))
# b.extend((range(5,10)))
# print(b)

# x=[1,2,1,3,4,1,1,2,1]
# for i in range(len(x)-1,-1,-1):   #成功
#     if x[i]==1:
#         del x[i]

# for i in x:     #错误
#     if i==1:
#         x.remove(i)

# for i in x:     #成功
#     x.remove(1)
# print(x)


# n = 3
# A = np.zeros([n, n, n], dtype=float)
# print(A)
# a = [0, 1]
# print(type(a))

# print(np.random.rand(1,3,2))

# print(np.random.randn(2,3))

# print(np.arange(5))

# #当replace = False时，返回的数中不可以有重复的
# print(np.random.choice(5,3,replace=False))

# x=np.random.uniform(-30,30,(5,4))
# print(x)
# print(x[1])

# # print(np.square([3,2]))
# print()
# a=np.random.rand(5,4)
# print(a)
# print()
# print(a[np.less(a,0.2)])
# print(a[a<0.2])

# list = [9, 12, 88, 14, 25] # 最小的话 max换成min
# max_index =  max(list) # 最大值的索引
# max_value = list.index(max(list)) # 返回最大值
# print(max_index,max_value)


# a = np.array([1, 2, 3, 4])
# b = np.array((5, 6, 7, 8))
# c = np.array([[11, 45, 8, 4], [4, 52, 6, 17], [2, 8, 9, 100]])
  
# print(a)
# print(b)
# print(c)
  
# print(np.argmin(c))
# print(np.argmin(c, axis=0)) # 按每列求出最小值的索引
# print(np.argmin(c, axis=1)) # 按每行求出最小值的索引

# print(np.min(23))

# pop = np.random.randint(0, 5, size=(1, 4))
# print(pop)
# print(pop.repeat(3, axis=0))

# a=np.array([1,2,3]).reshape(1,3)
# print(a.repeat(4,axis=0))

# a=np.array([7,8,3,1])
# b=np.array([1,12,1,20])
# update_id = np.greater(a, b).reshape(1,4)
# print(update_id)
# print(update_id.repeat(4,axis=0))
# print(np.argmin(a))
# print(a)


# t1=t2=3
# print(t1,t2)


# t1 = [0]*5
# print(t1)
# t1 =  [i for i in range(5)]
# t2 = t1
# print(t1)
# print(t2)
# i=0
# i+=1
# print(i)
# a=random.random()
# print(a)

# a=inf
# print(a)

# m=10
# g=[0]*m
# vis = [0 for i in range(m)]
# on = 0
# while on < m:
#     # rd = random.randint(0,m-1)
#     rd = (int)(random.uniform(0,m-1))
#     if(vis[rd] == 0):
#         vis[rd] = 1
#         g[on] = rd
#     on += 1
# for i in range(m):
# 	print(g[i])

# p=[i for i in range(10)]
# random.shuffle(p)
# print(p)

# print(list(enumerate('abcde')))
# # print(dir('__builtins__'))
# import math
# print(pi)

# x=list(map(int,input().split(",")))
# a=list(map(eval,input().split()))
# print(a,a[0])
# for i in a:
# 	print(i)
	# print('%d ' % i,end="")