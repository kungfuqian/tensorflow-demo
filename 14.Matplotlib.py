##########################################
##       course 1 基本用法

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(-1,1,50)   # 定义x轴
# #y = 2*x +1
# y = x**2                  # 定义y轴
# plt.plot(x,y)             # 将x,y轴上的数据映射到坐标上
# plt.show()                # 显示图片
#



##########################################
##      course 2 figure图像

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(-1,1,50)
# y1 = 2*x +1
# y2 = x**2 +1
#
# # 每生成一张图，需以plt.figure()开头，标题默认从1,2,3,4...顺序排列
# plt.figure()
# plt.plot(x,y1)
#
# plt.figure(num=3,figsize=(8,5))            # num:标题数字设定，figsize=（图像的长,宽）
# plt.plot(x,y2)
# plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')     # 在图3上重叠两条线,其中第二条线属性：
#                                                             # 颜色：红色,线粗细：1.0,线段风格：虚线
#
# plt.show()




##########################################
##      course 3 设置坐标轴1
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(-1,1,50)
# y1 = 2*x +1
# y2 = x**2
#
# plt.figure()
# plt.plot(x,y1)
# plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--')
#
# plt.xlim((-2,2))              # x坐标轴的范围
# plt.ylim((-2,4))         # y坐标轴的范围
# plt.xlabel('I\'m x')        # x坐标的描述
# plt.ylabel("I'm y")         # y坐标的描述
#
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# plt.xticks(new_ticks)       # 换x轴的角标
# plt.yticks([-2,-1.4,0,2,3.5],['very bad','bad','normal','good','very good']) # 添加y轴的角标
# plt.yticks([-2,-1.4,0,2,3.5,4],
#            [r'$very\ bad$',r'$bad$',r'$normal$',r'$good$',r'$very\ good$',r'$\alpha$'])
# # 使用正则表达式，前面加r,用美元符$选出要表达的部分，空格不能被识别，需要加’\‘表示,特殊字符：e.g.：alpha需要用到转义符'\'
#
# plt.show()


##########################################
##      course 4 设置坐标轴2

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(-1,1,50)
# y1 = 2*x +1
# y2 = x**2
#
# plt.figure()
# plt.plot(x,y1)
# plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--')
#
# plt.xlabel('I\'m x')
# plt.ylabel("I'm y")
#
# ticks_limit = np.linspace(-0.5,0.5,5)
# print(ticks_limit)
# plt.xticks(ticks_limit)
# plt.yticks([-2,-1,0,1,2,3],['very bad', r'$bad$','normal','good',r'$very\ good$',r'$\alpha$'])
#
# # gca = "get current axis"
# ax = plt.gca()      # 获取当前轴
# ax.spines['right'].set_color('none')    # 隐藏坐标轴的右边框和上边框
# ax.spines['top'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')       # 设置默认用'botton'作为x轴
# ax.yaxis.set_ticks_position('left')         # 设置默认用’left‘作为y轴
#
# ax.spines['bottom'].set_position(('data',-1))   # 将x轴绑定在y=-1处,即：y的中心为-1;即坐标原点为(*,-1)
# ax.spines['left'].set_position(('data',0.25))   # 同理，将x轴的中心设为0.25，即坐标原点为(0.25,-1)
#
# plt.show()



##########################################
##      course 5 Legend 图例

# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(-1,1,50)
# y1 = 2*x +1
# y2 = x**2
#
# plt.figure()
# l1, = plt.plot(x,y1,label='up')       # 添加'label'属性，即标签
# l2, = plt.plot(x,y2,color='r',linewidth=1.0,linestyle='--',label='down')
# #plt.legend()    # 显示默认图例
# plt.legend(handles=[l1,l2,],loc='best')       # 'loc'位置,best选择最优位置
#
# plt.xlim(-1,1)
# plt.ylim(-1,4)
# plt.xlabel('x轴')
# plt.ylabel('y轴')
# plt.xticks(np.linspace(-1,1,10))
# plt.yticks([-1,0,1,2,3],['very bad',r'$bad$','normal',r'$good$',r'$very\ good$'])
#
# plt.show()



##########################################
##      course 6 Annotation 标注

# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(-1,1,50)
# y1 = 2*x +1
#
# plt.figure()
# plt.plot(x,y1,label='cost')
#
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))
#
# x0 = 0.5
# y0 = 2*x0 + 1
# plt.scatter(x0,y0,s=50,color='red')
# plt.plot([x0,x0],[y0,0],'k--',lw=2.5)
#
#
# ## method 1
# #################
# plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),
#              # 显示的内容,    目标坐标,                    文本坐标,
# textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))   # 正则表达式
#   # 文本格式                  # 字体大小    # 箭头参数      # 箭头类型            # 箭头弧度,角度
#
# ## method 2
# #################
# # 文本标注
#          # 位置    # 内容（正则表达式）                                                      # 字符属性
# plt.text(-0.7,2.5,r'$this\ is\ the\ some\ test.\ \mu\ \sigma_i\ \alpha_t$',fontdict={'size':16,'color':'red'})
#
#
#
# plt.show()


##########################################
##      course 7 tick 能见度

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(-3,3,50)
# y = 0.1*x
#
# plt.figure()
# plt.plot(x,y,lw=10,alpha=0.6)  # alpha: 透明度
# plt.ylim(-2,2)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))
#
# for label in ax.get_xticklabels()+ax.get_yticklabels():
#     label.set_fontsize(12)  # 把字体放大
#     label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7))# 数据的线不会透过坐标
#                         # 背景的颜色        # bboxd边的颜色   # 透明度
#
# plt.show()




##########################################
##      course 8 scatter 散点图


# import matplotlib.pyplot as plt
# import numpy as np
#
# n = 1024
# X = np.random.normal(0,1,n)
# Y = np.random.normal(0,1,n)
#
# T = np.arctan2(Y,X)  # for color value
#
# plt.scatter(X,Y,s=75,c=T,alpha=0.5)   # 画散点图
# #plt.scatter(np.arange(5),np.arange(5))
#
# plt.xlim((-1.5,1.5))
# plt.ylim((-1.5,1.5))
# plt.xticks(())     #将坐标值取消
# plt.yticks(())
# plt.show()


##########################################
##      course 9 Bar 柱状图

# import matplotlib.pyplot as plt
# import numpy as np
#
# n = 12
# X = np.arange(n)
# Y1 = (1-X/float(n))*np.random.uniform(0.5,1.0,n)
# Y2 = (1-X/float(n))*np.random.uniform(0.5,1.0,n)
# print(Y1,Y2)
#
# plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')    #添加柱状图
# plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')
#
#
# for x,y in zip(X,Y1):    #zip的用法：分别把X,Y1的值传给x,y
#     plt.text(x,y,'%.2f'%y,ha='center',va='bottom')
#           # 柱状图的值坐标 # 显示的数值 # 水平对齐方式,垂直对齐方式
#
# for x,y in zip(X,Y2):    #zip的用法：分别把X,Y1的值传给x,y
#     plt.text(x,-y-0.1,'-%.2f'%y,ha='center',va='bottom')
#           # 柱状图的值坐标 # 显示的数值 # 水平对齐方式,垂直对齐方式
#
# plt.xlim(-.5,n)
# plt.ylim(-1.25,1.25)
# plt.xticks(())   #将坐标值取消
# plt.yticks(())
#
# plt.show()



##########################################
##      course 10 contour 等高线图

# import matplotlib.pyplot as plt
# import numpy as np
#
# def f(x,y):
#     # the height function
#     return (1 - x/2 + x**5 + y**3)*np.exp(-x**2-y**2)
#
# n = 256
# x = np.linspace(-3,3,n)
# y = np.linspace(-3,3,n)
# X,Y = np.meshgrid(x,y)    # 定义网格，将x,y绑定成网格输入值
# # use plt.contourf to filling contours 将颜色填充入网格
# # X,Y and value for (X,Y) point
# plt.contourf(X,Y,f(X,Y),8, alpha=0.75,cmap=plt.cm.hot)    # 画等高线
#             #三个坐标轴，等级   透明度    cmap="color_map"
# # use plt.contour to
# C =plt.contour(X,Y,f(X,Y),8, colors='black', lw=0.5)
# # adding label
# plt.clabel(C,inline=True,fontsize=10)
#
#
# plt.xticks(())
# plt.yticks(())
# plt.show()



##########################################
##      course 11 image 图片

# import matplotlib.pyplot as plt
# import numpy as np
#
# # image data
# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
#
# """
# for the value of "interpolation", check this:
# http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
# for the value of "origin"= ['upper', 'lower'], check this:
# http://matplotlib.org/examples/pylab_examples/image_origin.html
# """
# plt.imshow(a,interpolation='None',cmap='bone',origin='upper')
# plt.colorbar(shrink=0.9)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()



##########################################
##      course 12 3D plot 3D数据

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
# # X,Y value
# X = np.arange(-4,4,0.25)
# Y = np.arange(-4,4,0.25)
# X,Y = np.meshgrid(X,Y)
# R = np.sqrt(X**2 + Y**2)
# # height value
# Z = np.sin(R)
#
# ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))  # 建立3D视图
# ax.contourf(X,Y,Z,zdir='y',offset=6,cmap='rainbow') # 添加y轴的等高线
# ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow') # 添加z轴的等高线
# ax.set_zlim(-2,2)       # 限定高度范围
#
# plt.show()


##########################################
##      course 13 subplot 多个显示

# import matplotlib.pyplot as plt
#
# plt.figure()
#
# plt.subplot(2,2,1)  # 建立两行两列图片框,选择第一个图
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,2,2)  # 建立两行两列图片框,选择第二个图
# plt.plot([0,1],[0,1])
#
# plt.subplot(223)
# plt.plot([0,1],[1,0])
#
#
# plt.subplot(224)
# plt.plot([0,1],[1,0])
#
# plt.figure() # 另建一个图片  两行,第一行占一整排
# plt.subplot(2,1,1)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,3,4)     # 第二行是从第4开始计算
# plt.plot([0,1],[0,1])
# plt.subplot(2,3,5)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,3,6)
# plt.plot([0,1],[0,1])
#
# plt.show()




##########################################
##      course 14 subplot in grid 分格显示

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
#
# # method 1: subplot2grid
# #########################
# plt.figure()
# ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)  # 建立第一个子图
#                 #建立3x3个图片，从(0,0)开始，跨度3列，1行
# ax1.plot([1,2],[1,2])
# ax1.set_title('picture 1')
#
# ax2 = plt.subplot2grid((3,3),(1,0),colspan=2)
# ax3 = plt.subplot2grid((3,3),(1,2),rowspan=2)
# ax4 = plt.subplot2grid((3,3),(2,0))
# ax5 = plt.subplot2grid((3,3),(2,1))
#
# # method 2: gridspec
# #########################
# plt.figure()
# gs = gridspec.GridSpec(3,3)     # 构建3x3的图片
# ax6 = plt.subplot(gs[0,:])
# ax7 = plt.subplot(gs[1,:2])
# ax8 = plt.subplot(gs[1:,2])
# ax9 = plt.subplot(gs[-1,0])
# ax10 = plt.subplot(gs[-1,-2])
#
# # method 3: easy to define structure
# ####################################
# # 格式在((),()...)中给定
# f,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,sharex=True,sharey=True)
#
# ax11.scatter([1,0],[0,1])
#
#
# plt.tight_layout()
# plt.show()




##########################################
##      course 15 plot in plot 图中图

# import matplotlib.pyplot as plt
#
# fig = plt.figure()
#
# x = [1,2,3,4,5,6,7]
# y = [1,3,5,6,7,8,3]
#
# left, bottom, width, height = 0.1,0.1,0.8,0.8
# ax1 = fig.add_axes([left,bottom,width,height])   # 增加图
# ax1.plot(x,y,'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')
#
# left, bottom, width, height = 0.2,0.6,0.25,0.25
# ax2 = fig.add_axes([left,bottom,width,height])  # 增加子图
# ax2.plot(y,x,'b')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_title('title inside 1')
#
#
# plt.axes([0.6,0.2,0.25,0.25])                   # 增加子图
# plt.plot(y[::-1],x ,'g')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('title inside 2')
#
# plt.show()



##########################################
##      course 16 secondary axis 次坐标

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(0,10,0.1)
# y1 = 0.05*x**2
# y2 = -1*y1
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()       # 将ax1关于x轴对称,将数据翻转到上面
#
# ax1.plot(x,y1,'g-')
# ax2.plot(x,y2,'b--')
#
# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1',color='g')
# ax2.set_ylabel('Y2',color='b')
#
# plt.show()



##########################################
##      course 17 animation 动画

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation    # 加载动画的模块

fig, ax = plt.subplots()

x = np.arange(0,2*np.pi,0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,interval=20,blit=False)  # 产生动画的函数
                      # 动画框在哪显示  # 动画函数    # 动画的总帧数 # 动画最开始的位置 # 频率：20ms, 是否更新整张图片：
plt.show()