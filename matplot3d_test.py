import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 模拟数据
np.random.seed(0)  # 为了重现性
data = np.random.rand(10, 41)  # 二维数组，代表l的值

# 创建x轴和y轴的数据
length = np.linspace(0, 1, 41)  # 假设长度是0到1之间，分为41份
time_steps = np.arange(10)  # 10个时间步骤

# 将x轴和y轴的数据转换为网格形式
X, Y = np.meshgrid(length, time_steps)

# 创建图形和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维图形
surf = ax.plot_surface(X, Y, data, cmap='viridis')

# 添加标签
ax.set_xlabel('Length')
ax.set_ylabel('Time Step')
ax.set_zlabel('L')

# 显示图形
plt.show()
