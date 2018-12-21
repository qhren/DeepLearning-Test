import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 对于单样本而言，假设数据的输出有两个维度
Expect_Result = np.array([0, 1])
output_data = []
k = 0
for i in range(1, 100):  # range 只能生成整数列表，不同于Lin_space
    for j in range(1, 100):
        output_data.append([i/100, j/100])
Square_e = [np.linalg.norm((np.array(i)-Expect_Result))**2 for i in output_data]
Crossentrophy_e = [np.sum(np.nan_to_num(-Expect_Result*np.log(i)-(1-Expect_Result)*np.log(1-np.array(i)))) for i in output_data]
fig_1 = plt.figure(1)
ax = Axes3D(fig_1)
plt.ion()
X, Y = np.meshgrid(np.arange(0.01, 1, 0.01), np.arange(0.01, 1, 0.01))
ax.plot_surface(X, Y, np.array(Square_e).reshape((99, 99)))
plt.title('Square_error Surface')

fig_2 = plt.figure(2)
ax = Axes3D(fig_2)
ax.plot_surface(X, Y, np.array(Crossentrophy_e).reshape((99, 99)))
plt.title('Crossentrophy Surface')

fig_3 = plt.figure(3)
fig_3.add_subplot(2, 1, 1)
plt.contourf(X, Y, np.array(Square_e).reshape((99, 99)), 12, cmap=plt.cm.get_cmap(name='hot'))
C1 = plt.contour(X, Y, np.array(Square_e).reshape((99, 99)), 12, colors='black', linewidths=0.5)
plt.clabel(C1, inline=True, fontsize=12)
plt.title('Square_error Contour')
fig_3.add_subplot(2, 1, 2)
plt.contourf(X, Y, np.array(Crossentrophy_e).reshape((99, 99)), 12, cmap=plt.cm.get_cmap(name='hot'))
C2 = plt.contour(X, Y, np.array(Crossentrophy_e).reshape((99, 99)), 12, colors='black', linewidths=0.5)
plt.clabel(C2, inline=True, fontsize=12)
plt.title('Cross_entrophy Contour')
plt.ioff()
plt.show()








