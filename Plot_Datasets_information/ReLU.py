# 绘制ReLu函数及其导数图像
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.figure(1)
##################################################
def relu(x):
    ans = max(0, x)

    return ans


def d_relu(x):
    if x<=0:
        return 0
    else:
        return 1
##################################################
x = [-6,-5,-4,-3,-2,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,2,3,4,5,6]
y_1 = []
y_2 = []
##################################################
for i in range(len(x)):
    y_1.append(relu(x[i]))
    y_2.append(d_relu(x[i]))
print(y_1)
print(y_2)
##################################################

ax_1 = plt.subplot(121)
plt.plot(x,y_1,color="r",linestyle='--',label='f(x)')

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("ReLU function")
plt.legend()
##################################################

ax_2 = plt.subplot(122)
plt.plot(x,y_2,color="b",linestyle='-.',label="f'(x)")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("ReLU derivative")
plt.legend()
##################################################
plt.show()
fig.savefig('relu.png',dpi=900, format='png')
