import numpy as np
import matplotlib.pyplot as plt

data_str = "output_1.data"
print(data_str)

data = np.loadtxt(data_str)

x_iter = data[:, 0]
y_correct = data[:, 1]
y_error = data[:, 2]
time = data[:, 3]


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(11, 7)

ax[0].set(xlabel="Iteration", ylabel="Cost")
ax[0].grid(True, 'major', 'y')
ax[0].plot(x_iter, y_error, color='#DC143C', label='Error rate')
ax[0].legend(loc='center right')

ax[1].set(xlabel="Iteration", ylabel="Percent, %")
ax[1].grid(True, 'major', 'y')
ax[1].plot(x_iter, y_correct, color='c', label='Accuracy')
ax[1].legend(loc='center right')

plt.savefig('out.png', dpi=600)
plt.show()
