import matplotlib.pyplot as plt
import numpy as np

comms = ('$t_{comm} = 2$ ms', '$t_{comm} = 4$ ms', '$t_{comm} = 8$ ms', '$t_{comm} = 16$ ms')
split_cases = {
    '$Split = 2$': (4242.943, 4639.695, 5444.978, 7089.485),
    '$Split = 4$': (3514.953, 3828.743, 4416.359, 5765.017),
    '$Split = 8$': (3053.193, 3295.396, 3780.635, 4926.328),
}

x = np.arange(len(comms))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in split_cases.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

plt.xlabel('Các trường hợp mô phỏng với độ trễ mạng khác nhau')
ax.set_ylabel('Thời gian đào tạo (giây)')
ax.set_xticks(x + width, comms)
ax.legend(loc='upper left', ncols=3)
plt.savefig('test2.pdf')
plt.show()
