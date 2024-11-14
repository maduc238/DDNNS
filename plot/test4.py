import matplotlib.pyplot as plt
import numpy as np

comms = ('3 thiết bị', '4 thiết bị', '5 thiết bị', '6 thiết bị')
split_cases = {
    '$Split = 2$': (4677.082, 4639.168, 4692.276, 4807.316),
    '$Split = 4$': (4196.368, 3814.985, 3617.286, 3535.256),
    '$Split = 8$': (3788.632, 3292.941, 2979.251, 2822.514),
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

plt.xlabel('Các trường hợp mô phỏng với số lượng thiết bị khác nhau')
ax.set_ylabel('Thời gian đào tạo (giây)')
ax.set_xticks(x + width, comms)
ax.set_ylim(0, 5500)
ax.legend(loc='upper left', ncols=3)
plt.savefig('test4.pdf')
plt.show()
