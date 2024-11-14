import matplotlib.pyplot as plt
import numpy as np

comms = ('$(10-10-10-10)$', '$(8-10-12-10)$', '$(6-12-18-4)$')
split_cases = {
    '$Split = 2$': (4638.238, 4724.216, 5185.653),
    '$Split = 4$': (3808.155, 3800.330, 4295.968),
    '$Split = 8$': (3295.413, 3233.635, 3752.535),
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

plt.xlabel('Các trường hợp mô phỏng với sự chênh lệch tài nguyên khác nhau')
ax.set_ylabel('Thời gian đào tạo (giây)')
ax.set_xticks(x + width, comms)
ax.legend(loc='upper left', ncols=3)
plt.savefig('test3.pdf')
plt.show()
