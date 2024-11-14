import matplotlib.pyplot as plt

names = ['$split=1$', '$split=2$', '$split=4$', '$split=8$', '$split=16$', '$split=32$']
datas = [5987.824, 3854.793, 3155.391, 2735.202, 2446.925, 2239.504]

plt.figure()
plt.bar(names, datas)
plt.xlabel('Các trường hợp mô phỏng')
plt.ylabel('Thời gian đào tạo (giây)')
plt.savefig('test1.pdf')
plt.show()

