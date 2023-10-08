from src.Utils import *


def myFunc(e):
    return e['year']


if __name__ == "__main__":
    data = [1,2,3,4,5]
    sum = 0
    for a in range(0,2+1):
        sum += data[a]

    print(generate_normal_random())