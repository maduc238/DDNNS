from numpy import random


def generate_normal_random():
    result = random.normal(scale=0.1)
    if result >= -1.0:
        return result
    else:
        return 0.0


def timer(e):
    return e['time']
