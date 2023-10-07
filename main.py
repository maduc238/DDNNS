import logging as log
import networkx as nx

from src.Device import Device
from src.Neural import Neural

log.basicConfig(level=log.INFO, format='%(levelname)s - %(message)s')

if __name__ == '__main__':
    log.info('Start simulation')
    # TODO: create devices (mem, speed rate)
    dev_a = Device("A", 2048, 1, 1)
    dev_b = Device("B", 4096, 1, 1.1)
    dev_c = Device("C", 2048, 0.5, 3.6)
    log.info("Set device")

    # TODO: create graph and network connected bandwidth
    g = nx.Graph()
    g.add_node(dev_a)
    g.add_node(dev_b)
    g.add_node(dev_c)
    g.add_edge(dev_a, dev_b, weight=3.2)
    g.add_edge(dev_b, dev_c, weight=2.5)
    log.info("Set graph")

    # TODO: create neural network (time, memory alloc)
    neu = Neural(first_layer_mem=100, first_layer_exec_time=200)
    log.info("Set neural")

    # TODO: create data
    # TODO: use with graph, generate
    # TODO: create optimal
