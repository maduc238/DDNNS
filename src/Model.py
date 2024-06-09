import networkx as nx

from src.Logging import *
from src.Enumerated import *
from src.Utils import *


def insert_layer(graph: nx.Graph, lists):
    for rank, a in enumerate(lists[1:-1]):
        if lists[rank] not in graph.nodes[a]['prev_dev']:
            graph.nodes[a]['prev_dev'].append(lists[rank])

        if lists[rank + 2] not in graph.nodes[a]['next_dev']:
            graph.nodes[a]['next_dev'].append(lists[rank + 2])


class Model:
    def __init__(self):
        self.devices_graph = None
        self.output_device = None
        self.input_devices = None

    def set_input_device(self, input_devices):
        self.input_devices = input_devices

    def set_output_device(self, output_device):
        self.output_device = output_device

    def set_dev_basic_property(self, dev):
        node = self.devices_graph.nodes
        node[dev]['id'] = name_generate_id(str(dev))
        node[dev]['handler'] = False
        node[dev]['last_lock'] = 0.0
        node[dev]['last_unlock'] = 0.0

    def set_layer_group(self, graph: nx.Graph):
        """

        :param graph: Devices graph
        :return:
        """
        self.devices_graph = graph
        if (self.input_devices is None) or (self.output_device is None):
            log.error("Devices data is None")
            exit()

        # initial devices data
        node = self.devices_graph.nodes
        for a in self.devices_graph.nodes():
            if a not in self.input_devices and a not in self.output_device:
                node[a]['next_dev'] = []
                node[a]['prev_dev'] = []
                self.set_dev_basic_property(a)

        for a in self.input_devices:
            node[a]['next_dev'] = []
            self.set_dev_basic_property(a)

            for b in self.devices_graph.neighbors(a):
                if b not in node[a]['next_dev']:
                    node[a]['next_dev'].append(b)

        for a in self.output_device:
            node[a]['prev_dev'] = []
            self.set_dev_basic_property(a)
            for b in self.devices_graph.neighbors(a):
                if b not in node[a]['prev_dev']:
                    node[a]['prev_dev'].append(b)

        # set link property
        for e1, e2 in self.devices_graph.edges:
            self.devices_graph[e1][e2]['id'] = name_generate_id(str(e1)+"/"+str(e2))
            self.devices_graph[e1][e2]['handler'] = False
            self.devices_graph[e1][e2]['last_lock'] = 0.0
            self.devices_graph[e1][e2]['last_unlock'] = 0.0

        # other devices
        for i in self.input_devices:
            for o in self.output_device:
                for path in nx.all_simple_paths(self.devices_graph, source=i, target=o):
                    insert_layer(self.devices_graph, path)

    def get_exec_time(self, name: str, data_size, flow):
        if flow == FLOW_FORWARD:
            return self.devices_graph.nodes[name]['training_rate'] * data_size * (1 + generate_normal_random())
        else:
            return self.devices_graph.nodes[name]['training_rate'] * data_size * 2 * (1 + generate_normal_random())

    def get_trans_time(self, _from: str, _to: str, data_size, flow):
        return self.devices_graph[_from][_to]['trans_rate'] * data_size * (1 + generate_normal_random())
