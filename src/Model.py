import networkx as nx

from src.Logging import *
from src.Enumerated import *


def insert_layer(graph: nx.Graph, lists, group):
    for rank, a in enumerate(lists[1:-1]):
        if graph.nodes[a]['start_layer'] is None or graph.nodes[a]['end_layer'] is None:
            graph.nodes[a]['start_layer'] = group[rank]
            graph.nodes[a]['end_layer'] = group[rank + 1] - 1

        if lists[rank] not in graph.nodes[a]['prev_dev']:
            graph.nodes[a]['prev_dev'].append(lists[rank])

        if lists[rank + 2] not in graph.nodes[a]['next_dev']:
            graph.nodes[a]['next_dev'].append(lists[rank + 2])


class Model:
    def __init__(self, first_layer_mem: float, first_layer_exec_time: float):  # MB - ms
        self.neural_mem = []
        self.neural_exec_time = []
        self.neural_inter_layer_size = []

        self.num_layer = 1
        self.neural_mem.append(first_layer_mem)
        self.neural_exec_time.append(first_layer_exec_time)

        self.devices_graph = None
        self.output_device = None
        self.input_devices = None

    def append(self, layer_mem: float, layer_exec_time: float, inter_layer_size: int):  # MB - ms - size
        self.neural_mem.append(layer_mem)
        self.neural_exec_time.append(layer_exec_time)
        self.neural_inter_layer_size.append(inter_layer_size)
        self.num_layer += 1

    def get_num_layer(self):
        return self.num_layer

    def set_input_device(self, input_devices):
        self.input_devices = input_devices

    def set_output_device(self, output_device):
        self.output_device = output_device

    def set_layer_group(self, graph: nx.Graph, cut_group):
        """

        :param graph: Devices graph
        :param cut_group: List of cut point index
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
                node[a]['start_layer'] = None
                node[a]['end_layer'] = None
                node[a]['next_dev'] = []
                node[a]['prev_dev'] = []
                node[a]['handler'] = False
                node[a]['mem_usage'] = node[a]['idle_mem']

        for a in self.input_devices:
            node[a]['start_layer'] = 0
            node[a]['end_layer'] = cut_group[0] - 1
            node[a]['next_dev'] = []
            node[a]['handler'] = False
            node[a]['mem_usage'] = node[a]['idle_mem']

            for b in self.devices_graph.neighbors(a):
                if b not in node[a]['next_dev']:
                    node[a]['next_dev'].append(b)

        for a in self.output_device:
            node[a]['start_layer'] = cut_group[-1]
            node[a]['end_layer'] = self.num_layer - 1
            node[a]['prev_dev'] = []
            node[a]['handler'] = False
            node[a]['mem_usage'] = node[a]['idle_mem']
            for b in self.devices_graph.neighbors(a):
                if b not in node[a]['prev_dev']:
                    node[a]['prev_dev'].append(b)

        # set link property
        for e1, e2 in self.devices_graph.edges:
            self.devices_graph[e1][e2]['handler'] = False

        if len(cut_group) < 2:
            return

        # other devices
        for i in self.input_devices:
            for o in self.output_device:
                for path in nx.all_simple_paths(self.devices_graph, source=i, target=o):
                    if len(path) == len(cut_group) + 1:
                        insert_layer(self.devices_graph, path, cut_group)

        # check all available memory
        mem_operation = True
        for dev in self.devices_graph.nodes():
            mem_req = self.get_mem_requirement(dev)
            if node[dev]['mem_usage'] + mem_req > node[dev]['mem_size']:
                log.error(f"Device {dev} can not operate with memory requirement {node[dev]['mem_usage'] + mem_req} - "
                          f"memory size {node[dev]['mem_size']}")
                mem_operation = False

        if not mem_operation:
            log.error("False while allocate memory, stop simulation")
            exit()

    def get_exec_time(self, name: str):
        from_layer = self.devices_graph.nodes[name]['start_layer']
        to_layer = self.devices_graph.nodes[name]['end_layer']
        exec_time = 0.0
        for i in range(from_layer, to_layer + 1):
            exec_time += self.neural_exec_time[i]

        return exec_time

    def get_trans_time(self, _from: str, _to: str, flow: str):
        sample_size = 0.0
        trans_rate = self.devices_graph[_from][_to]['trans_rate']
        if flow == FLOW_FORWARD:
            sample_size = self.neural_inter_layer_size[self.devices_graph.nodes[_from]['end_layer']]
        elif flow == FLOW_BACKPROPAGATION:
            sample_size = self.neural_inter_layer_size[self.devices_graph.nodes[_to]['end_layer']]
        return trans_rate * sample_size

    def get_mem_requirement(self, name: str):
        from_layer = self.devices_graph.nodes[name]['start_layer']
        to_layer = self.devices_graph.nodes[name]['end_layer']
        log.debug(f"From layer {from_layer} to layer {to_layer}")
        mem_rate = self.devices_graph.nodes[name]['mem_rate']
        mem_req = 0.0
        for i in range(from_layer, to_layer + 1):
            mem_req += self.neural_mem[i]

        return mem_req * mem_rate
