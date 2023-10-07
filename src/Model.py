import networkx as nx


def insert_layer(graph: nx.Graph, lists, group):
    for rank, a in enumerate(lists[1:-1]):
        if graph.nodes[a]['start_layer'] is None or graph.nodes[a]['end_layer'] is None:
            graph.nodes[a]['start_layer'] = group[rank]
            graph.nodes[a]['end_layer'] = group[rank + 1] - 1

        if lists[rank] not in graph.nodes[a]['prev_dev']:
            graph.nodes[a]['prev_dev'].append(lists[rank])

        if lists[rank+1] not in graph.nodes[a]['next_dev']:
            graph.nodes[a]['next_dev'].append(lists[rank + 2])


class Model:
    def __init__(self, first_layer_mem: float, first_layer_exec_time: float):  # MB - ms
        self.neural_mem = []
        self.neural_exec_time = []
        self.neural_inter_layer_size = []

        self.num_layer = 1
        self.neural_mem.append(first_layer_mem)
        self.neural_exec_time.append(first_layer_exec_time)

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
        if (self.input_devices is None) or (self.output_device is None):
            raise False
        for a in graph.nodes():
            if a not in self.input_devices and a not in self.output_device:
                graph.nodes[a]['start_layer'] = None
                graph.nodes[a]['end_layer'] = None
                graph.nodes[a]['next_dev'] = []
                graph.nodes[a]['prev_dev'] = []

        for a in self.input_devices:
            graph.nodes[a]['start_layer'] = 0
            graph.nodes[a]['end_layer'] = cut_group[0] - 1
            graph.nodes[a]['next_dev'] = []
            for b in graph.neighbors(a):
                graph.nodes[a]['next_dev'].append(b)

        for a in self.output_device:
            graph.nodes[a]['start_layer'] = cut_group[-1]
            graph.nodes[a]['end_layer'] = self.num_layer - 1
            graph.nodes[a]['prev_dev'] = []
            for b in graph.neighbors(a):
                graph.nodes[a]['prev_dev'].append(b)

        if len(cut_group) < 2:
            return

        # other devices
        for i in self.input_devices:
            for o in self.output_device:
                for path in nx.all_simple_paths(graph, source=i, target=o):
                    if len(path) == len(cut_group) + 1:
                        insert_layer(graph, path, cut_group)
