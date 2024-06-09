import networkx as nx
import matplotlib.pyplot as plt

from src.Runner import Runner


class Plot:
    def __init__(self, graph: nx.Graph, run: Runner):
        self.graph = graph
        self.run = run
        self.log_event = run.log_event

    def plot_time(self):
        if not self.run.test_flow:
            return
        data = {}
        for i in self.graph.nodes:
            data[i] = []

        # insert data
        for e in self.log_event:
            if e['type'] == 'device':
                data[e['name']].append(e['time'])

        # plot
        fig, ax = plt.subplots()

        for i, name in enumerate(self.graph.nodes):
            timer = data[name]
            for j in range(len(timer) // 2):
                ax.broken_barh([(timer[2 * j], timer[2 * j + 1] - timer[2 * j])], (5 * i, 1))

        ax.set_xlabel('seconds')
        ax.set_yticks([5 * i for i in range(len(self.graph.nodes))], labels=[i for i in self.graph.nodes])
        plt.show()
