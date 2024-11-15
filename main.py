import networkx as nx
import time

from src.Logging import *
from src.Data import Data
from src.Model import Model
from src.Optim import Optim
from src.Runner import Runner
from src.Plot import Plot

if __name__ == '__main__':
    log.info('DDNN simulation')
    start_time = time.time()

    g = nx.Graph(name="Device Connection Graph")
    g.add_node("A", training_rate=10)
    g.add_node("B", training_rate=10)
    g.add_node("C", training_rate=10)
    g.add_node("D", training_rate=10)
    g.add_edge("A", "B", trans_rate=4)
    g.add_edge("B", "C", trans_rate=4)
    g.add_edge("C", "D", trans_rate=4)
    log.info(f"Set graph {g.nodes()}")

    model = Model()

    model.set_input_device(["A"])
    model.set_output_device(["D"])
    model.set_layer_group(g,)
    for n in g.nodes(data=True):
        log.info(n)
    for e in g.edges(data=True):
        log.info(e)

    data = Data(128 * 128 * 3, 50_000)

    opt = Optim(batch_size=128, num_micro_batch=4)

    run = Runner(model, data, opt)
    run.set_test_flow()
    run.start()

    stop_time = time.time()
    log.info(f"Simulation running for {stop_time - start_time} seconds")

    # log.info(run.log_event)
    plot = Plot(g, run)
    plot.plot_time()
