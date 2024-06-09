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
    g.add_node("A", training_rate=100)
    g.add_node("B", training_rate=200)
    g.add_node("C", training_rate=100)
    g.add_node("D", training_rate=200)
    g.add_edge("A", "B", trans_rate=1)
    g.add_edge("B", "C", trans_rate=1)
    g.add_edge("A", "D", trans_rate=1)
    g.add_edge("D", "C", trans_rate=1)
    log.info(f"Set graph {g.nodes()}")

    model = Model()

    model.set_input_device(["A"])
    model.set_output_device(["C"])
    model.set_layer_group(g,)        # cut at layers
    for n in g.nodes(data=True):
        log.info(n)
    for e in g.edges(data=True):
        log.info(e)

    data = Data(128 * 128 * 3, 50_000)

    opt = Optim(batch_size=128, num_micro_batch=128)

    run = Runner(model, data, opt)
    run.set_test_flow()
    run.start()

    # TODO: add energy consumption

    stop_time = time.time()
    log.info(f"Simulation running for {stop_time - start_time} seconds")

    # log.info(run.log_event)
    plot = Plot(g, run)
    plot.plot_time()
