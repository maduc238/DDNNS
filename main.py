import networkx as nx
import time

from src.Logging import *
from src.Data import Data
from src.Model import Model
from src.Optim import Optim
from src.Runner import Runner

if __name__ == '__main__':
    log.info('DDNN simulation')
    start_time = time.time()

    g = nx.Graph(name="Device Connection Graph")
    g.add_node("A", mem_size=2048, idle_mem=100, mem_rate=1, training_rate=1)
    g.add_node("B", mem_size=4096, idle_mem=100, mem_rate=1, training_rate=1.1)
    g.add_node("C", mem_size=2048, idle_mem=100, mem_rate=0.5, training_rate=3.6)
    g.add_node("D", mem_size=4096, idle_mem=100, mem_rate=1, training_rate=1.1)
    g.add_edge("A", "B", trans_rate=1)
    g.add_edge("B", "C", trans_rate=1.5)
    g.add_edge("A", "D", trans_rate=1)
    g.add_edge("D", "C", trans_rate=1.5)
    log.info(f"Set graph {g.nodes()}")

    model = Model(first_layer_mem=100, first_layer_exec_time=200)
    model.append(90, 600, 128 * 128 * 8)
    model.append(120, 900, 256 * 256 * 4)
    model.append(100, 500, 128 * 128 * 4)
    model.append(80, 400, 256 * 256 * 4)
    model.append(110, 700, 512 * 512 * 2)
    model.append(70, 200, 128 * 128 * 1)
    model.append(40, 100, 64 * 64)
    log.info(f"Set neural with num_layer = {model.get_num_layer()}")
    log.info(f"Neural layers memory: {model.neural_mem}")
    log.info(f"Neural layers execution time: {model.neural_exec_time}")
    log.info(f"Neural inter layer data size: {model.neural_inter_layer_size}")

    model.set_input_device(["A"])
    model.set_output_device(["C"])
    model.set_layer_group(g, [3, 5])        # cut at layers
    log.info(list(g.nodes(data=True)))
    log.info(list(g.edges(data=True)))

    data = Data(128 * 128 * 3, 50_000)

    opt = Optim(batch_size=128, num_micro_batch=2)

    run = Runner(model, data, opt)
    run.start()

    # TODO: add tree connection
    # TODO: add energy consumption

    stop_time = time.time()
    log.info(f"Simulation running for {stop_time - start_time} seconds")
