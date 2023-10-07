import logging as log
import networkx as nx

from src.Data import Data
from src.Model import Model
from src.Optim import Optim

if __name__ == '__main__':
    log.basicConfig(level=log.INFO, format='%(levelname)s - %(message)s')

    log.info('Start simulation')

    g = nx.Graph(name="Device Connection Graph")
    g.add_node("A", mem_size=2048, idle_mem=100, mem_rate=1, training_rate=1)
    g.add_node("B", mem_size=4096, idle_mem=100, mem_rate=1, training_rate=1.1)
    g.add_node("C", mem_size=2048, idle_mem=100, mem_rate=0.5, training_rate=3.6)
    g.add_edge("A", "B", trans_rate=3.2)
    g.add_edge("B", "C", trans_rate=2.5)
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

    model.set_input_device(["A"])
    model.set_output_device(["C"])
    model.set_layer_group(g, [3, 5])
    log.info(list(g.nodes(data=True)))

    # TODO: create data
    data = Data(128 * 128 * 3, 50_000)

    # TODO: create optimal
    opt = Optim(batch_size=128)

    # TODO: run training
    # run = Training()
    # TODO: run forward
    # TODO: run backward
