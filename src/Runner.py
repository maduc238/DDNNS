from src.Logging import *
from src.Data import Data
from src.Model import Model
from src.Optim import Optim
from src.Utils import *
from src.Enumerated import *


class Runner:
    def __init__(self, model: Model, data: Data, opt: Optim):
        self.model = model
        self.data = data
        self.opt = opt
        self.time = 0.0
        self.trained_data = 0

        # format event queue: time, action, data_size,
        self.event_queue = []
        self.wait_queue = []

    def insert_dev_event(self, _id: str, time: float, action: str, name: str, data_size: int, flow: str):
        data = {'id': _id,
                'time': time,
                'action': action,
                'type': TYPE_DEVICE,
                'name': name,
                'data_size': data_size,
                'flow': flow}
        self.event_queue.append(data)
        self.event_queue.sort(key=timer)
        if time > self.time:
            self.time = time
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            log.info(f"Training: {data}")

    def insert_link_event(self, _id: str, time: float, action: str, _from: str, _to: str, data_size: int, flow: str):
        data = {'id': _id,
                'time': time,
                'action': action,
                'type': TYPE_LINK,
                'from': _from,
                'to': _to,
                'data_size': data_size,
                'flow': flow}
        self.event_queue.append(data)
        self.event_queue.sort(key=timer)
        if time > self.time:
            self.time = time
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            log.info(f"Training: {data}")

    def insert_event(self, data):
        self.event_queue.append(data)
        self.event_queue.sort(key=timer)
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            log.info(f"Training: {data}")

    def insert_wait(self, data):
        data['action'] = ACTION_WAIT
        self.wait_queue.append(data)
        self.wait_queue.sort(key=timer)
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            log.info(f"Training: {data}")

    def start(self):
        log.warn(f"Start run simulation at 'time': {self.time}")
        while self.trained_data <= self.data.num_data:
            # start initial with input data devices
            for micro_batch_data in generate_micro_batch(self.opt.batch_size, self.opt.num_micro_batch):
                for (a, data_size_a) in zip(self.model.input_devices,
                                            generate_micro_batch(micro_batch_data, len(self.model.input_devices))):
                    if self.model.devices_graph.nodes[a]["handler"]:
                        event = {'id': generate_id(), 'time': self.time, 'action': ACTION_START, 'type': TYPE_DEVICE,
                                 'name': a, 'data_size': data_size_a, 'flow': FLOW_FORWARD}
                        self.wait_queue.append(event)
                    else:
                        log.info(f"Insert input data size {data_size_a} on device {a} at 'time': {self.time}")
                        self.insert_dev_event(generate_id(), self.time, ACTION_START, a, int(data_size_a), FLOW_FORWARD)
                        self.model.devices_graph.nodes[a]["handler"] = True

            # handler event (event action means handler after this event)
            while len(self.event_queue) != 0:
                self.handler_event()

            # and now start backpropagation
            for micro_batch_data in generate_micro_batch(self.opt.batch_size, self.opt.num_micro_batch):
                for (a, data_size_a) in zip(self.model.output_device,
                                            generate_micro_batch(micro_batch_data, len(self.model.output_device))):
                    if self.model.devices_graph.nodes[a]["handler"]:
                        event = {'id': generate_id(), 'time': self.time, 'action': ACTION_START, 'type': TYPE_DEVICE,
                                 'name': a, 'data_size': data_size_a, 'flow': FLOW_BACKPROPAGATION}
                        self.wait_queue.append(event)
                    else:
                        log.info(f"Start backpropagation data size {data_size_a} on device {a} at 'time': {self.time}")
                        self.insert_dev_event(generate_id(), self.time, ACTION_START, a, int(data_size_a),
                                              FLOW_BACKPROPAGATION)
                        self.model.devices_graph.nodes[a]["handler"] = True

            while len(self.event_queue) != 0:
                self.handler_event()

            break       # for debug

        # check available queue -> get error
        if len(self.event_queue) != 0:
            log.error(f"Still have available queue process: {self.event_queue}")
            exit()

        if len(self.wait_queue) != 0:
            log.error(f"Still have available wait process: {self.wait_queue}")
            exit()

        log.info(f"Training time: {self.time} ms")

    def handler_event(self):
        # log.info(self.event_queue)
        # log.info(self.model.devices_graph.nodes(data=True))
        event = self.event_queue.pop(0)
        if event['type'] == TYPE_DEVICE:  # device event
            dev = event['name']
            time = event['time']
            data_size = event['data_size']
            node_dev = self.model.devices_graph.nodes[dev]

            # start training event
            if event['action'] == ACTION_START:
                training_rate = node_dev['training_rate']
                exec_time = self.model.get_exec_time(dev)
                time += training_rate * exec_time * (1 + generate_normal_random())
                self.insert_dev_event(event['id'], time, ACTION_END, dev, int(data_size), event['flow'])

            # end training event -> transfer
            elif event['action'] == ACTION_END or event['action'] == ACTION_WAIT:
                node_dev["handler"] = False         # unlock

                # handler wait process
                for wait in self.wait_queue:
                    if wait['type'] == TYPE_DEVICE:
                        if wait['name'] == dev:
                            wait['time'] = time
                            self.insert_event(wait)
                            self.wait_queue.remove(wait)
                            break

                next_dev = None
                if event['flow'] == FLOW_FORWARD and dev not in self.model.output_device:
                    next_dev = node_dev['next_dev']
                elif event['flow'] == FLOW_BACKPROPAGATION:
                    if dev not in self.model.input_devices:
                        next_dev = node_dev['prev_dev']
                    else:
                        # done training
                        self.trained_data += data_size

                if next_dev is not None:
                    for n_dev in next_dev:
                        # check busy link
                        if self.model.devices_graph[dev][n_dev]["handler"]:
                            event['time'] = time
                            self.insert_wait(event)
                            log.info(f"Training insert wait {event}")
                        # if not, continue flow
                        else:
                            self.model.devices_graph[dev][n_dev]["handler"] = True
                            self.insert_link_event(event['id'], time, ACTION_START, dev, n_dev,
                                                   int(data_size / len(next_dev)), event['flow'])

        # start transfer
        elif event['type'] == TYPE_LINK:  # link event
            from_dev = event['from']
            to_dev = event['to']
            time = event['time']
            data_size = event['data_size']

            # link start receive data
            if event['action'] == ACTION_START:
                data_rate_time = self.model.get_trans_time(from_dev, to_dev, event['flow'])
                # assume that root data gain 1_000 ms
                time += (data_rate_time * (1 + generate_normal_random()) * 1_000 * data_size) / \
                        (self.data.size * self.opt.batch_size)
                # unlock process
                self.model.devices_graph[from_dev][to_dev]["handler"] = False
                self.insert_link_event(event['id'], time, ACTION_END, from_dev, to_dev, int(data_size), event['flow'])

                # handler wait process
                for wait in self.wait_queue:
                    if wait['type'] == TYPE_LINK:
                        if (wait['from'] == from_dev and wait['to'] == to_dev) or \
                                (wait['from'] == to_dev and wait['to'] == from_dev):
                            wait['time'] = time
                            self.insert_event(wait)
                            self.wait_queue.remove(wait)

            # link done transmission
            elif event['action'] == ACTION_END or event['action'] == ACTION_WAIT:
                self.model.devices_graph[from_dev][to_dev]["handler"] = False
                # handler wait process
                for wait in self.wait_queue:
                    if wait['type'] == TYPE_LINK:
                        if (wait['from'] == from_dev and wait['to'] == to_dev) or \
                                (wait['from'] == to_dev and wait['to'] == from_dev):
                            wait['time'] = time
                            self.insert_event(wait)
                            self.wait_queue.remove(wait)
                            break

                # check busy dev
                if self.model.devices_graph.nodes[to_dev]["handler"]:
                    event['time'] = time
                    self.insert_wait(event)
                    log.info(f"Training insert wait {event}")
                # if not, continue flow
                else:
                    self.model.devices_graph.nodes[to_dev]["handler"] = True
                    self.insert_dev_event(event['id'], time, ACTION_START, to_dev, int(data_size), event['flow'])

        if len(self.event_queue) == 0 and len(self.wait_queue) != 0:
            wait_event = self.wait_queue.pop(0)
            wait_event['time'] = self.time
            self.insert_event(wait_event)
            