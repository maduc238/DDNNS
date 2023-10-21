from src.Logging import *
from src.Data import Data
from src.Model import Model
from src.Optim import Optim
from src.Utils import *
from src.Enumerated import *

import copy


class Runner:
    def __init__(self, model: Model, data: Data, opt: Optim):
        self.model = model
        self.data = data
        self.opt = opt

        self.test_flow = False
        self.test_runner_event = {}

        self.time = 0.0
        self.trained_data = 0

        self.event_queue = []
        self.wait_queue = []

    def set_test_flow(self):
        self.test_flow = True

    def append_runner_event(self, _id, time, action):
        if _id not in self.test_runner_event:
            self.test_runner_event[_id] = []
        self.test_runner_event[_id].append({'time': time, 'action': action})

    def check_test_flow(self):
        for _id in self.test_runner_event:
            event = self.test_runner_event[_id]
            event.sort(key=timer)
            if len(event) % 2 != 0:
                log.warn(f"Test {_id}: Error!")

            for i in range(len(event) - 1):
                if event[i]['time'] == event[i+1]['time'] and \
                        event[i]['action'] == ACTION_START and \
                        event[i+1]['action'] == ACTION_END:
                    event.insert(i+1, event.pop(i))
                    log.info(f"Check: {event[i]}, {event[i+1]}")
            log.info(f"Test {_id}: {event}")

            for i in range(len(event) // 2):
                if event[2*i]['action'] == ACTION_END and \
                        event[2*i+1]['action'] == ACTION_START:
                    log.warn(f"Test {_id}: Got conflict at event {event[2*i]} and {event[2*i+1]}")

    def insert_dev_event(self, _id: str, time: float, action: str, name: str, data_size: int, flow: str):
        data = {'id': _id,
                'time': time,
                'action': action,
                'type': TYPE_DEVICE,
                'name': name,
                'data_size': data_size,
                'flow': flow,
                'send_to': []}
        self.event_queue.append(data)
        self.update_time()
        self.event_queue.sort(key=timer)
        if time > self.time:
            self.time = time
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            if self.test_flow:
                _id = self.model.devices_graph.nodes[data['name']]['id']
                self.append_runner_event(_id, time, action)
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
        self.update_time()
        self.event_queue.sort(key=timer)
        if time > self.time:
            self.time = time
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            if self.test_flow:
                _id = self.model.devices_graph[data['from']][data['to']]['id']
                self.append_runner_event(_id, time, action)
            log.info(f"Training: {data}")

    def insert_event(self, data):
        # update time
        self.event_queue.append(data)
        self.update_time()
        self.event_queue.sort(key=timer)

        if not self.all_event_is_end():
            # TODO: push start event to top
            ...

        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            if self.test_flow:
                if data['type'] == TYPE_DEVICE:
                    _id = self.model.devices_graph.nodes[data['name']]['id']
                    self.append_runner_event(_id, data['time'], data['action'])
                elif data['type'] == TYPE_LINK:
                    _id = self.model.devices_graph[data['from']][data['to']]['id']
                    self.append_runner_event(_id, data['time'], data['action'])
            log.info(f"Training: {data}")

    def update_time(self):
        for data in self.event_queue:
            if data['type'] == TYPE_DEVICE:
                # of next link
                _from = data['name']
                if data['flow'] == FLOW_FORWARD and _from not in self.model.output_device:
                    _to = self.model.devices_graph.nodes[_from]['next_dev']
                    time_temp = self.model.devices_graph[_from][_to[0]]["last_unlock"]
                    for _ in _to:
                        if time_temp > self.model.devices_graph[_from][_]["last_unlock"]:
                            time_temp = self.model.devices_graph[_from][_]["last_unlock"]
                    if data['time'] < time_temp:
                        data['time'] = time_temp
                elif data['flow'] == FLOW_BACKPROPAGATION and _from not in self.model.input_devices:
                    _to = self.model.devices_graph.nodes[_from]['prev_dev']
                    time_temp = self.model.devices_graph[_from][_to[0]]["last_unlock"]
                    for _ in _to:
                        if time_temp > self.model.devices_graph[_from][_]["last_unlock"]:
                            time_temp = self.model.devices_graph[_from][_]["last_unlock"]
                    if data['time'] < time_temp:
                        data['time'] = time_temp

            elif data['type'] == TYPE_LINK:
                _to = data['to']
                time_temp = self.model.devices_graph.nodes[_to]["last_unlock"]
                log.debug(f"time = {data['time']}, temp = {time_temp}")
                if data['time'] < time_temp:
                    data['time'] = time_temp
                    log.debug(f"Check = {data}")

    def all_event_is_end(self):
        if len(self.event_queue) <= 1:
            return False
        for e in self.event_queue:
            if e['action'] != ACTION_END and e['action'] != ACTION_WAIT:
                return False
        for i in range(len(self.event_queue)):
            if self.event_queue[i]['type'] == TYPE_LINK:
                for j in range(i + 1, len(self.event_queue)):
                    if self.event_queue[j]['type'] == TYPE_DEVICE:
                        # check dependency
                        if self.event_queue[i]['to'] == self.event_queue[j]['name']:
                            # change queue position
                            event_temp = self.event_queue[i]
                            self.event_queue[i] = self.event_queue[j]
                            self.event_queue[j] = event_temp
                            return True
            if self.event_queue[i]['type'] == TYPE_DEVICE:
                for j in range(i + 1, len(self.event_queue)):
                    if self.event_queue[j]['type'] == TYPE_LINK:
                        # check dependency
                        link = [self.event_queue[j]['from'], self.event_queue[j]['to']]
                        for next_dev in self.event_queue[i]['send_to']:
                            if self.event_queue[i]['name'] in link \
                                    and next_dev in link:
                                # change queue position
                                event_temp = self.event_queue[i]
                                self.event_queue[i] = self.event_queue[j]
                                self.event_queue[j] = event_temp
                                return True
        return True

    def insert_wait(self, data):
        data['action'] = ACTION_WAIT
        self.wait_queue.append(data)
        self.wait_queue.sort(key=timer)
        if data['action'] == ACTION_WAIT:
            log.debug(f"Training: {data}")
        else:
            log.info(f"Training: {data}")

    def start(self):
        log.info(f"Start run simulation at 'time': {self.time}")
        while self.trained_data <= self.data.num_data:
            # start initial with input data devices
            for micro_batch_data in generate_micro_batch(self.opt.batch_size, self.opt.num_micro_batch):
                for (a, data_size_a) in zip(self.model.input_devices,
                                            generate_micro_batch(micro_batch_data, len(self.model.input_devices))):
                    if self.model.devices_graph.nodes[a]["handler"]:
                        event = {'id': generate_id(), 'time': self.time, 'action': ACTION_START, 'type': TYPE_DEVICE,
                                 'name': a, 'data_size': data_size_a, 'flow': FLOW_FORWARD, 'send_to': []}
                        self.wait_queue.append(event)
                    else:
                        log.info(f"Insert input data size {data_size_a} on device {a} at 'time': {self.time}")
                        self.insert_dev_event(generate_id(), self.time, ACTION_START, a, int(data_size_a), FLOW_FORWARD)
                        self.model.devices_graph.nodes[a]["handler"] = True
                        if self.model.devices_graph.nodes[a]["last_lock"] < self.time:
                            self.model.devices_graph.nodes[a]["last_lock"] = self.time

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
                        if self.model.devices_graph.nodes[a]["last_lock"] < self.time:
                            self.model.devices_graph.nodes[a]["last_lock"] = self.time

            while len(self.event_queue) != 0:
                self.handler_event()

            if self.test_flow:
                log.info("Start check test flow")
                log.debug(self.test_runner_event)
                self.check_test_flow()
                break  # for debug

        # check available queue -> get error
        if len(self.event_queue) != 0:
            log.error(f"Still have available queue process: {self.event_queue}")
            exit()

        if len(self.wait_queue) != 0:
            log.error(f"Still have available wait process: {self.wait_queue}")
            exit()

        log.info(f"Training time: {self.time} ms")

    def handler_event(self):
        log.info(self.event_queue)
        log.debug(self.wait_queue)
        log.debug(self.model.devices_graph.nodes(data=True))
        event = self.event_queue.pop(0)
        if event['type'] == TYPE_DEVICE:  # device event
            dev = event['name']
            time = event['time']
            data_size = event['data_size']
            node_dev = self.model.devices_graph.nodes[dev]

            # start training event
            if event['action'] == ACTION_START:
                node_dev["handler"] = True
                training_rate = node_dev['training_rate']
                exec_time = self.model.get_exec_time(dev)
                time += training_rate * exec_time * (1 + generate_normal_random())
                self.insert_dev_event(event['id'], time, ACTION_END, dev, int(data_size), event['flow'])

                if node_dev["last_unlock"] < time:
                    node_dev["last_unlock"] = time
                # if (event['flow'] == FLOW_FORWARD and dev in self.model.output_device) or \
                #         (event['flow'] == FLOW_BACKPROPAGATION and dev in self.model.input_devices):
                #     if node_dev["last_unlock"] < time:
                #         node_dev["last_unlock"] = time

            # end training event -> transfer
            elif event['action'] == ACTION_END or event['action'] == ACTION_WAIT:
                node_dev["handler"] = False  # unlock
                if node_dev["last_unlock"] < time:
                    node_dev["last_unlock"] = time

                # handler wait process
                for wait in self.wait_queue:
                    if wait['type'] == TYPE_DEVICE:
                        if wait['name'] == dev:
                            wait['time'] = time
                            if wait['action'] == ACTION_START:
                                if wait['flow'] == FLOW_FORWARD:
                                    log.info(f"Insert input data size {wait['data_size']} on device {wait['name']} at "
                                             f"'time': {wait['time']}")
                                elif wait['flow'] == FLOW_BACKPROPAGATION:
                                    log.info(f"Start backpropagation data size {wait['data_size']} on device "
                                             f"{wait['name']} at 'time': {wait['time']}")
                            self.insert_event(wait)
                            self.wait_queue.remove(wait)
                            break

                next_dev = None
                if event['flow'] == FLOW_FORWARD:
                    if dev not in self.model.output_device:
                        next_dev = node_dev['next_dev']
                elif event['flow'] == FLOW_BACKPROPAGATION:
                    if dev not in self.model.input_devices:
                        next_dev = node_dev['prev_dev']
                    else:
                        # done training
                        self.trained_data += data_size
                log.debug(f"Next device = {next_dev}")

                if next_dev is not None:
                    if event['action'] == ACTION_END:
                        event['send_to'] = copy.deepcopy(next_dev)
                    ne_dev = copy.deepcopy(event['send_to'])
                    need_resend = False  # avoid resend
                    for n_dev in ne_dev:
                        log.debug(f"Dev {dev} send to dev {n_dev}")
                        # check busy link
                        if self.model.devices_graph[dev][n_dev]["handler"]:
                            need_resend = True
                        # if not, continue flow
                        else:
                            self.model.devices_graph[dev][n_dev]["handler"] = True
                            if self.model.devices_graph[dev][n_dev]["last_lock"] < time:
                                self.model.devices_graph[dev][n_dev]["last_lock"] = time
                            self.insert_link_event(event['id'], time, ACTION_START, dev, n_dev,
                                                   int(data_size / len(next_dev)), event['flow'])
                            event['send_to'].remove(n_dev)

                    if need_resend:
                        event['time'] = time
                        self.insert_wait(event)
                        log.info(f"Training insert wait {event}")

        # start transfer
        elif event['type'] == TYPE_LINK:  # link event
            from_dev = event['from']
            to_dev = event['to']
            time = event['time']
            data_size = event['data_size']

            # link start receive data
            if event['action'] == ACTION_START:
                self.model.devices_graph[from_dev][to_dev]["handler"] = True
                data_rate_time = self.model.get_trans_time(from_dev, to_dev, event['flow'])
                # assume that root data gain 1_000 ms
                time += (data_rate_time * (1 + generate_normal_random()) * 1_000 * data_size) / \
                        (self.data.size * self.opt.batch_size)
                # unlock process
                # self.model.devices_graph[from_dev][to_dev]["handler"] = False
                self.insert_link_event(event['id'], time, ACTION_END, from_dev, to_dev, int(data_size), event['flow'])

                if self.model.devices_graph[from_dev][to_dev]["last_unlock"] < time:
                    self.model.devices_graph[from_dev][to_dev]["last_unlock"] = time
                # handler wait process
                # for wait in self.wait_queue:
                #     if wait['type'] == TYPE_LINK:
                #         if (wait['from'] == from_dev and wait['to'] == to_dev) or \
                #                 (wait['from'] == to_dev and wait['to'] == from_dev):
                #             wait['time'] = time
                #             self.insert_event(wait)
                #             self.wait_queue.remove(wait)

            # link done transmission
            elif event['action'] == ACTION_END or event['action'] == ACTION_WAIT:
                self.model.devices_graph[from_dev][to_dev]["handler"] = False
                if self.model.devices_graph[from_dev][to_dev]["last_unlock"] < time:
                    self.model.devices_graph[from_dev][to_dev]["last_unlock"] = time

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
                    if self.model.devices_graph.nodes[to_dev]["last_lock"] < time:
                        self.model.devices_graph.nodes[to_dev]["last_lock"] = time
                    self.insert_dev_event(event['id'], time, ACTION_START, to_dev, int(data_size), event['flow'])

        if len(self.event_queue) == 0 and len(self.wait_queue) != 0:
            wait_event = self.wait_queue.pop(0)
            self.insert_event(wait_event)
