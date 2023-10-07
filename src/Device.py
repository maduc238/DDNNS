class Device:
    def __init__(self, name: str, mem_size: float, mem_rate: float, training_rate: float):
        """
        :param name: Name of device
        :param mem_size: memory size with MB
        :param mem_rate: memory rate when allocating neural module
        :param training_rate: training time rate when allocating neural module
        """
        self.neural = None
        self.name = name
        self.mem_size = mem_size
        self.mem_rate = mem_rate
        self.training_rate = training_rate

    def set_neural(self, neural):
        self.neural = neural


class Device_Runner:
    def __init__(self):
        self._mem = 0
        self._data = None
