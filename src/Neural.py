class Neural:
    def __init__(self, first_layer_mem: float, first_layer_exec_time: float):   # MB / ms
        self.neural_layer_size = 0
        self.neural_mem = []
        self.neural_exec_time = []

        self.neural_mem.append(first_layer_mem)
        self.neural_exec_time.append(first_layer_exec_time)

    # def append(self, ):