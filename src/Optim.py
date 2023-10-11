class Optim:
    def __init__(self, batch_size: int, num_micro_batch: int = None):
        self.batch_size = batch_size
        self.num_micro_batch = num_micro_batch
