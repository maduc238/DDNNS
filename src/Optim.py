class Optim:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.micro_batch_size = None