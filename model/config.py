class Config:
    NAME = 'brain_convolution'

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    DIM_CONV1 = 64
    DIM_CONV2 = 64
    DIM_CONV3 = 64
    DIM_READOUT = 256

    def __init__(self, node_dim, num_classes, batch_size=1):
        self.NODE_DIM = node_dim
        self.NUM_CLASSES = num_classes
        self.BATCH_SIZE = batch_size
