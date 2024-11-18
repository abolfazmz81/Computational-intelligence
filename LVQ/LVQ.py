class LVQ:

    def __init__(self,prototype_per_class=1, epoch=20,learning_rate=0.1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.prototype_per_class = prototype_per_class
        self.prototypes = None
        self.labels = None

    