import numpy as np
import pandas

class N1:
    #neiron with N inputs

    def __init__(self,counts):
        self.inputs = np.zeros(counts)
        self.output = 0
        self.weights = np.random.random(counts)
        self.counts = counts

