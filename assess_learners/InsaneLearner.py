import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, bags=20)] * 20
        pass  # move along, these aren't the drones you're looking for
    def author(self):
        return 'agaurav6'  # replace tb34 with your Georgia Tech username
    def addEvidence(self, dataX, dataY):
        for learner in self.learners:
            learner.addEvidence(dataX, dataY)
    def query(self, points):
        result = np.zeros((points.shape[0],))
        count = 0
        for i in self.learners:
            result += i.query(points)
            count += 1
        return result / count