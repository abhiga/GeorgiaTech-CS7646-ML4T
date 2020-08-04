import numpy as np
import LinRegLearner as lrl


class BagLearner(object):

    def __init__(self, learner=lrl.LinRegLearner, kwargs={}, bags=10, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'agaurav6'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        for i in range(self.bags):
            data_index = np.random.randint(0, dataX.shape[0], dataX.shape[0])
            self.learners[i].addEvidence(dataX[data_index], dataY[data_index])

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = np.zeros((points.shape[0],))
        for i in range(self.bags):
            result += self.learners[i].query(points)
        return result / self.bags

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
