import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.tree = None
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'agaurav6'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)
        return

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        y_result = []
        for point in points:
            y_result.append(self.lookup(point, 0))
        return np.array(y_result)

    def lookup(self, point, root):
        root_index = self.tree[root]
        current_index_int = int(root_index[0])
        if current_index_int == -1:
            return root_index[1]
        elif point[current_index_int] > root_index[1]:
            new_index = root + root_index[3]
        else:
            new_index = root + root_index[2]
        return self.lookup(point, int(new_index))

    # Directly taken from the Decision Tree Algorithm(JR Quinlan) pseudo code
    def build_tree(self, dataX, dataY):

        if dataX.shape[0] == 1:
            return self.mean_y(dataY)

        is_y_same = (dataY.std() == 0)
        if is_y_same:
            return self.mean_y(dataY)

        if dataY.shape[0] <= self.leaf_size:
            return self.mean_y(dataY)

        best_index = np.random.randint(dataX.shape[1])

        split_val = np.median(dataX[:, best_index])
        x_left_tree, y_left_tree = self.left_tree_index(dataX, dataY, best_index, split_val)

        if x_left_tree.shape[0] == 0:
            return self.mean_y(dataY)

        if x_left_tree.shape[0] == dataY.shape[0]:
            return self.mean_y(dataY)

        x_right_tree, y_right_tree = self.right_tree_index(dataX, dataY, best_index, split_val)

        left_tree = self.build_tree(x_left_tree, y_left_tree)
        right_tree = self.build_tree(x_right_tree, y_right_tree)

        root = [best_index, split_val, 1, left_tree.shape[0] + 1]
        return np.vstack((root, left_tree, right_tree))

    def left_tree_index(self, dataX, dataY, best_index, split_val):
        left_tree_indexes = dataX[:, best_index] <= split_val
        return dataX[left_tree_indexes], dataY[left_tree_indexes]

    def right_tree_index(self, dataX, dataY, best_index, split_val):
        right_tree_indexes = dataX[:, best_index] > split_val
        return dataX[right_tree_indexes], dataY[right_tree_indexes]

    def mean_y(self, dataY):
        return np.array([[-1, np.mean(dataY), np.nan, np.nan]])


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
