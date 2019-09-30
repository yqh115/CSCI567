import numpy as np
from typing import List
from classifier import Classifier


class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert (len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels) + 1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')

        string = ''
        for idx_cls in range(node.num_cls):
            string += str(node.labels.count(idx_cls)) + ' '
        print(indent + ' num of sample / cls: ' + string)

        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name='  ' + name + '/' + str(idx_child), indent=indent + '  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent + '}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label  # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
			branches: C x B array,
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of
					  corresponding training samples
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●

				      branches = [[2,2], [4,0]]
			'''
            # compute the conditional entropy
            b = np.array(branches)
            b_t = np.transpose(b).tolist()
            totsum = float(np.sum(b))
            c_e = 0.0
            for branch in b_t:
                b_sum = float(sum(branch))
                if b_sum == 0:
                    continue
                for class_label in branch:
                    if class_label > 0:
                        pi = float(class_label) / b_sum
                        c_e -= pi * np.log(pi) * (b_sum / totsum)
            return c_e

        for idx_dim in range(len(self.features[0])):
            try:
                min_entropy
            except NameError:
                min_entropy = np.inf
            xi = np.array(self.features)[:, idx_dim]

            if None in xi:
                continue

            classes = np.unique(xi)

            branches = np.zeros((self.num_cls, len(classes)))
            for i, cls in enumerate(classes):

                y = np.array(self.labels)[np.where(xi == cls)]

                for yi in y:
                    branches[yi, i] += 1
            entropy = conditional_entropy(branches)

            if entropy < min_entropy:
                min_entropy = entropy
                self.dim_split = idx_dim
                self.feature_uniq_split = classes.tolist()

        x = np.array(self.features, dtype=object)
        x[:, self.dim_split] = None
        xi = np.array(self.features)[:, self.dim_split]
        for val in self.feature_uniq_split:
            indexes = np.where(xi == val)
            x_new = x[indexes].tolist()
            y_new = np.array(self.labels)[indexes].tolist()
            child = TreeNode(x_new, y_new, self.num_cls)
            if len(x_new) == 0 or x_new[0].count(None) == len(x_new[0]):
                child.splittable = False
            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()
        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])

            return self.children[idx_child].predict(feature)

        else:

            return self.cls_max
