# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:54:13 2018

@author: UID0982089

Unsupervised randomized Bayesian Hierarchical clustering
"""


from __future__ import print_function, division
import abc
import numpy as np
import itertools as it
import sys

from numpy import logaddexp
import math
from numpy.linalg import slogdet

from scipy import stats
from scipy.special import gammaln, multigammaln, factorial


LOG2PI = math.log(2*math.pi)
LOG2 = math.log(2)
LOGPI = math.log(math.pi)

def Poisson_pdf(k, lam = 7):
    return (lam**k * np.exp(-lam))/(factorial(k)) 


class DATA_MODEL(object):
    """
    The classs that learns the distribution of the data in an unsupervised fashion
    """
    def __init__(self, df):
        print('dataset size = {}'.format(df.size))
        if df.size == 1: print(df)
        self.est_density = stats.gaussian_kde(df.T)
        
    def update_parameters(self, df_new):
        self.est_density = stats.gaussian_kde(df_new.T)
        
    #def calc_log_z()
   
    def log_marginal_likelihood(self, X):
        lml = np.log(self.est_density.evaluate(X.T))
        return np.sum(lml)
    
    def log_posterior_predictive(self, X_new):
        ed = stats.gaussian_kde(X_new.T)
        lpp = np.log(ed.evaluate(X_new.T))
        return np.sum(lpp)
    
    def conditional_sample(self, size = 1):
        cs = self.est_density.resample(size)
        return cs
        
    
###############################################################################
###############################################################################
###############################################################################
class bhc(object):
    """
    An instance of Bayesian hierarchical clustering CRP mixture model.
    Attributes
    ----------
    assignments : list(list(int))
        A list of lists, where each list records the clustering at
        each step by giving the index of the leftmost member of the
        cluster a leaf is traced to.
    root_node : Node
        The root node of the clustering tree.
    lml : float
        An estimate of the log marginal likelihood of the model
        under a DPMM.
    Notes
    -----
    The cost of BHC scales as O(n^2) and so becomes inpractically
    large for datasets of more than a few hundred points.
    """

    def __init__(self, data, crp_alpha=1.0,
                 verbose=False):
        """
        Init a bhc instance and perform the clustering.

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each
            column is a dimension.
        
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        verbose : bool, optional
            Determibes whetrher info gets dumped to stdout.
        """
        self.data = data
        self.data_model = DATA_MODEL(data)
        self.crp_alpha = crp_alpha

        self.verbose = verbose

        # initialize the tree
        nodes = dict((i, Node(np.array([x]), crp_alpha,
                              indexes=i))
                     for i, x in enumerate(data))
        n_nodes = len(nodes)
        start_n_nodes = len(nodes)
        assignment = [i for i in range(n_nodes)]
        self.assignments = [list(assignment)]
        rks = []

        while n_nodes > 1:
            if self.verbose:
                sys.stdout.write("\r{0:d} of {1:d} ".format(n_nodes,
                                                            start_n_nodes))
                sys.stdout.flush()

            max_rk = float('-Inf')
            merged_node = None

            # for each pair of clusters (nodes), compute the merger
            # score.
            for left_idx, right_idx in it.combinations(nodes.keys(),
                                                       2):
                tmp_node = Node.as_merge(nodes[left_idx],
                                         nodes[right_idx])

                if tmp_node.log_rk > max_rk:
                    max_rk = tmp_node.log_rk
                    merged_node = tmp_node
                    merged_right = right_idx
                    merged_left = left_idx

            rks.append(math.exp(max_rk))

            # Merge the highest-scoring pair
            del nodes[merged_right]
            nodes[merged_left] = merged_node

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

            n_nodes -= 1

        self.root_node = nodes[0]
        self.assignments = np.array(self.assignments)

        # The denominator of log_rk is at the final merge is an
        # estimate of the marginal likelihood of the data under DPMM
        self.lml = self.root_node.log_ml

    def left_run(self):
        node = self.root_node
        while node.left_child is not None:
            print(node.indexes, np.mean(node.data, axis=0), node.data.shape)
            node = node.left_child
        print(node.indexes, np.mean(node.data, axis=0), node.data.shape)

    def right_run(self):
        node = self.root_node
        while node.right_child is not None:
            print(node.indexes, np.mean(node.data, axis=0), node.data.shape)
            node = node.right_child
        print(node.indexes, np.mean(node.data, axis=0), node.data.shape)

    def find_path(self, index):
        """ find_path(index)

            Finds the sequence of left and right merges needed to
            run from the root node to a particular leaf.

            Parameters
            ----------
            index : int
                The index of the leaf for which we want the path
                from the root node.
        """
        merge_path = []
        last_leftmost_index = self.assignments[-1][index]
        last_right_incluster = (self.assignments[-1]
                                == last_leftmost_index)

        for it in range(len(self.assignments)-2, -1, -1):
            new_leftmost_index = self.assignments[it][index]

            if new_leftmost_index != last_leftmost_index:
                # True if leaf is on the right hand side of a merge
                merge_path.append("right")
                last_leftmost_index = new_leftmost_index
                last_right_incluster = (self.assignments[it]
                                        == new_leftmost_index)

            else:       # Not in a right hand side of a merge

                new_right_incluster = (self.assignments[it]
                                       == last_leftmost_index)

                if (new_right_incluster != last_right_incluster).any():
                    # True if leaf is on the left hand side of a merge
                    merge_path.append("left")
                    last_right_incluster = new_right_incluster

        return merge_path

    def sample(self, size=1):

        output = np.zeros((size, self.root_node.data.shape[1]))

        for it in range(size):

            sampled = False
            node = self.root_node

            while not sampled:

                if node.log_rk is None:     # Node is a leaf
                    output[it, :] = self.data_model.conditional_sample()
##### modified here                                        node.data)
                    sampled = True

                elif np.random.rand() < math.exp(node.log_rk):
                    # sample from node
                    output[it, :] = self.data_model.conditional_sample()
##### modified here                                                                    node.data)
                    sampled = True

                else:   # drop to next level
                    child_ratio = (node.left_child.nk
                                   / (node.left_child.nk+node.right_child.nk))
                    if np.random.rand() >= child_ratio:
                        node = node.right_child
                    else:
                        node = node.left_child

        return output


class Node(object):
    """ A node in the hierarchical clustering.
    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    
    crp_alpha : float
        Chinese restaurant process concentration parameter
    log_dk : float
        Used in the calculation of the prior probability. Defined in
        Fig 3 of Heller & Ghahramani (2005).
    log_pi : float
        Prior probability that all associated leaves belong to one
        cluster.
    log_ml : float
        The log marginal likelihood for the tree of the node and
        its children. This is given by eqn 2 of Heller &
        Ghahrimani (2005). Note that this definition is
        recursive.  Do not define if the node is
        a leaf.
    logp : float
        The log marginal likelihood for the particular cluster
        represented by the node. Given by eqn 1 of Heller &
        Ghahramani (2005).
    log_rk : float
        The log-probability of the merge that created the node. For
        nodes that are leaves (i.e. not created by a merge) this is
        None.
    left_child : Node
        The left child of a merge. For nodes that are leaves (i.e.
        the original data points and not made by a merge) this is
        None.
    right_child : Node
        The right child of a merge. For nodes that are leaves
        (i.e. the original data points and not made by a merge)
        this is None.
    index : int
        The indexes of the leaves associated with the node in some
        indexing scheme.
    """

    def __init__(self, data, crp_alpha=1.0, log_dk=None,
                 log_pi=0.0, log_ml=None, logp=None, log_rk=None,
                 left_child=None, right_child=None, indexes=None):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Array of data_model-appropriate data
        
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_ml : float
            The log marginal likelihood for the tree of the node and
            its children. This is given by eqn 2 of Heller &
            Ghahrimani (2005). Note that this definition is
            recursive.  Do not define if the node is
            a leaf.
        logp : float
            The log marginal likelihood for the particular cluster
            represented by the node. Given by eqn 1 of Heller &
            Ghahramani (2005).
        log_rk : float
            The probability of the merged hypothesis for the node.
            Given by eqn 3 of Heller & Ghahrimani (2005). Do not
            define if the node is a leaf.
        left_child : Node, optional
            The left child of a merge. For nodes that are leaves (i.e.
            the original data points and not made by a merge) this is
            None.
        right_child : Node, optional
            The right child of a merge. For nodes that are leaves
            (i.e. the original data points and not made by a merge)
            this is None.
        index : int, optional
            The index of the node in some indexing scheme.
        """
        self.data_model = DATA_MODEL(data)
        self.data = data
        self.nk = data.shape[0]
        self.crp_alpha = crp_alpha
        self.log_pi = log_pi
        self.log_rk = log_rk

        self.left_child = left_child
        self.right_child = right_child

        if isinstance(indexes, int):
            self.indexes = [indexes]
        else:
            self.indexes = indexes

        if log_dk is None:
            self.log_dk = math.log(crp_alpha)
        else:
            self.log_dk = log_dk

        if logp is None:    # i.e. for a leaf
            self.logp = self.data_model.\
                            log_marginal_likelihood(self.data)
        else:
            self.logp = logp

        if log_ml is None:  # i.e. for a leaf
            self.log_ml = self.logp
        else:
            self.log_ml = log_ml

    @classmethod
    def as_merge(cls, node_left, node_right):
        """ Create a node from two other nodes
        Parameters
        ----------
        node_left : Node
            the Node on the left
        node_right : Node
            The Node on the right
        """
        crp_alpha = node_left.crp_alpha
        data_model = node_left.data_model
        data = np.vstack((node_left.data, node_right.data))
        indexes = node_left.indexes + node_right.indexes
        indexes.sort()

        nk = data.shape[0]
        log_dk = logaddexp(math.log(crp_alpha) + math.lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = -math.log1p(math.exp(node_left.log_dk
                                      + node_right.log_dk
                                      - math.log(crp_alpha)
                                      - math.lgamma(nk)))

        # Calculate log_rk - the log probability of the merge

        logp = data_model.log_marginal_likelihood(data)
        numer = log_pi + logp

        neg_pi = math.log(-math.expm1(log_pi))
        log_ml = logaddexp(numer, neg_pi+node_left.log_ml + node_right.log_ml)

        log_rk = numer-log_ml

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, crp_alpha, log_dk, log_pi,
                   log_ml, logp, log_rk, node_left, node_right,
                   indexes)

###############################################################################
###############################################################################
###############################################################################   
class rbhc(object):
    """
    An instance of Randomized Bayesian hierarchical clustering CRP
    mixture model.
    Attributes
    ----------

    Notes
    -----
    The cost of rBHC scales as O(nlogn) and so should be preferred
    for large data sets.
    """

    def __init__(self, data, crp_alpha=1.0, sub_size=100,
                 verbose=False):

        """
        Init a rbhc instance and perform the clustering.

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each
            column is a dimension.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood``
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        sub_size : int
            The size of the random subset of pooints used to form the
            tree whose top split is employed to filter the data.
            Denoted m in the Heller & Ghahramani (2005b).
        verbose : bool
            If true various bits of information, possibly with
            diagnostic uses, will be printed.
        """
        self.data = data
        self.data_model = DATA_MODEL(data)
        self.crp_alpha = crp_alpha
        self.sub_size = sub_size

        self.verbose = verbose

        self.nodes = {}

        # initialize the tree

        self.assignments = []

        root_node = rbhc_Node(data, self.data_model, crp_alpha)
        self.nodes[0] = {0: root_node}

#        self.tree = rbhc_Node.recursive_split(root_node, 50)
        self.recursive_split(root_node)

        self.find_assignments()
        self.refine_probs()

    def recursive_split(self, parent_node):
        print('split')
        rBHC_split, children = rbhc_Node.as_split(parent_node,
                                                  self.sub_size)

        if self.verbose:
            print("Parent node [{0}][{1}] ".format(
                       parent_node.node_level,
                       parent_node.level_index), end="")

        if rBHC_split:      # continue recussing down
            if children[0].node_level not in self.nodes:
                self.nodes[children[0].node_level] = {}

            self.nodes[children[0].node_level][children[0].level_index] = (
                                                                children[0])
            self.nodes[children[1].node_level][children[1].level_index] = (
                                                                children[1])

            if self.verbose:
                print("split to children:\n"
                      "\tnode [{0}][{1}], size : {2}\n"
                      "\tnode [{3}][{4}], size : {5}\n".format(
                       children[0].node_level,
                       children[0].level_index, children[0].nk,
                       children[1].node_level,
                       children[1].level_index, children[1].nk))

            self.recursive_split(children[0])
            self.recursive_split(children[1])

        else:               # terminate
            if parent_node.tree_terminated and self.verbose:
                print("terminated with bhc tree")
            elif parent_node.truncation_terminated and self.verbose:
                print("truncated")

    def find_assignments(self):
        """ find_assignements()

            Find which Node each data point is assigned to on each
            level.
            This fills self.assignemnts - which is a list, with an
            ndarray for each level. The array for each level gives
            the level index of the nde it is associated with.
            If a data point is not assigned to a node on a given
            level it is given the value -1.
        """

        self.assignments.append(np.zeros(self.data.shape[0]))

        for level_key in self.nodes:
            if level_key != 0:
                self.assignments.append(
                            np.zeros(self.data.shape[0])-1)

                for index_key in self.nodes[level_key]:
                    if index_key % 2 == 0:
                        parent_index = int(index_key/2)
                        write_indexes = (self.assignments[level_key-1]
                                         == parent_index)

                        self.assignments[level_key][write_indexes] = (
                              parent_index*2+1
                              - self.nodes[level_key-1][parent_index].
                              left_allocate.astype(int))

    def refine_probs(self):
        """ refine_probs()

            Improve the estimated probabilities used by working with
            the full set of data allocated to each node, rather than
            just the initial sub-set used to create/split nodes.
        """
        # travel up from leaves improving log_rk etc.

        for level_it in range(len(self.assignments)-1, -1, -1):
            # print(level_it, self.nodes[level_it].keys())

            for node_it in self.nodes[level_it]:
                node = self.nodes[level_it][node_it]

                if node.tree_terminated:
                    if node.nk > 1:
                        # log_rk, etc are accurate
                        node.log_dk = node.true_bhc.root_node.log_dk
                        node.log_pi = node.true_bhc.root_node.log_pi
                        node.logp = node.true_bhc.root_node.logp
                        node.log_ml = node.true_bhc.root_node.log_ml
                        node.log_rk = node.true_bhc.root_node.log_rk
                    else:
                        node.log_dk = self.crp_alpha
                        node.log_pi = 0.
                        node.logp = self.data_model.log_marginal_likelihood(
                                                                    node.data)
                        node.log_ml = node.logp
                        node.log_rk = 0.

                elif node.truncation_terminated:
                    node.log_dk = (math.log(self.crp_alpha)
                                   + math.lgamma(node.nk))
                    node.log_pi = 0.
                    node.logp = self.data_model.log_marginal_likelihood(
                                                                node.data)
                    node.log_ml = node.logp
                    node.log_rk = 0.

                else:
                    left_child = self.nodes[level_it+1][node_it*2]
                    right_child = self.nodes[level_it+1][node_it*2+1]

                    node.log_dk = np.logaddexp(
                           math.log(self.crp_alpha)
                           + math.lgamma(node.nk),
                           left_child.log_dk + right_child.log_dk)

                    node.log_pi = -math.log1p(math.exp(
                                           left_child.log_dk
                                           + right_child.log_dk
                                           - math.log(self.crp_alpha)
                                           - math.lgamma(node.nk)))
                    neg_pi = math.log(-math.expm1(node.log_pi))

                    node.logp = self.data_model.log_marginal_likelihood(
                                                                node.data)

                    node.log_ml = np.logaddexp(node.log_pi+node.logp,
                                               neg_pi + left_child.log_ml
                                               + right_child.log_ml)
                    node.log_rk = node.log_pi + node.logp - node.log_ml

        # travel down from top improving

        for level_it in range(1, len(self.assignments)):
            for node_it in self.nodes[level_it]:
                node = self.nodes[level_it][node_it]
                parent_node = self.nodes[level_it-1][int(node_it/2)]

                node.prev_wk = (parent_node.prev_wk
                                * (1-math.exp(parent_node.log_rk)))

    def __str__(self):
        bhc_str = ("==================================\n"
                   "rBHC fit to {0} data points, with "
                   "alpha={1} and sub_size={2} .\n".format(
                       self.data.shape[0], self.crp_alpha, self.sub_size))

        for l_it in range(len(self.nodes)):
            bhc_str += "===== LEVEL {0} =====\n".format(l_it)
            for n_it in self.nodes[l_it]:
                node = self.nodes[l_it][n_it]
                bhc_str += ("node : {0} size : {1} "
                            "node_prob : {2:.5f} \n".format(
                                   n_it, node.nk,
                                   node.prev_wk*np.exp(node.log_rk)))
        return bhc_str

    def sample(self, size=1):
        """ sample(size)

            Sample from a fitted rBHC tree.

            Parameters
            ----------
            size : int
                The number of samples to draw
        """
        output = np.zeros((size, self.data.shape[1]))

        for it in range(size):

            sampled = False
            node = self.nodes[0][0]
            l_it = 0
            n_it = 0

            while not sampled:

                if node.tree_terminated:     # tree has BHC child at this node
                    if node.nk > 1:
                        output[it, :] = node.true_bhc.sample()
                    else:
                        output[it, :] = self.data_model.conditional_sample()       
##### modified here                                            node.data)
                    sampled = True

                elif node.truncation_terminated:
                    output[it, :] = self.data_model.conditional_sample()
##### modified here                                        node.data)
                    sampled = True

                elif np.random.rand() < math.exp(node.log_rk):
                    # sample from node
                    output[it, :] = self.data_model.conditional_sample()
#### modified here                                        node.data)
                    sampled = True

                else:  # drop to next level
                    child_ratio = (self.nodes[l_it+1][n_it*2].nk
                                   / (self.nodes[l_it+1][n_it*2].nk
                                      + self.nodes[l_it+1][n_it*2+1].nk))

                    if np.random.rand() < child_ratio:
                        l_it += 1
                        n_it = n_it*2
                    else:
                        l_it += 1
                        n_it = n_it*2+1
                    node = self.nodes[l_it][n_it]

        return output


class rbhc_Node(object):
    """ A node in the randomised Bayesian hierarchical clustering.
        Attributes
        ----------
        nk : int
            Number of data points assigned to the node
        D : int
            The dimension of the data points
        data : numpy.ndarrary (n, d)
            The data assigned to the Node. Each row is a datum.
       
        crp_alpha : float
            Chinese restaurant process concentration parameter
        log_rk : float
            The probability of the merged hypothesis for the node.
            Given by eqn 3 of Heller & Ghahrimani (2005).
        prev_wk : float
            The product of the (1-r_k) factors for the nodes leading
            to this node from (and including) the root node. Used in
            eqn 9 of Heller & ghahramani (2005a).
        node_level : int, optional
            The level in the hierarchy at which the node is found.
            The root node lives in level 0 and the level number
            increases down the tree.
        level_index : int, optional
            An index that identifies each node within a level.
        left_allocate : ndarray(bool)
            An array that records if a datum has been allocated
            to the left child (True) or the right(False).
        log_dk : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_ml : float
            The log marginal likelihood for the tree of the node and
            its children. This is given by eqn 2 of Heller &
            Ghahrimani (2005). Note that this definition is
            recursive.  Do not define if the node is
            a leaf.
        logp : float
            The log marginal likelihood for the particular cluster
            represented by the node. Given by eqn 1 of Heller &
            Ghahramani (2005).
    """
    def __init__(self, data, data_model, crp_alpha=1.0, prev_wk=1.,
                 node_level=0, level_index=0):
        """ __init__(data, data_model, crp_alpha=1.0)

            Initialise a rBHC node.

            Parameters
            ----------
            data : numpy.ndarrary (n, d)
                The data assigned to the Node. Each row is a datum.
            data_model : idsteach.CollapsibleDistribution
                The data model used to calcuate marginal likelihoods
            crp_alpha : float, optional
                Chinese restaurant process concentration parameter
            prev_wk : float
                The product of the (1-r_k) factors for the nodes
                leading to this node from (and including) the root
                node. Used in eqn 9 of Heller & ghahramani (2005a).
            node_level : int, optional
                The level in the hierarchy at which the node is found.
                The root node lives in level 0 and the level number
                increases down the tree.
            level_index : int, optional
                An index that identifies each node within a level.

        """

        self.data = data
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self.prev_wk = prev_wk
        self.node_level = node_level
        self.level_index = level_index

        self.nk = data.shape[0]
        self.D = data.shape[1]

        self.log_rk = 0

        self.tree_terminated = False
        self.truncation_terminated = False

    def set_rk(self, log_rk):
        """ set_rk(log_rk)

            Set the value of the ln(r_k) The probability of the
            merged hypothesis as given in Eqn 3 of Heller & Ghahramani
            (2005a)

            Parameters
            ----------
            log_rk : float
                The value of log_rk for the node
        """
        self.log_rk = log_rk

    @classmethod
    def as_split(cls, parent_node, sub_size):
        """ as_split(parent_node, subsize)

            Perform a splitting of a rBHC node into two children.
            If the number of data points is large a randomized
            filtered split, as in Fig 4 of Heller & Ghahramani (2005b)
            is performed.
            Otherwise, if the number of points is less than or equal
            to subsize then these are simply subject to a bhc
            clustering.

            Parameters
            ----------
            parent_node : rbhc_Node
                The parent node that is going to be split
            sub_size : int
                The size of the random subset of pooints used to form
                the tree whose top split is employed to filter the
                data.
                Denoted m in Heller & Ghahramani (2005b).

            Returns
            -------
            rBHC_split : bool
                True if the size of data is greater than sub_size and
                so a rBHC split/filtering has occured.
                False if the size of data is less than/equal to
                sub_size and so an bhc clustering that includes all
                the data has been found.
            children : list(rbhc_Node) , bhc
                A clustering of the data, either onto two child
                rbhc_Nodes or as a full bhc tree of all the data
                within parent_node.
            left_allocate : ndarray(bool)
                An array that records if a datum has been allocated
                to the left child (True) or the right(False).
        """

        if (parent_node.prev_wk*parent_node.nk) < 1E-3:
            print("Truncating", parent_node.prev_wk, parent_node.nk,
                  parent_node.prev_wk*parent_node.nk)
            rBHC_split = False
            parent_node.truncation_terminated = True
            children = []

            # make subsample tree
            if parent_node.nk > sub_size:
                parent_node.subsample_bhc(sub_size)

                # set log_rk from the estimate given by self.sub_bhc
                parent_node.set_rk(parent_node.sub_bhc.root_node.log_rk)
            elif parent_node.nk > 1:
                parent_node.true_bhc = bhc(parent_node.data,
                                           parent_node.crp_alpha)
                parent_node.set_rk(parent_node.true_bhc.root_node.log_rk)
                parent_node.tree_terminated = True
            else:
                parent_node.set_rk(0.)
                parent_node.tree_terminated = True

        else:

            if parent_node.nk > sub_size:    # do rBHC filter
                # make subsample tree
                parent_node.subsample_bhc(sub_size)

                # set log_rk from the estimate given by self.sub_bhc
                parent_node.set_rk(parent_node.sub_bhc.root_node.log_rk)

                # filter data through top level of subsample_bhc
                parent_node.filter_data()

                # create new nodes

                child_prev_wk = (parent_node.prev_wk
                                 * (1-math.exp(parent_node.log_rk)))
                child_level = parent_node.node_level+1

                left_child = cls(parent_node.left_data,
                                 parent_node.crp_alpha, child_prev_wk,
                                 child_level, parent_node.level_index*2)
                right_child = cls(parent_node.right_data,
                                  parent_node.crp_alpha, child_prev_wk,
                                  child_level, parent_node.level_index*2+1)
                rBHC_split = True
                children = [left_child, right_child]

            elif parent_node.nk > 1:             # just use the bhc tree
                parent_node.true_bhc = bhc(parent_node.data,
                                           parent_node.crp_alpha)
                children = parent_node.true_bhc
                rBHC_split = False
                parent_node.tree_terminated = True

                parent_node.set_rk(children.root_node.log_rk)

            else:                       # only 1 datum
                children = []
                rBHC_split = False
                parent_node.tree_terminated = True

                parent_node.set_rk(0.)

        print("\n", parent_node.node_level, parent_node.level_index,
              parent_node.nk, parent_node.prev_wk,
              math.exp(parent_node.log_rk), (1-math.exp(parent_node.log_rk)))

        return (rBHC_split, children)

    def subsample_bhc(self, sub_size):
        """ subsample_bhc(sub_size)

            Produce a subsample of sub_size data points and then
            perform an bhc clustering on it.

            Parameters
            ----------
            sub_size : int
                The size of the random subset of pooints used to form
                the tree whose top split is employed to filter the
                data.
                Denoted m in Heller & Ghahramani (2005b).
        """

        self.sub_indexes = np.random.choice(self.data.index.values.ravel(),
                                            sub_size, replace=False)
        sub_data = self.data.loc[self.sub_indexes]
        #print(sub_data)
        self.sub_bhc = bhc(sub_data, self.crp_alpha)

    def filter_data(self):
        """ filter_data()

            Filter the data in a rbhc_node onto the two Nodes at the
            second from top layer of a bhc tree.
        """
        # set up data arrays
        self.left_data = np.empty(shape=(0, self.D))
        self.right_data = np.empty(shape=(0, self.D))

        # create assignemnt array
        self.left_allocate = np.zeros(self.nk, dtype=bool)

        # Run through data

        for ind in np.arange(self.nk):

            # check if in subset
            if ind in self.sub_indexes:
                sub_ind = np.argwhere(self.sub_indexes == ind)[0][0]
                if self.sub_bhc.assignments[-2][sub_ind] == 0:
                    self.left_allocate[ind] = True
                    self.left_data = np.vstack((self.left_data,
                                                self.data[ind]))

                else:
                    self.right_data = np.vstack((self.right_data,
                                                 self.data[ind]))

            # non subset data
            else:
                left_prob = (self.sub_bhc.root_node.left_child.log_pi
                             + self.data_model.log_posterior_predictive(
                                 self.data[ind],
                                 self.sub_bhc.root_node.left_child.data))

                right_prob = (self.sub_bhc.root_node.right_child.log_pi
                              + self.data_model.log_posterior_predictive(
                                  self.data[ind],
                                  self.sub_bhc.root_node.right_child.data))

                if left_prob >= right_prob:
                    # possibly change this to make tupe and vstack at
                    # end if cost is high
                    self.left_allocate[ind] = True
                    self.left_data = np.vstack((self.left_data,
                                                self.data[ind]))

                else:
                    self.right_data = np.vstack((self.right_data,
                                                 self.data[ind]))

        print("split", np.sum(self.left_allocate), self.left_allocate.size)
