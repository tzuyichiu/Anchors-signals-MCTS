"""
Inspired from:
    A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
import math
import itertools
import heapq

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, batch_size=256, max_depth=5, delta=0.01, epsilon=0.01,
                    method='ucb1tuned', ancestors=True):
        
        self.batch_size = batch_size
        self.max_depth  = max_depth
        self.delta      = delta
        self.epsilon    = epsilon
        self.method     = method        # uct / ucb1tuned
        
        self.children = defaultdict(set)
        if ancestors:
            self.ancestors = defaultdict(set)
        else:
            self.ancestors = None
        
        # Monte-Carlo (score of a node)
        self.Q = defaultdict(int)
        self.N = defaultdict(int)

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        return heapq.nlargest(5, self.children[node], key=self.__score)

    def train(self, node):
        "Rollout the tree from `node` until error is smaller than `epsilon`"
        err = 1.0
        i = 0
        while err > self.epsilon:
            i += 1
            print(f'\033[1;93m Iter {i} Error {err:5.2%}\033[1;m', end='  \r')
            self._rollout(node)
            
            # Hoeffding's bound
            best = self._select(node)
            if self.N[best]:
                div = math.sqrt(-math.log(self.delta/2) / (2*self.N[best]))
                err = min(div, 1.0)
        print()

    def _rollout(self, node):
        "Make the tree one layer better (train for one iteration)"
        path = self._select_path(node)
        leaf = path[-1]
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)
        if len(leaf) < self.max_depth:
            self._expand(leaf)

    def __score(self, node):
        "Empirical score of `node`"
        if not self.N[node]:
            return float('-inf')
        return self.Q[node] / self.N[node]
    
    def _select_path(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                path.append(node)
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._select(node)
            path.append(node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        self.children[node] = node.get_children()
        if self.ancestors is not None:
            ancestors = self.ancestors[node] # for back-propagation
            for child in self.children[node]:
                self.ancestors[child].add(node)

    def _simulate(self, node):
        "Return the reward for a random simulation of `node`"
        return node.reward(self.batch_size)
    
    def _backpropagate(self, path, reward):
        "Send `reward` back up to the ancestors of `node`"
        if self.ancestors is None:
            for n in path:
                self.Q[n] += reward
                self.N[n] += self.batch_size
        else:
            ancestors = {path[-1]}
            for n in path:
                ancestors.update(self.ancestors[n])
            for n in ancestors:
                self.Q[n] += reward
                self.N[n] += self.batch_size

    def _select(self, node, ce=2):
        "Select a child of node, balancing exploration & exploitation"

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            if not self.N[n]:
                return float('-inf')
            mu = self.__score(n)
            sqrt = math.sqrt(ce * log_N_vertex / self.N[n])
            return mu + sqrt

        def ucb1tuned(n):
            if not self.N[n]:
                return float('-inf')
            mu = self.__score(n)
            tmp = math.sqrt(ce * log_N_vertex / self.N[n])
            return mu + tmp * math.sqrt(min(0.25, mu * (1 - mu) + tmp))

        return max(self.children[node], key=eval(self.method))