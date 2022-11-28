"""
@file   stl.py
@brief  STL objects (primitives & formulas)

@author Tzu-Yi Chiu <tzuyi.chiu@gmail.com>
"""

from __future__ import annotations

import numpy as np
import itertools

from dataclasses import dataclass
from typing import List, FrozenSet, Set, Tuple

class Primitive:
    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        raise NotImplementedError("`robust` not implemented")

    def satisfied(self, s: np.ndarray) -> bool:
        "Verify if satisfied by signal `s`"
        return self.robust(s) > 0

    def is_child_of(self, parent: Primitive) -> bool:
        raise NotImplementedError("`is_child_of` not implemented")

    def __hash__(self):
        raise NotImplementedError("`__hash__` not implemented")

    def __eq__(self, other):
        return isinstance(other, Primitive) and hash(self) == hash(other)
    
    def __repr__(self):
        raise NotImplementedError("`__repr__` not implemented")

from typing import List, FrozenSet, Set, Tuple

class Primitive:
    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        raise NotImplementedError("`robust` not implemented")

    def satisfied(self, s: np.ndarray) -> bool:
        "Verify if satisfied by signal `s`"
        return self.robust(s) > 0

    def is_child_of(self, parent: Primitive) -> bool:
        raise NotImplementedError("`is_child_of` not implemented")

    def __hash__(self):
        raise NotImplementedError("`__hash__` not implemented")

    def __eq__(self, other):
        return isinstance(other, Primitive) and hash(self) == hash(other)
    
    def __repr__(self):
        raise NotImplementedError("`__repr__` not implemented")

@dataclass
class Eventually(Primitive):
    "Ex: Eventually((0, 5), (0, '>', 20), 1) <=> F[0,5](s1>20)"
    
    _interval: Tuple[int, int]   # bound delay
    _phi: Tuple[int, str, float] # s_d > mu or s_d < mu
    
    def __post_init__(self):
        try:
            a, b = self._interval
            d, comp, mu = self._phi
        except ValueError:
            raise ValueError('Invalid primitive parameters')
        
        repr_ = f'F[{a},{b}](s{d+1}{comp}{mu:.2f})'
        if a * b < 0 or a > b or (a < 0 and b == 0):
            raise ValueError(f'Invalid interval: {repr_}')
        if d < 0:
            raise ValueError(f'Invalid dimension: {repr_}')
        if comp not in ['<', '>']:
            raise ValueError(f'Invalid comparison: {repr_}')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        a, b = self._interval
        slicing = np.arange(a, b+1)
        if not slicing.size:
            return -1
        d, comp, mu = self._phi
        if comp == '<':
            return mu - np.min(s[d, slicing])
        return np.max(s[d, slicing]) - mu

    def is_child_of(self, parent: Primitive) -> bool:
        if self == parent or isinstance(parent, Globally):
            return False
                
        a, b = self._interval
        d, comp, mu = self._phi
        
        # Eventually
        if isinstance(parent, Eventually):
            a_, b_ = parent._interval
            d_, comp_, mu_ = parent._phi
            if comp != comp_ or d != d_:
                return False
            if a < a_ or b > b_: 
                return False
            if comp == '<' and mu <= mu_:
                return True
            if comp == '>' and mu >= mu_:
                return True
            return False

        # Until
        if not parent._negation:
            return False
        a_, b_ = parent._interval
        d1_, comp1_, mu1_ = parent._phi1
        if a > a_:
            return False
        if comp == '<' and comp1_ == '>' and d == d1_ and mu <= mu1_:
            return True
        if comp == '>' and comp1_ == '<' and d == d1_ and mu >= mu1_:
            return True
        return False

    def __hash__(self):
        return hash(('F', self._interval, self._phi))

    def __repr__(self):
        a, b = self._interval
        d, comp, mu = self._phi
        return f'F[{a},{b}](s{d+1}{comp}{mu:.2f})'

@dataclass
class Globally(Primitive):
    "Ex: Globally((0, 5), (0, '>', 20), 1) <=> G[0,5](s1>20)"
    
    _interval: Tuple[int, int]   # bound delay
    _phi: Tuple[int, str, float] # s_d > mu or s_d < mu
    
    def __post_init__(self):
        try:
            a, b = self._interval
            d, comp, mu = self._phi
        except ValueError:
            raise ValueError('Invalid primitive parameters')

        if a == b:
            raise ValueError('In case a = b, use Eventually (F).')
        repr_ = f'G[{a},{b}](s{d+1}{comp}{mu:.2f})'
        if a * b < 0 or a > b or (a < 0 and b == 0):
            raise ValueError(f'Invalid interval: {repr_}')
        if d < 0:
            raise ValueError(f'Invalid dimension: {repr_}')
        if comp not in ['<', '>']:
            raise ValueError(f'Invalid comparison: {repr_}')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        a, b = self._interval
        slicing = np.arange(a, b+1)
        if not slicing.size:
            return -1
        d, comp, mu = self._phi
        if comp == '<':
            return mu - np.max(s[d, slicing])
        return np.min(s[d, slicing]) - mu

    def is_child_of(self, parent: Primitive) -> bool:
        if parent == self:
            return False
                
        a, b = self._interval
        d, comp, mu = self._phi
        
        # Eventually
        if isinstance(parent, Eventually):
            a_, b_ = parent._interval
            d_, comp_, mu_ = parent._phi
            if comp != comp_ or d != d_:
                return False
            if a > b_ or b < a_: # empty intersection
                return False
            if comp == '<' and mu <= mu_:
                return True 
            if comp == '>' and mu >= mu_:
                return True
            return False
            
        # Globally
        if isinstance(parent, Globally):
            a_, b_ = parent._interval
            d_, comp_, mu_ = parent._phi
            if comp != comp_ or d != d_:
                return False
            if a > a_ or b < b_:
                return False
            if comp == '<' and mu <= mu_:
                return True 
            if comp == '>' and mu >= mu_:
                return True
            return False
            
        # Until
        if not parent._negation:
            return False
        a_, b_ = parent._interval
        d1_, comp1_, mu1_ = parent._phi1
        d2_, comp2_, mu2_ = parent._phi2
        if a > a_:
            return False
        if comp == '<' and comp1_ == '>' and d == d1_ and mu <= mu1_:
            return True
        if comp == '>' and comp1_ == '<' and d == d1_ and mu >= mu1_:
            return True
        if b < b_:
            return False
        if comp == '<' and comp2_ == '>' and d == d2_ and mu <= mu2_:
            return True
        if comp == '>' and comp2_ == '<' and d == d2_ and mu >= mu2_:
            return True    
        return False
    
    def __hash__(self):
        return hash(('G', self._interval, self._phi))

    def __repr__(self):
        a, b = self._interval
        d, comp, mu = self._phi
        return f'G[{a},{b}](s{d+1}{comp}{mu:.2f})'

@dataclass
class Until(Primitive):
    """
    Ex:
    Until(False, (0, '<', 0), (0, 5), (1, '>', 0)) <=> (s0 < 0)U[0,5](s1 > 0) 
    which reads s0 < 0 (phi1) until s1 > 0 (phi2)
    where phi2 happens within time interval [0, 5]. 
    Note that phi1 should not imply phi2. 
    Until(True, (0, '<', 0), (0, 5), (1, '>', 0)) <=> !(s0 < 0)U[0,5](s1 > 0)
    """
    _negation: bool = False
    _phi1: Tuple[int, str, float] # s_d > mu or s_d < mu
    _interval: Tuple[int, int]    # bound delay
    _phi2: Tuple[int, str, float] # s_d > mu or s_d < mu
    
    def __post_init__(self):
        try:
            d1, comp1, mu1 = self._phi1
            a, b = self._interval
            d2, comp2, mu2 = self._phi2
        except ValueError:
            raise ValueError('Invalid primitive parameters')
        
        n = '!' * self._negation
        d1 += 1
        d2 += 1
        repr_ = f'{n}(s{d1}{comp1}{mu1:.2f})U[{a},{b}](s{d2}{comp2}{mu2:.2f})'
        if a * b < 0 or a > b or (a < 0 and b == 0):
            raise ValueError(f'Invalid interval: {repr_}')
        if d1 < 1 or d2 < 1:
            raise ValueError(f'Invalid dimension: {repr_}')
        if comp1 not in ['<', '>'] or comp2 not in ['<', '>']:
            raise ValueError(f'Invalid comparison: {repr_}')
        if comp1 == comp2 == '>' and mu1 >= mu2:
            raise ValueError(f'phi1 should not imply phi2: {repr_}')
        if comp1 == comp2 == '<' and mu1 <= mu2:
            raise ValueError(f'phi1 should not imply phi2: {repr_}')

    def robust(self, s: np.ndarray) -> float:
        "Compute the robustness degree relative to signal `s`"
        d1, comp1, mu1 = self._phi1
        a, b = self._interval
        d2, comp2, mu2 = self._phi2
        slicing = np.arange(a, b+1)
        if not slicing.size:
            return -1
        
        robust1 = mu1 - s[d] if comp1 == '<' else s[d] - mu1 # rho(phi1, s, t)
        robust2 = mu2 - s[d] if comp2 == '<' else s[d] - mu2 # rho(phi2, s, t)
        
        # min_robust1 := min_{t' in [0,t]} rho(phi1, s, t')
        arange = range(a, 0) if a < 0 else range(a + 1)
        min_robust1 = [min(robust1[t] for t in arange)]
        for t in slicing[1:]:
            min_robust1.append(min(min_robust1[-1], s[d, t]))
        result = max(np.minimum(robust2, min_robust1))
        return - result if self._negation else result
        
    def is_child_of(self, parent: Primitive) -> bool:
        if parent == self or isinstance(parent, Globally):
            return False

        n = self._negation
        d1, comp1, mu1 = self._phi1
        a, b = self._interval
        d2, comp2, mu2 = self._phi2

        # Eventually
        if isinstance(parent, Eventually):
            if n:
                return False
            a_, b_ = parent._interval
            d_, comp_, mu_ = parent._phi
            if a < a_:
                return False
            if comp1 == '<' and mu1 <= mu1_:
                return True 
            if comp1 == '>' and mu1 >= mu1_:
                return True
            if b > b_:
                return False
            if comp2 == '<' and mu2 <= mu2_:
                return True
            if comp2 == '>' and mu2 >= mu2_:
                return True
            return False

        # Until
        n_ = parent._negation
        a_, b_ = parent._interval
        d1_, comp1_, mu1_ = parent._phi1
        d2_, comp2_, mu2_ = parent._phi2
        
        if (n and not n_) or (n_ and not n):
            return False
        
        if comp1 != comp1_ or comp2 != comp2_ or d1 != d1_ or d2 != d2_:
            return False
        
        if not (n or n_): # U v.s. U
            if a < a_ or b > b_:
                return False
            if comp1 == '<' and mu1 <= mu1_:
                return True
            if comp1 == '>' and mu1 >= mu1_:
                return True
            if comp2 == '<' and mu2 <= mu2_:
                return True
            if comp2 == '>' and mu2 >= mu2_:
                return True

        if n and n_: # !U v.s. !U
            if a > a_ or b < b_:
                return False
            if comp1 == '<' and mu1 <= mu1_:
                return True
            if comp1 == '>' and mu1 >= mu1_:
                return True
            if comp2 == '<' and mu2 <= mu2_:
                return True
            if comp2 == '>' and mu2 >= mu2_:
                return True
        return False

    def __hash__(self):
        return hash(('U', self._negation, self._phi1, 
                          self._interval, self._phi2))

    def __repr__(self):
        n = '!' * self._negation
        d1, comp1, mu1 = self._phi1
        a, b = self._interval
        d2, comp2, mu2 = self._phi2
        d1 += 1
        d2 += 1
        return f'{n}(s{d1}{comp1}{mu1:.2f})U[{a},{b}](s{d2}{comp2}{mu2:.2f})'

@dataclass
class PrimitiveGenerator:
    "(static) generator of primitives"
    
    # (instance attributes)
    _s: np.ndarray      # signal being explained
    _srange: list       # list of (min, max, stepsize) for each dimension
    _rho: float         # robustness degree (~coverage) threshold
    _past: bool = False # true if PtSTL, false if STL
        
    def generate(self) -> List[Primitive]:
        "Generate STL primitives whose robustness is greater than `rho`"
            
        result = []
        sdim, slen = self._s.shape
        arange = range(-slen, 0) if self._past else range(slen)
        for d_ in range(sdim):
            smin, smax, stepsize = self._srange[d_]
            mus = np.linspace(smin, smax, num=stepsize, endpoint=False)[1:]
            n = smax - smin
            for a_ in arange:
                for typ_ in ['Eventually', 'Globally']:
                    b_ = a_ + int(typ_ == 'Globally')
                    brange = range(b_, 0) if self._past else range(b_, slen)
                    l = [[typ_], [a_], brange, [d_], ['>', '<']]
                    for typ, a, b, d, comp in itertools.product(*l):
                        stop = False
                        phi0 = eval(typ)((a, b), (d, comp, mus[0]), n)
                        phi1 = eval(typ)((a, b), (d, comp, mus[-1]), n)
                        if phi0.robust(self._s) >= self._rho:
                            u = 0
                            if phi1.robust(self._s) < self._rho:
                                l = len(mus) - 1
                                from_begin = True
                            else:
                                stop = True
                                from_begin = False
                        else:
                            from_begin = False
                            l = 0
                            if phi1.robust(self._s) >= self._rho:
                                u = len(mus) - 1
                            else:
                                u = len(mus)
                                stop = True
                        if not stop:
                            while True:
                                phi0 = eval(typ)((a, b), (d, comp, mus[l]), n)
                                phi1 = eval(typ)((a, b), (d, comp, mus[u]), n)
                                if (phi0.robust(self._s) >= self._rho and 
                                        phi1.robust(self._s) >= self._rho):
                                    break
                                elif (phi0.robust(self._s) < self._rho and 
                                        phi1.robust(self._s) < self._rho):
                                    break
                                q = (u + l) // 2
                                if u == q or l == q:
                                    break
                                phi2 = eval(typ)((a, b), (d, comp, mus[q]), n)
                                if phi2.robust(self._s) >= self._rho:
                                    u = q
                                else:
                                    l = q
                        
                        rng = range(u+1) if from_begin else range(u, len(mus))
                        for q in rng:
                            phi = eval(typ)((a, b), (d, comp, mus[q]), n)
                            result.append(phi)
        return result


"""
Any newly instanciated STL formula remains the same instance.
Since the primitives is fixed from the beginning, an STL formula 
consisting of conjunction of some of these primitives is just 
represented as a frozenset of their indices.
"""
class STL(object):
    __cache = {}

    # (class attributes) to be set during init
    __primitives        = []    # list of generated primitives
    __parents           = {}    # dict {child: parents} among primitives

    def __new__(cls, indices: FrozenSet[int]=frozenset()):
        for child in indices.copy():
            indices -= STL.__parents[child]
        
        if indices in STL.__cache:
            return STL.__cache[indices]
        else:
            o = object.__new__(cls)
            STL.__cache[indices] = o
            return o
    
    def __init__(self, indices: FrozenSet[int]=frozenset()):
        self._indices = indices
        for child in indices:
            self._indices -= STL.__parents[child]
    
    def init(self, primitives: List[Primitive]) -> int:
        STL.__primitives = primitives
        nb = len(primitives)
        STL.__parents = {child: {parent for parent in range(nb)
            if STL.__primitives[child].is_child_of(STL.__primitives[parent])} 
            for child in range(nb)}
        return nb
    
    def satisfied(self, s: np.ndarray) -> bool:
        "Verify if STL is satisfied by signal `s`"
        if s is None:
            return False
        return all(STL.__primitives[i].satisfied(s) for i in self._indices)
    
    def get_children(self) -> Set[STL]:
        length = len(STL.__primitives)
        parents = set()
        for i in self._indices:
            parents.update(STL.__parents[i])
        return {STL(self._indices.union([i])) 
            for i in set(range(length)) - parents} - {self}

    def __len__(self):
        return len(self._indices)

    def __hash__(self):
        return hash(self._indices)

    def __repr__(self):
        if not len(self._indices):
            return 'T'
        return '^'.join(repr(STL.__primitives[i]) for i in self._indices)
