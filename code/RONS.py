import sympy as sp
import numpy as np

class ValidatedCaseMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.validate()
        return obj

class Case(metaclass=ValidatedCaseMeta):
    '''
    Base Class for builing RONS cases
    '''
    def __init__(self):
        self.VARS       = None
        self.Q          = None
        self.ANSATZ     = None
        self.DANSATZ_DQ = None
        self.F_ANSATZ   = None
        self.Q0         = None

    def validate(self):
        '''
        Validate function.
        This function checks if the basic members are created and if the lengths
        match.
        '''
        all_self = [
            self.VARS,
            self.Q,
            self.ANSATZ,
            self.DANSATZ_DQ,
            self.F_ANSATZ,
            self.Q0
        ]
        isNone = [a is None for a in all_self]
        # print(f"Validating {self.__class__.__name__}: {isNone}")
        if any(isNone):
            raise("Not all essential variables and initialized. "
            "Check initialization function.")
        
        size_q = len(self.Q)
        size_dwdq = len(self.Q)
        size_q0 = len(self.Q0)

        if not(size_q==size_q0):
            raise("Different length symbolic q and initial data q0")
        
        if not(size_q==size_dwdq):
            raise("Different length symbolic q and d ansatz wrt q")
        
class VortexPair(Case):
    '''
    Vortex Pair Case.
    Two vortexes with equal stregths and signs.
    '''
    def __init__(self, data):
        super().__init__()

        self.N = 2
        self.x, self.y = sp.symbols('x y', real=True)
        self.VARS = (self.x, self.y)
                
        # Define parameters Q 
        self.A = sp.symbols("A", real=True)
        self.L = sp.symbols("L", real=True, positive=True)
        self.xc = [sp.symbols(f"x_c_{i+1}", real=True) for i in range(self.N)]
        self.yc = [sp.symbols(f"y_c_{i+1}", real=True) for i in range(self.N)]
      
        self.Q = [self.A, self.L]
        for i in range(self.N):
            self.Q.extend([self.xc[i], self.yc[i]])
                    
        # Build symbolic fields
        self.gamma = sum(
            self.A * sp.exp(-((self.x - self.xc[i])**2 + (self.y - self.yc[i])**2) / self.L**2)
            for i in range(self.N)
        )
        self.u = sp.Matrix([
            sp.diff(self.gamma, self.y),
            -sp.diff(self.gamma, self.x)
        ])

        self.w = (-sp.diff(self.gamma, self.x, 2) 
                      - sp.diff(self.gamma, self.y, 2)).doit()
        
        self.ANSATZ = self.w
        
        # Advection term: FU = -u . grad(w)
        grad_w = sp.Matrix([sp.diff(self.w, self.x), 
                            sp.diff(self.w, self.y)])
        self.F_ANSATZ = -(self.u[0]*grad_w[0] + self.u[1]*grad_w[1])
        
        # Derivatives wrt Q
        self.dwdq = [self.w.diff(q_i) for q_i in self.Q]

        self.DANSATZ_DQ = self.dwdq

        # Initial data
        self.Q0 = np.array(data)


class Ansatz:
    def __init__(self, case):
        self.VARS = case.VARS
        self.q_sym = case.Q
        self.ansatz_sym = case.ANSATZ
        self.dudq_sym = case.DANSATZ_DQ
        self.Fu_sym = case.F_ANSATZ
    
    def compile_numeric_functions(self):
        '''
        lambdify for all spatial coordinates and Q parameters.
        '''
        all_args = (self.x, self.y) + tuple(self.q_syms)
        
        dwdq_funcs = [sp.lambdify(all_args, expr, "numpy") 
                      for expr in self.dwdq_sym]
        Fu_func = sp.lambdify(all_args, self.FU_sym, "numpy")
        
        return dwdq_funcs, Fu_func


# will be deleted
if __name__ == "__main__":

    data = [1., 1., 1., 0., 1., 0.]

    v = VortexPair(data=data)
