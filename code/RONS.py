import sympy as sp

class Case:
    def __init__(self):
        self.vars   = []
        self.q      = []
        self.ansatz = []
        self.dudq   = []
        self.Fu     = []


class Vortexes(Case):
    def __init__(self):
        self.N = 2
        self.x, self.y = sp.symbols('x y', real=True)
        self.vars = (self.x, self.y)
                
        # Define parameters q = [A, L, x_c1, y_c1, ...]
        self.A = sp.symbols("A", real=True)
        self.L = sp.symbols("L", real=True, positive=True)
        self.xc = [sp.symbols(f"x_c_{i+1}", real=True) for i in range(self.N)]
        self.yc = [sp.symbols(f"y_c_{i+1}", real=True) for i in range(self.N)]
      
        self.q = [self.A, self.L]
        for i in range(self.N):
            self.q.extend([self.xc[i], self.yc[i]])
                    
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
        
        # Advection term: FU = -u . grad(w)
        grad_w = sp.Matrix([sp.diff(self.w, self.x), 
                            sp.diff(self.w, self.y)])
        self.Fu = -(self.u[0]*grad_w[0] + self.u[1]*grad_w[1])
        
        # Derivatives wrt q
        self.dwdq = [self.w.diff(q_i) for q_i in self.q]

class Ansatz:
    def __init__(self, case=Case()):
        self.vars = case.vars
        self.q = case.q
        self.ansatz = case.ansatz
        self.dudq = case.dudq
        self.Fu = case.Fu

if __name__ == "__main__":

    v = Vortexes()

    print(type(Case().Fu))
    print(type(v.Fu))