import numpy as np 
"""
Golden Section Search (GSS):
    used to find minimum or maximum of an uni-modal function 
    that contains only one minimum or maximum on interval [a,b]

Golden ratio:
    two quantites are in the golden ratio if their ratio is the same 
    as the ratio of their sum to the larger quantity

let a > b > 0, a is in a golden ratio to be if a / b =  (a + b)/ a

Golden ratio (φ)  =  a / b, then φ^2 = φ + 1                    = 1.61803 
Golden number (r) =  b / a, then r^2 + r - 1  = (sqrt(5) - 1)/2 = 0.61803

solution :
for min problem
1 - Determine the two intermediate points x , y:
            x =  a + r    (b - a), 
            y =  a + r**2 (b - a)

2 - Evaluate f(x), f(y):
            if   f(x) > f(y): a = y, y = x, f(y) = f(x), b = b, evaluate new x and f(x)
            eilf f(x) < f(y): b = x, x = y, f(x) = f(y), a = a, evaluate new y and f(y) 

3 -  if b - a < ε (a sufficiently small number), then the maximum occurs at (a + b)/ 2 and stop
"""

class GSS:
    def __init__(self, func, a, b, epsilon=1e-5, find_max=True , no_iter = 3):
        """
        Initialize the General Golden Section Search.
        
        Parameters:
        - func: The target function to optimize (minimize) and if maximize multiply by -1.
        - a, b: The interval [a, b].
        - epsilon(ε): The precision threshold (default 1e-5).
        - find_max: If True, multiply by -1 and find the minimum; otherwise, find the minimum.
        """
        self.func = func
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.find_max = find_max
        self.r = (np.sqrt(5) - 1) / 2  # Golden number
        self.no_iter = no_iter
        self.x = None
        self.y = None
        self.f_x = None
        self.f_y = None

    def fit(self):
        """Run the Golden Section Search."""
        # Initialize intermediate points (x, y)
        self.x = self.a + (self.r * (self.b - self.a))
        self.y = self.a + ((self.r**2) * (self.b - self.a))

        # Evaluate function values at initial points
        self.f_x = self.func(self.x)
        self.f_y = self.func(self.y)

        i = 0 # no of iteration

        # if find_max is true multiply the func by -1
        if self.find_max:
            self.f_x = -self.f_x
            self.f_y = -self.f_y
        
        print(f"Iteration 0:\na = {self.a}, b = {self.b}\nx = {self.x}, y = {self.y}\nf(x) = {self.f_x}, f(y) = {self.f_y}\n")
        while self.b - self.a > self.epsilon:
            
            if self.f_x < self.f_y:
                self.a = self.y
                self.y = self.x
                self.f_y = self.f_x
                self.x = self.a + self.r * (self.b - self.a)
                self.f_x = self.func(self.x)
                if self.find_max:
                    self.f_x = -self.f_x

            else:
                self.b = self.x
                self.x = self.y
                self.f_x = self.f_y
                self.y = self.a + (self.r**2) * (self.b - self.a)
                self.f_y = self.func(self.y)
                if self.find_max:
                    self.f_y = -self.f_y
            if self.no_iter == i :
                break
            i += 1
            print(f"Iteration {i}:\na = {self.a}, b = {self.b}\nx = {self.x}, y = {self.y}\nf(x) = {self.f_x}, f(y) = {self.f_y}\n")