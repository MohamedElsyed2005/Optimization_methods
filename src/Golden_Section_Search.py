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
step 1: 
1 - Determine the two intermediate points x , y:
            x =  a + r( b - a), y =  a + r^2 (b - a)

step 2: 
2 - Evaluate f(x), f(y):
            if   f(x) > f(y): a = y, y = x, f(y) = f(x), b = b, evaluate new x and f(x)
            eilf f(x) < f(y): b = x, x = y, f(x) = f(y), a = a, evaluate new y and f(y) 

step 3:
3 -  if b - a < ε (a sufficiently small number), then the maximum occurs at (a + b)/ 2 and stop
"""




"""
Consider the problem of finding the angle 𝜃 which maximizes the function 
f(𝜃)= 4(1 + cos𝜃) sin𝜃 
Compute three iterations of the golden  section search algorithm using the initial interval 
[0, 𝜋/2]. 
"""

import numpy as np

class GSS:
    def __init__(self, start_interval, end_interval):
        self.a = start_interval
        self.b = end_interval
        self.r = (np.sqrt(5) - 1) / 2
        self.epsilon = 1e-5
        self.x = self.a + (self.r * (self.b - self.a))
        self.y = self.a + ((1 - self.r) * (self.b - self.a))
        self.f_x_val = None
        self.f_y_val = None

    
    def f(self, theta):
        return 4 * np.sin(theta) * (1 + np.cos(theta))

    def fit(self):
        self.f_x_val = -self.f(self.x)
        self.f_y_val = -self.f(self.y)
        i = 0

        while self.b - self.a > self.epsilon:
            print(f"Iteration {i}:\na = {self.a}, b = {self.b}\nx = {self.x}, y = {self.y}\nf(x) = {self.f_x_val}, f(y) = {self.f_y_val}\n")

            if self.f_x_val < self.f_y_val:
                self.a = self.y
                self.y = self.x
                self.f_y_val = self.f_x_val
                self.x = self.a + (self.r * (self.b - self.a))
                self.f_x_val = -self.f(self.x)
                
            else:
                self.b = self.x
                self.x = self.y
                self.f_x_val = self.f_y_val
                self.y = self.a + ((1 - self.r) * (self.b - self.a))
                self.f_y_val = -self.f(self.y)
                
            i += 1
        maximum = (self.a + self.b) / 2
        print(f"Maximum occurs at θ = {maximum}, f(θ) = {self.f(maximum)}")

gss = GSS(0, np.pi / 2)
gss.fit()