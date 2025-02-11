import numpy as np

class Batch_GD:
    def __init__(self, data, target, learning_rate=0.1, tolerance=1e-6, max_iters=1000):
        self.data = np.concatenate((np.ones((len(data), 1)), data), axis=1) 
        self.target = np.array(target).ravel()  # convert to 1D
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.initialize_parameters()
        
    def initialize_parameters(self):
        self.no_of_parameter = self.data.shape[1]
        self.parameters = np.zeros((self.no_of_parameter,))
        
    def predicted_value(self):
        return self.data @ self.parameters
        
    def compute_cost(self):
        errors = self.predicted_value() - self.target
        return np.mean(errors**2) / 2  # MSE
    
    def compute_gradients(self):
        errors = (self.predicted_value() - self.target).reshape(-1, 1)  # ensure for correct dim
        gradients = (self.data.T @ errors) / len(self.target)  
        return gradients.ravel()
    
    def update_parameters(self):
        gradients = self.compute_gradients()
        self.parameters -= self.learning_rate * gradients

    def fit(self):
        for _ in range(self.max_iters):
            gradients = self.compute_gradients()
            norm = np.linalg.norm(gradients)  
            if norm < self.tolerance:
                print(f"Stopped early at iteration {_}, gradient norm: {norm:.6f}")
                break
            self.update_parameters()
            
    def predict(self, new_data):
        new_data = np.concatenate((np.ones((len(new_data), 1)), new_data), axis=1)
        return new_data @ self.parameters

