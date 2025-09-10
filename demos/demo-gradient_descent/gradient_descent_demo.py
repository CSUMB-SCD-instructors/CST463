import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class GradientDescentDemo:
    def __init__(self, learning_rate=0.01, num_iterations=100, train_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.train_intercept = train_intercept
        
        # Generate sample data with non-zero intercept
        np.random.seed(42)
        self.x_data = np.linspace(0, 10, 20)
        self.y_data = 2.5 * self.x_data + 3.5 + np.random.normal(0, 1, 20)
        
        # Calculate optimal line using closed form solution
        if train_intercept:
            X = np.column_stack([np.ones(len(self.x_data)), self.x_data])
            self.optimal_params = np.linalg.solve(X.T @ X, X.T @ self.y_data)
        else:
            # Force through origin - no intercept
            X = self.x_data.reshape(-1, 1)
            self.optimal_params = [0, np.linalg.solve(X.T @ X, X.T @ self.y_data)[0]]
        
        # Initialize parameters (start far from optimal)
        self.slope = 0.0
        self.intercept = 0.0 if train_intercept else 0.0
        
        # Storage for visualization
        self.slopes_history = []
        self.intercepts_history = []
        self.errors_history = []
        
    def compute_predictions(self, slope, intercept):
        if self.train_intercept:
            return slope * self.x_data + intercept
        else:
            return slope * self.x_data
    
    def compute_error(self, slope, intercept):
        predictions = self.compute_predictions(slope, intercept)
        return np.mean((predictions - self.y_data) ** 2)
    
    def compute_gradients(self, slope, intercept):
        predictions = self.compute_predictions(slope, intercept)
        errors = predictions - self.y_data
        
        slope_gradient = np.mean(errors * self.x_data)
        intercept_gradient = np.mean(errors) if self.train_intercept else 0
        
        return slope_gradient, intercept_gradient
    
    def step_by_step_demo(self):
        """Interactive step-by-step demonstration"""
        print("=== Gradient Descent Step-by-Step Demo ===")
        if self.train_intercept:
            print(f"Starting: slope={self.slope:.3f}, intercept={self.intercept:.3f}")
            print(f"Target optimal: slope={self.optimal_params[1]:.3f}, intercept={self.optimal_params[0]:.3f}")
        else:
            print(f"Starting: slope={self.slope:.3f} (no intercept - forced through origin)")
            print(f"Target optimal: slope={self.optimal_params[1]:.3f} (no intercept)")
        print("\nPress Enter to see each step...")
        
        plt.figure(figsize=(15, 5))
        
        for i in range(self.num_iterations):
            # Store current state
            self.slopes_history.append(self.slope)
            self.intercepts_history.append(self.intercept)
            current_error = self.compute_error(self.slope, self.intercept)
            self.errors_history.append(current_error)
            
            # Create visualization
            plt.clf()
            
            # Left plot: Data and fitting lines
            plt.subplot(1, 2, 1)
            plt.scatter(self.x_data, self.y_data, c='blue', alpha=0.7, label='Data')
            
            # Show faded history of previous lines
            for j, (old_slope, old_intercept) in enumerate(zip(self.slopes_history[:-1], self.intercepts_history[:-1])):
                alpha = max(0.1, 0.8 * (j / len(self.slopes_history)))
                y_line = old_slope * self.x_data + old_intercept
                plt.plot(self.x_data, y_line, 'gray', alpha=alpha, linewidth=1)
            
            # Current line in bright red
            y_current = self.slope * self.x_data + self.intercept
            plt.plot(self.x_data, y_current, 'red', linewidth=3, label=f'Current fit (step {i})')
            
            # Target optimal line (colorblind-friendly: orange with triangles)
            if self.train_intercept:
                y_optimal = self.optimal_params[1] * self.x_data + self.optimal_params[0]
            else:
                y_optimal = self.optimal_params[1] * self.x_data
            plt.plot(self.x_data, y_optimal, color='orange', linestyle='--', linewidth=2, 
                    marker='^', markersize=4, markevery=3, label='Optimal fit')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Step {i}: slope={self.slope:.3f}, intercept={self.intercept:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Right plot: Error over time
            plt.subplot(1, 2, 2)
            if len(self.errors_history) > 1:
                plt.plot(range(len(self.errors_history)), self.errors_history, 'b-', linewidth=2)
                plt.scatter([i], [current_error], color='red', s=50, zorder=5)
            plt.xlabel('Iteration')
            plt.ylabel('Mean Squared Error')
            plt.title('Error Reduction Over Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
            # Print current state
            if self.train_intercept:
                print(f"Step {i}: slope={self.slope:.4f}, intercept={self.intercept:.4f}, error={current_error:.4f}")
            else:
                print(f"Step {i}: slope={self.slope:.4f}, error={current_error:.4f}")
            
            if i < self.num_iterations - 1:
                input("Press Enter for next step...")
                
                # THE SIMPLE GRADIENT DESCENT LOOP!
                slope_grad, intercept_grad = self.compute_gradients(self.slope, self.intercept)
                self.slope = self.slope - self.learning_rate * slope_grad
                if self.train_intercept:
                    self.intercept = self.intercept - self.learning_rate * intercept_grad
        
        plt.show()
        print("\nDemo complete!")
        if self.train_intercept:
            print(f"Final: slope={self.slope:.3f}, intercept={self.intercept:.3f}")
            print(f"Target: slope={self.optimal_params[1]:.3f}, intercept={self.optimal_params[0]:.3f}")
        else:
            print(f"Final: slope={self.slope:.3f} (no intercept)")
            print(f"Target: slope={self.optimal_params[1]:.3f} (no intercept)")

def run_demo():
    """Run the interactive demo"""
    print("Choose demo mode:")
    print("1. Step-by-step (press Enter for each step)")
    print("2. Quick run (automatic)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    print("\nTrain intercept/bias?")
    print("1. Yes - train both slope and intercept (full linear regression)")
    print("2. No - only train slope, force line through origin")
    
    intercept_choice = input("Enter choice (1 or 2): ").strip()
    train_intercept = intercept_choice != "2"
    
    if choice == "1":
        demo = GradientDescentDemo(learning_rate=0.05, num_iterations=50, train_intercept=train_intercept)
        demo.step_by_step_demo()
    else:
        demo = GradientDescentDemo(learning_rate=0.1, num_iterations=100, train_intercept=train_intercept)
        
        # Run gradient descent
        for i in range(demo.num_iterations):
            demo.slopes_history.append(demo.slope)
            demo.intercepts_history.append(demo.intercept)
            error = demo.compute_error(demo.slope, demo.intercept)
            demo.errors_history.append(error)
            
            if i < demo.num_iterations - 1:
                slope_grad, intercept_grad = demo.compute_gradients(demo.slope, demo.intercept)
                demo.slope = demo.slope - demo.learning_rate * slope_grad
                demo.intercept = demo.intercept - demo.learning_rate * intercept_grad
        
        # Show final visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(demo.x_data, demo.y_data, c='blue', alpha=0.7, label='Data')
        
        # Show evolution of lines
        for j, (slope, intercept) in enumerate(zip(demo.slopes_history[::5], demo.intercepts_history[::5])):
            alpha = 0.3 + 0.7 * (j / len(demo.slopes_history[::5]))
            y_line = slope * demo.x_data + intercept
            color = plt.cm.Reds(alpha)
            plt.plot(demo.x_data, y_line, color=color, linewidth=1)
        
        # Final line
        y_final = demo.slope * demo.x_data + demo.intercept
        plt.plot(demo.x_data, y_final, 'red', linewidth=3, label='Final fit')
        
        # Optimal line (colorblind-friendly)
        if demo.train_intercept:
            y_optimal = demo.optimal_params[1] * demo.x_data + demo.optimal_params[0]
        else:
            y_optimal = demo.optimal_params[1] * demo.x_data
        plt.plot(demo.x_data, y_optimal, color='orange', linestyle='--', linewidth=2, 
                marker='^', markersize=4, markevery=3, label='Optimal fit')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Gradient Descent Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(demo.errors_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Error Reduction')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_demo()