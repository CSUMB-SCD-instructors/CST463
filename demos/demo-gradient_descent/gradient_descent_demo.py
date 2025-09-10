import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class GradientDescentDemo:
    def __init__(self, learning_rate=0.01, num_iterations=100, train_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.train_intercept = train_intercept
        
        # Generate sample data with non-zero intercept and higher variation
        np.random.seed(42)
        self.x_data = np.linspace(0, 10, 20)
        self.y_data = 2.5 * self.x_data + 3.5 + np.random.normal(0, 1.8, 20)
        
        # Calculate optimal line using closed form solution
        # ALWAYS calculate the true best fit (with intercept) to show its importance
        X_full = np.column_stack([np.ones(len(self.x_data)), self.x_data])
        self.true_optimal_params = np.linalg.solve(X_full.T @ X_full, X_full.T @ self.y_data)
        
        if train_intercept:
            self.optimal_params = self.true_optimal_params
        else:
            # Force through origin - no intercept (but we'll show the true optimal too)
            X = self.x_data.reshape(-1, 1)
            self.optimal_params = [0, np.linalg.solve(X.T @ X, X.T @ self.y_data)[0]]
        
        # Initialize parameters (start far from optimal)
        self.slope = 0.0
        self.intercept = 0.0 if train_intercept else 0.0
        
        # Storage for visualization
        self.slopes_history = []
        self.intercepts_history = []
        self.errors_history = []
        
        # Pre-calculate parameter space bounds for consistent plotting
        self.slope_range = (-1.0, 6.0)  # Expected range for slope exploration
        self.intercept_range = (-5.0, 10.0)  # Expected range for intercept exploration
        
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
            print(f"Target optimal (no intercept): slope={self.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={self.true_optimal_params[1]:.3f}, intercept={self.true_optimal_params[0]:.3f}")
        print("\nPress Enter to see each step...")
        
        plt.figure(figsize=(20, 5))
        
        for i in range(self.num_iterations):
            # Store current state
            self.slopes_history.append(self.slope)
            self.intercepts_history.append(self.intercept)
            current_error = self.compute_error(self.slope, self.intercept)
            self.errors_history.append(current_error)
            
            # Create visualization
            plt.clf()
            
            # Left plot: Data and fitting lines  
            plt.subplot(1, 3, 1)
            plt.scatter(self.x_data, self.y_data, c='blue', alpha=0.7, label='Data')
            
            # Show faded history of previous lines
            for j, (old_slope, old_intercept) in enumerate(zip(self.slopes_history[:-1], self.intercepts_history[:-1])):
                alpha = max(0.1, 0.8 * (j / len(self.slopes_history)))
                y_line = old_slope * self.x_data + old_intercept
                plt.plot(self.x_data, y_line, 'gray', alpha=alpha, linewidth=1)
            
            # Current line in bright red
            y_current = self.slope * self.x_data + self.intercept
            plt.plot(self.x_data, y_current, 'red', linewidth=3, label=f'Current fit (step {i})')
            
            # Target optimal line for current model (orange)
            if self.train_intercept:
                y_optimal = self.optimal_params[1] * self.x_data + self.optimal_params[0]
            else:
                y_optimal = self.optimal_params[1] * self.x_data
            plt.plot(self.x_data, y_optimal, color='orange', linestyle='--', linewidth=2, 
                    marker='^', markersize=4, markevery=3, label='Optimal (current model)')
            
            # True optimal line with intercept (purple) to show importance of intercept
            if not self.train_intercept:
                y_true_optimal = self.true_optimal_params[1] * self.x_data + self.true_optimal_params[0]
                plt.plot(self.x_data, y_true_optimal, color='purple', linestyle='-.', linewidth=2,
                        marker='s', markersize=3, markevery=4, label='True optimal (with intercept)')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            if self.train_intercept:
                plt.title(f'Step {i}: slope={self.slope:.3f}, intercept={self.intercept:.3f}')
            else:
                plt.title(f'Step {i}: slope={self.slope:.3f} (no intercept)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Middle plot: Error over time
            plt.subplot(1, 3, 2)
            if len(self.errors_history) > 1:
                plt.plot(range(len(self.errors_history)), self.errors_history, 'b-', linewidth=2)
                plt.scatter([i], [current_error], color='red', s=50, zorder=5)
            plt.xlabel('Iteration')
            plt.ylabel('Mean Squared Error')
            plt.title('Error Reduction Over Time')
            plt.grid(True, alpha=0.3)
            
            # Right plot: Parameter space exploration (with fixed dimensions)
            plt.subplot(1, 3, 3)
            if len(self.slopes_history) > 1:
                if self.train_intercept:
                    plt.scatter(self.slopes_history[:-1], self.intercepts_history[:-1], 
                              c=range(len(self.slopes_history)-1), cmap='Blues', s=30, alpha=0.7)
                    plt.scatter([self.slope], [self.intercept], color='red', s=100, 
                              marker='*', zorder=5, label='Current')
                    plt.scatter([self.optimal_params[1]], [self.optimal_params[0]], 
                              color='orange', s=100, marker='^', zorder=5, label='Target')
                    plt.xlim(self.slope_range)
                    plt.ylim(self.intercept_range)
                    plt.xlabel('Slope')
                    plt.ylabel('Intercept')
                    plt.title('Parameter Space Exploration')
                else:
                    plt.scatter(self.slopes_history[:-1], [0]*len(self.slopes_history[:-1]), 
                              c=range(len(self.slopes_history)-1), cmap='Blues', s=30, alpha=0.7)
                    plt.scatter([self.slope], [0], color='red', s=100, 
                              marker='*', zorder=5, label='Current')
                    plt.scatter([self.optimal_params[1]], [0], color='orange', s=100, 
                              marker='^', zorder=5, label='Target (no intercept)')
                    plt.scatter([self.true_optimal_params[1]], [self.true_optimal_params[0]], 
                              color='purple', s=100, marker='s', zorder=5, label='True optimal')
                    plt.xlim(self.slope_range)
                    plt.ylim(-1, 10)
                    plt.xlabel('Slope')
                    plt.ylabel('Intercept (fixed at 0)')
                    plt.title('Parameter Space (Slope Only)')
                plt.legend()
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
            print(f"Target (no intercept): slope={self.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={self.true_optimal_params[1]:.3f}, intercept={self.true_optimal_params[0]:.3f}")
            print(f"Notice how much better the true optimal line fits the data!")

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
        demo = GradientDescentDemo(learning_rate=0.05, num_iterations=200, train_intercept=train_intercept)
        demo.step_by_step_demo()
    else:
        demo = GradientDescentDemo(learning_rate=0.05, num_iterations=200, train_intercept=train_intercept)
        
        print("Running automatic demo with 0.05 second delays...")
        if demo.train_intercept:
            print(f"Starting: slope={demo.slope:.3f}, intercept={demo.intercept:.3f}")
            print(f"Target optimal: slope={demo.optimal_params[1]:.3f}, intercept={demo.optimal_params[0]:.3f}")
        else:
            print(f"Starting: slope={demo.slope:.3f} (no intercept - forced through origin)")
            print(f"Target optimal (no intercept): slope={demo.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={demo.true_optimal_params[1]:.3f}, intercept={demo.true_optimal_params[0]:.3f}")
        
        # Use the SAME logic as step-by-step mode to ensure consistency
        plt.figure(figsize=(20, 5))
        
        for i in range(demo.num_iterations):
            # Store current state
            demo.slopes_history.append(demo.slope)
            demo.intercepts_history.append(demo.intercept)
            current_error = demo.compute_error(demo.slope, demo.intercept)
            demo.errors_history.append(current_error)
            
            # Update visualization every step
            plt.clf()
            
            # Data and lines plot
            plt.subplot(1, 3, 1)
            plt.scatter(demo.x_data, demo.y_data, c='blue', alpha=0.7, label='Data')
            
            # Show faded history
            for j, (old_slope, old_intercept) in enumerate(zip(demo.slopes_history[:-1], demo.intercepts_history[:-1])):
                alpha = max(0.1, 0.8 * (j / len(demo.slopes_history)))
                if demo.train_intercept:
                    y_line = old_slope * demo.x_data + old_intercept
                else:
                    y_line = old_slope * demo.x_data
                plt.plot(demo.x_data, y_line, 'gray', alpha=alpha, linewidth=1)
            
            # Current line
            if demo.train_intercept:
                y_current = demo.slope * demo.x_data + demo.intercept
            else:
                y_current = demo.slope * demo.x_data
            plt.plot(demo.x_data, y_current, 'red', linewidth=3, label=f'Current fit (step {i})')
            
            # Target lines
            if demo.train_intercept:
                y_optimal = demo.optimal_params[1] * demo.x_data + demo.optimal_params[0]
            else:
                y_optimal = demo.optimal_params[1] * demo.x_data
            plt.plot(demo.x_data, y_optimal, color='orange', linestyle='--', linewidth=2, 
                    marker='^', markersize=4, markevery=3, label='Optimal (current model)')
            
            if not demo.train_intercept:
                y_true = demo.true_optimal_params[1] * demo.x_data + demo.true_optimal_params[0]
                plt.plot(demo.x_data, y_true, color='purple', linestyle='-.', linewidth=2,
                        marker='s', markersize=3, markevery=4, label='True optimal (with intercept)')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            if demo.train_intercept:
                plt.title(f'Step {i}: slope={demo.slope:.3f}, intercept={demo.intercept:.3f}')
            else:
                plt.title(f'Step {i}: slope={demo.slope:.3f} (no intercept)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Error plot
            plt.subplot(1, 3, 2)
            if len(demo.errors_history) > 1:
                plt.plot(range(len(demo.errors_history)), demo.errors_history, 'b-', linewidth=2)
                plt.scatter([i], [current_error], color='red', s=50, zorder=5)
            plt.xlabel('Iteration')
            plt.ylabel('Mean Squared Error')
            plt.title('Error Reduction Over Time')
            plt.grid(True, alpha=0.3)
            
            # Parameter space (with fixed dimensions)
            plt.subplot(1, 3, 3)
            if demo.train_intercept:
                plt.scatter(demo.slopes_history, demo.intercepts_history, 
                          c=range(len(demo.slopes_history)), cmap='Blues', s=20, alpha=0.7)
                plt.scatter([demo.slope], [demo.intercept], color='red', s=100, marker='*', zorder=5)
                plt.scatter([demo.optimal_params[1]], [demo.optimal_params[0]], 
                          color='orange', s=100, marker='^', zorder=5)
                plt.xlim(demo.slope_range)
                plt.ylim(demo.intercept_range)
                plt.xlabel('Slope')
                plt.ylabel('Intercept')
                plt.title('Parameter Space')
            else:
                plt.scatter(demo.slopes_history, [0]*len(demo.slopes_history), 
                          c=range(len(demo.slopes_history)), cmap='Blues', s=20, alpha=0.7)
                plt.scatter([demo.slope], [0], color='red', s=100, marker='*', zorder=5)
                plt.scatter([demo.optimal_params[1]], [0], color='orange', s=100, marker='^', zorder=5)
                plt.scatter([demo.true_optimal_params[1]], [demo.true_optimal_params[0]], 
                          color='purple', s=100, marker='s', zorder=5)
                plt.xlim(demo.slope_range)
                plt.ylim(-1, 10)
                plt.xlabel('Slope')
                plt.ylabel('Intercept')
                plt.title('Parameter Space')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.05)
            
            # Print current state
            if demo.train_intercept:
                print(f"Step {i}: slope={demo.slope:.4f}, intercept={demo.intercept:.4f}, error={current_error:.4f}")
            else:
                print(f"Step {i}: slope={demo.slope:.4f}, error={current_error:.4f}")
            
            if i < demo.num_iterations - 1:
                # THE SIMPLE GRADIENT DESCENT LOOP! (Same as manual mode)
                slope_grad, intercept_grad = demo.compute_gradients(demo.slope, demo.intercept)
                demo.slope = demo.slope - demo.learning_rate * slope_grad
                if demo.train_intercept:
                    demo.intercept = demo.intercept - demo.learning_rate * intercept_grad
        
        plt.show()
        print("\nDemo complete!")
        if demo.train_intercept:
            print(f"Final: slope={demo.slope:.3f}, intercept={demo.intercept:.3f}")
            print(f"Target: slope={demo.optimal_params[1]:.3f}, intercept={demo.optimal_params[0]:.3f}")
        else:
            print(f"Final: slope={demo.slope:.3f} (no intercept)")
            print(f"Target (no intercept): slope={demo.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={demo.true_optimal_params[1]:.3f}, intercept={demo.true_optimal_params[0]:.3f}")
            print(f"Notice how much better the true optimal line fits the data!")
        return

if __name__ == "__main__":
    run_demo()