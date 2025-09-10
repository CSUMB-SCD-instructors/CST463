import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class GradientDescentDemo:
    def __init__(self, learning_rate=0.01, num_iterations=100, train_intercept=True, batch_size=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.train_intercept = train_intercept
        self.batch_size = batch_size  # None means full batch (all data)
        
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
        # Stochastic gradient descent: use batch_size samples
        if self.batch_size is None or self.batch_size >= len(self.x_data):
            # Full batch gradient descent
            x_batch = self.x_data
            y_batch = self.y_data
        else:
            # Sample a random batch
            indices = np.random.choice(len(self.x_data), size=self.batch_size, replace=False)
            x_batch = self.x_data[indices]
            y_batch = self.y_data[indices]
        
        if self.train_intercept:
            predictions = slope * x_batch + intercept
        else:
            predictions = slope * x_batch
        
        errors = predictions - y_batch
        
        slope_gradient = np.mean(errors * x_batch)
        intercept_gradient = np.mean(errors) if self.train_intercept else 0
        
        return slope_gradient, intercept_gradient
    
    def _print_demo_header(self):
        """Print demo header with starting information"""
        batch_desc = f"batch size {self.batch_size}" if self.batch_size else "full batch"
        print(f"=== Gradient Descent Demo ({batch_desc}) ===")
        if self.train_intercept:
            print(f"Starting: slope={self.slope:.3f}, intercept={self.intercept:.3f}")
            print(f"Target optimal: slope={self.optimal_params[1]:.3f}, intercept={self.optimal_params[0]:.3f}")
        else:
            print(f"Starting: slope={self.slope:.3f} (no intercept - forced through origin)")
            print(f"Target optimal (no intercept): slope={self.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={self.true_optimal_params[1]:.3f}, intercept={self.true_optimal_params[0]:.3f}")
    
    def _create_visualization(self, step):
        """Create the three-panel visualization"""
        plt.clf()
        
        # Left plot: Data and fitting lines  
        plt.subplot(1, 3, 1)
        plt.scatter(self.x_data, self.y_data, c='blue', alpha=0.7, label='Data')
        
        # Show faded history of previous lines
        for j, (old_slope, old_intercept) in enumerate(zip(self.slopes_history[:-1], self.intercepts_history[:-1])):
            alpha = max(0.1, 0.8 * (j / len(self.slopes_history)))
            if self.train_intercept:
                y_line = old_slope * self.x_data + old_intercept
            else:
                y_line = old_slope * self.x_data
            plt.plot(self.x_data, y_line, 'gray', alpha=alpha, linewidth=1)
        
        # Current line in bright red
        if self.train_intercept:
            y_current = self.slope * self.x_data + self.intercept
        else:
            y_current = self.slope * self.x_data
        plt.plot(self.x_data, y_current, 'red', linewidth=3, label=f'Current fit (step {step})')
        
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
            plt.title(f'Step {step}: slope={self.slope:.3f}, intercept={self.intercept:.3f}')
        else:
            plt.title(f'Step {step}: slope={self.slope:.3f} (no intercept)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Middle plot: Error over time
        plt.subplot(1, 3, 2)
        if len(self.errors_history) > 1:
            plt.plot(range(len(self.errors_history)), self.errors_history, 'b-', linewidth=2)
            plt.scatter([step], [self.errors_history[-1]], color='red', s=50, zorder=5)
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
    
    def _print_step_info(self, step):
        """Print current step information"""
        batch_desc = f" (batch size {self.batch_size})" if self.batch_size else " (full batch)"
        if self.train_intercept:
            print(f"Step {step}: slope={self.slope:.4f}, intercept={self.intercept:.4f}, error={self.errors_history[-1]:.4f}{batch_desc}")
        else:
            print(f"Step {step}: slope={self.slope:.4f}, error={self.errors_history[-1]:.4f}{batch_desc}")
    
    def _print_final_results(self):
        """Print final demo results"""
        print("\nDemo complete!")
        if self.train_intercept:
            print(f"Final: slope={self.slope:.3f}, intercept={self.intercept:.3f}")
            print(f"Target: slope={self.optimal_params[1]:.3f}, intercept={self.optimal_params[0]:.3f}")
        else:
            print(f"Final: slope={self.slope:.3f} (no intercept)")
            print(f"Target (no intercept): slope={self.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={self.true_optimal_params[1]:.3f}, intercept={self.true_optimal_params[0]:.3f}")
            print(f"Notice how much better the true optimal line fits the data!")
    
    def _update_parameters(self):
        """Perform one gradient descent step"""
        slope_grad, intercept_grad = self.compute_gradients(self.slope, self.intercept)
        self.slope = self.slope - self.learning_rate * slope_grad
        if self.train_intercept:
            self.intercept = self.intercept - self.learning_rate * intercept_grad
    
    def step_by_step_demo(self):
        """Interactive step-by-step demonstration"""
        self._print_demo_header()
        print("\nPress Enter to see each step...")
        
        plt.figure(figsize=(20, 5))
        
        for i in range(self.num_iterations):
            # Store current state
            self.slopes_history.append(self.slope)
            self.intercepts_history.append(self.intercept)
            current_error = self.compute_error(self.slope, self.intercept)
            self.errors_history.append(current_error)
            
            # Create visualization
            self._create_visualization(i)
            plt.draw()
            plt.pause(0.1)
            
            # Print current state
            self._print_step_info(i)
            
            if i < self.num_iterations - 1:
                input("Press Enter for next step...")
                # THE SIMPLE GRADIENT DESCENT LOOP!
                self._update_parameters()
        
        plt.show()
        self._print_final_results()
    
    def automatic_demo(self):
        """Automatic demonstration with timed updates"""
        batch_desc = f"batch size {self.batch_size}" if self.batch_size else "full batch"
        print(f"Running automatic demo with 0.125 second delays...")
        print(f"Using {batch_desc}, learning rate {self.learning_rate}")
        self._print_demo_header()
        
        plt.figure(figsize=(20, 5))
        
        for i in range(self.num_iterations):
            # Store current state
            self.slopes_history.append(self.slope)
            self.intercepts_history.append(self.intercept)
            current_error = self.compute_error(self.slope, self.intercept)
            self.errors_history.append(current_error)
            
            # Create visualization
            self._create_visualization(i)
            plt.draw()
            plt.pause(0.125)
            
            # Print current state
            self._print_step_info(i)
            
            if i < self.num_iterations - 1:
                # THE SIMPLE GRADIENT DESCENT LOOP!
                self._update_parameters()
        
        plt.show()
        self._print_final_results()


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
    
    # Learning rate selection
    lr_input = input("\nLearning rate (default 0.05): ").strip()
    try:
        learning_rate = float(lr_input) if lr_input else 0.05
    except ValueError:
        learning_rate = 0.05
    
    # Batch size selection
    print("\nGradient descent type:")
    print("1. Full batch (use all data points)")
    print("2. Stochastic (use mini-batches)")
    
    batch_choice = input("Enter choice (1 or 2): ").strip()
    if batch_choice == "2":
        batch_input = input("Batch size (default 5): ").strip()
        try:
            batch_size = int(batch_input) if batch_input else 5
        except ValueError:
            batch_size = 5
    else:
        batch_size = None
    
    if choice == "1":
        # Manual mode
        iter_input = input("\nNumber of iterations (default 50): ").strip()
        try:
            num_iterations = int(iter_input) if iter_input else 50
        except ValueError:
            num_iterations = 50
        
        demo = GradientDescentDemo(learning_rate=learning_rate, num_iterations=num_iterations, 
                                 train_intercept=train_intercept, batch_size=batch_size)
        demo.step_by_step_demo()
    else:
        # Automatic mode
        iter_input = input("\nNumber of iterations (default 100): ").strip()
        try:
            num_iterations = int(iter_input) if iter_input else 100
        except ValueError:
            num_iterations = 100
        
        demo = GradientDescentDemo(learning_rate=learning_rate, num_iterations=num_iterations, 
                                 train_intercept=train_intercept, batch_size=batch_size)
        demo.automatic_demo()

if __name__ == "__main__":
    run_demo()
