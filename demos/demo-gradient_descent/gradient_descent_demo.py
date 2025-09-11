import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse

class GradientDescentDemo:
    def __init__(self, learning_rate=0.01, num_iterations=100, train_intercept=True, batch_size=None, num_samples=200):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.train_intercept = train_intercept
        self.batch_size = batch_size  # None means full batch (all data)
        
        # Generate sample data with non-zero intercept and higher variation
        np.random.seed(42)
        self.x_data = np.linspace(0, 10, num_samples)
        self.y_data = 2.5 * self.x_data + 3.5 + np.random.normal(0, 5, num_samples)
        
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
        
        # Pre-compute 3D loss surface for visualization
        self._compute_loss_surface()
        
    def compute_predictions(self, slope, intercept):
        if self.train_intercept:
            return slope * self.x_data + intercept
        else:
            return slope * self.x_data
    
    def compute_error(self, slope, intercept):
        predictions = self.compute_predictions(slope, intercept)
        return np.mean((predictions - self.y_data) ** 2)
    
    def compute_gradients(self, slope, intercept):
        # Stochastic gradient descent: use batch_size samples for GRADIENT computation only
        # Note: Loss surface visualization always uses full dataset for consistency
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
    
    def _compute_loss_surface(self):
        """Pre-compute the 3D loss surface for visualization"""
        if self.train_intercept:
            # Create meshgrid for parameter space (slope and intercept)
            slope_vals = np.linspace(self.slope_range[0], self.slope_range[1], 50)
            intercept_vals = np.linspace(self.intercept_range[0], self.intercept_range[1], 50)
            self.slope_mesh, self.intercept_mesh = np.meshgrid(slope_vals, intercept_vals)
            
            # Compute loss for each point in parameter space using FULL dataset
            self.loss_surface = np.zeros_like(self.slope_mesh)
            for i in range(self.slope_mesh.shape[0]):
                for j in range(self.slope_mesh.shape[1]):
                    slope = self.slope_mesh[i, j]
                    intercept = self.intercept_mesh[i, j]
                    # Use _compute_loss_at_point which always uses full dataset
                    self.loss_surface[i, j] = self._compute_loss_at_point(slope, intercept)
        else:
            # For no-intercept case, create 1D loss curve for slope only
            # Still create 2D arrays for compatibility, but intercept fixed at 0
            slope_vals = np.linspace(self.slope_range[0], self.slope_range[1], 50)
            intercept_vals = np.array([0])  # Fixed at 0
            self.slope_mesh, self.intercept_mesh = np.meshgrid(slope_vals, intercept_vals)
            
            # Compute loss for each slope value with intercept=0
            self.loss_surface = np.zeros_like(self.slope_mesh)
            for i in range(self.slope_mesh.shape[0]):
                for j in range(self.slope_mesh.shape[1]):
                    slope = self.slope_mesh[i, j]
                    # Always use intercept=0 for no-intercept case
                    self.loss_surface[i, j] = self._compute_loss_at_point(slope, 0)
    
    def _compute_loss_at_point(self, slope, intercept):
        """Compute loss at a specific parameter point using FULL dataset"""
        # Always use full dataset for loss computation, regardless of batch_size
        if self.train_intercept:
            predictions = slope * self.x_data + intercept
        else:
            predictions = slope * self.x_data
        return np.mean((predictions - self.y_data) ** 2)
    
    def _print_demo_header(self):
        """Print demo header with starting information"""
        batch_desc = f"batch size {self.batch_size}" if self.batch_size else "full batch"
        print(f"=== Gradient Descent Demo ({batch_desc}) ===")
        if self.batch_size:
            print("Note: Loss surface shows true loss on full dataset, gradients computed on mini-batches")
        if self.train_intercept:
            print(f"Starting: slope={self.slope:.3f}, intercept={self.intercept:.3f}")
            print(f"Target optimal: slope={self.optimal_params[1]:.3f}, intercept={self.optimal_params[0]:.3f}")
        else:
            print(f"Starting: slope={self.slope:.3f} (no intercept - forced through origin)")
            print(f"Target optimal (no intercept): slope={self.optimal_params[1]:.3f}")
            print(f"True optimal (with intercept): slope={self.true_optimal_params[1]:.3f}, intercept={self.true_optimal_params[0]:.3f}")
    
    def _create_visualization(self, step):
        """Create the four-panel visualization (2x2 grid)"""
        plt.clf()
        
        # Top-left plot: Data and fitting lines  
        plt.subplot(2, 2, 1)
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
            y_optimal = self.optimal_params[1] * np.linspace(self.x_data.min(), self.x_data.max(), 10) + self.optimal_params[0]
        else:
            y_optimal = self.optimal_params[1] * np.linspace(self.x_data.min(), self.x_data.max(), 10)
        plt.plot(np.linspace(self.x_data.min(), self.x_data.max(), 10), y_optimal, color='orange', linestyle='--', linewidth=2,
                marker='^', markersize=4, label='Optimal (current model)')
        
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
        
        # Top-right plot: Error over time
        plt.subplot(2, 2, 2)
        if len(self.errors_history) > 1:
            plt.plot(range(len(self.errors_history)), self.errors_history, 'b-', linewidth=2)
            plt.scatter([step], [self.errors_history[-1]], color='red', s=50, zorder=5)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Error Reduction Over Time')
        plt.grid(True, alpha=0.3)
        
        # Bottom-left plot: Parameter space exploration (with fixed dimensions)
        plt.subplot(2, 2, 3)
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
        
        # Bottom-right plot: 3D Loss Surface
        ax = plt.subplot(2, 2, 4, projection='3d')
        
        if len(self.slopes_history) > 1:
            if self.train_intercept:
                # Plot the 3D loss surface for intercept case
                surf = ax.plot_surface(self.slope_mesh, self.intercept_mesh, self.loss_surface, 
                                     cmap='coolwarm', alpha=0.6, linewidth=0, antialiased=True)
                
                # Plot the parameter path
                path_losses = [self._compute_loss_at_point(s, i) for s, i in 
                             zip(self.slopes_history, self.intercepts_history)]
                ax.plot(self.slopes_history, self.intercepts_history, path_losses, 
                       'ro-', linewidth=2, markersize=4, alpha=0.8, label='Gradient path')
                
                # Highlight current position
                current_loss = self._compute_loss_at_point(self.slope, self.intercept)
                ax.scatter([self.slope], [self.intercept], [current_loss], 
                         color='red', s=100, marker='*', zorder=5)
                
                # Mark optimal point
                optimal_loss = self._compute_loss_at_point(self.optimal_params[1], self.optimal_params[0])
                ax.scatter([self.optimal_params[1]], [self.optimal_params[0]], [optimal_loss], 
                         color='orange', s=100, marker='^', zorder=5)
                
                ax.set_xlabel('Slope')
                ax.set_ylabel('Intercept')
                ax.set_zlabel('Loss')
                ax.set_title('3D Loss Surface')
                ax.view_init(elev=30, azim=45)
            else:
                # For no-intercept case, show 3D surface with intercept=0 plane
                # This shows the true loss landscape when constrained to go through origin
                slope_vals = np.linspace(self.slope_range[0], self.slope_range[1], 100)
                intercept_vals = np.linspace(self.intercept_range[0], self.intercept_range[1], 50)
                slope_mesh_full, intercept_mesh_full = np.meshgrid(slope_vals, intercept_vals)
                
                # Compute full loss surface (including where intercept != 0)
                loss_surface_full = np.zeros_like(slope_mesh_full)
                for i in range(slope_mesh_full.shape[0]):
                    for j in range(slope_mesh_full.shape[1]):
                        loss_surface_full[i, j] = self._compute_loss_at_point(
                            slope_mesh_full[i, j], intercept_mesh_full[i, j])
                
                # Plot the full 3D surface
                surf = ax.plot_surface(slope_mesh_full, intercept_mesh_full, loss_surface_full, 
                                     cmap='coolwarm', alpha=0.3, linewidth=0, antialiased=True)
                
                # Highlight the constraint plane (intercept=0)
                constraint_slopes = np.linspace(self.slope_range[0], self.slope_range[1], 100)
                constraint_intercepts = np.zeros_like(constraint_slopes)
                constraint_losses = [self._compute_loss_at_point(s, 0) for s in constraint_slopes]
                ax.plot(constraint_slopes, constraint_intercepts, constraint_losses, 
                       'b-', linewidth=3, alpha=0.8, label='Constraint: intercept=0')
                
                # Plot parameter path (constrained to intercept=0)
                path_losses = [self._compute_loss_at_point(s, 0) for s in self.slopes_history]
                ax.plot(self.slopes_history, [0]*len(self.slopes_history), path_losses, 
                       'ro-', linewidth=2, markersize=4, alpha=0.8, label='Gradient path')
                
                # Current position
                current_loss = self._compute_loss_at_point(self.slope, 0)
                ax.scatter([self.slope], [0], [current_loss], 
                         color='red', s=100, marker='*', zorder=5)
                
                # Mark optimal point on constraint
                optimal_loss = self._compute_loss_at_point(self.optimal_params[1], 0)
                ax.scatter([self.optimal_params[1]], [0], [optimal_loss], 
                         color='orange', s=100, marker='^', zorder=5)
                
                # Mark true optimal (with intercept) to show what we're missing
                true_optimal_loss = self._compute_loss_at_point(self.true_optimal_params[1], self.true_optimal_params[0])
                ax.scatter([self.true_optimal_params[1]], [self.true_optimal_params[0]], [true_optimal_loss], 
                         color='purple', s=100, marker='s', zorder=5, label='True optimal')
                
                ax.set_xlabel('Slope')
                ax.set_ylabel('Intercept')
                ax.set_zlabel('Loss')
                ax.set_title('Loss Surface with Constraint')
                ax.view_init(elev=30, azim=45)
                ax.legend()
        
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
        
        plt.figure(figsize=(15, 12))
        
        for i in range(self.num_iterations):
            # Store current state
            self.slopes_history.append(self.slope)
            self.intercepts_history.append(self.intercept)
            # Always compute error on full dataset for visualization consistency
            current_error = self._compute_loss_at_point(self.slope, self.intercept)
            self.errors_history.append(current_error)
            
            # Create visualization
            self._create_visualization(i)
            plt.draw()
            plt.pause(0.05)
            
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
        print(f"Running automatic demo with 0.05 second delays...")
        print(f"Using {batch_desc}, learning rate {self.learning_rate}")
        self._print_demo_header()
        
        plt.figure(figsize=(15, 12))
        
        for i in range(self.num_iterations):
            # Store current state
            self.slopes_history.append(self.slope)
            self.intercepts_history.append(self.intercept)
            # Always compute error on full dataset for visualization consistency
            current_error = self._compute_loss_at_point(self.slope, self.intercept)
            self.errors_history.append(current_error)
            
            # Create visualization
            self._create_visualization(i)
            plt.draw()
            plt.pause(0.05)
            
            # Print current state
            self._print_step_info(i)
            
            if i < self.num_iterations - 1:
                # THE SIMPLE GRADIENT DESCENT LOOP!
                self._update_parameters()
        
        plt.show()
        self._print_final_results()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Interactive Gradient Descent Demo for Linear Regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gradient_descent_demo.py                           # Interactive mode
  python gradient_descent_demo.py --auto                    # Automatic mode with defaults
  python gradient_descent_demo.py --step-by-step --lr 0.1   # Step-by-step with learning rate 0.1
  python gradient_descent_demo.py --no-intercept --batch 10 # No intercept, batch size 10
  python gradient_descent_demo.py --auto --lr 0.01 --iter 200 --samples 500  # Full config
        """)
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--step-by-step', '-s', action='store_true',
                           help='Run step-by-step demo (press Enter for each step)')
    mode_group.add_argument('--auto', '-a', action='store_true',
                           help='Run automatic demo with timed updates')
    
    # Model parameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.05,
                       help='Learning rate (default: 0.05)')
    parser.add_argument('--iterations', '--iter', type=int, default=None,
                       help='Number of iterations (default: 50 for step-by-step, 100 for auto)')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of data samples (default: 200)')
    
    # Training options
    parser.add_argument('--no-intercept', action='store_true',
                       help='Force line through origin (no intercept/bias training)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size for stochastic gradient descent (default: full batch)')
    
    return parser.parse_args()


def run_demo_interactive():
    """Run the interactive demo (original behavior)"""
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
    learning_rate = float(lr_input) if lr_input else 0.05
    
    # Num samples
    ns_input = input("\nNumber of samples(default 200): ").strip()
    num_samples = int(ns_input) if ns_input else 200
    
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
        
        demo = GradientDescentDemo(
          learning_rate=learning_rate,
          num_iterations=num_iterations,
          train_intercept=train_intercept,
          batch_size=batch_size,
          num_samples=num_samples
        )
        demo.step_by_step_demo()
    else:
        # Automatic mode
        iter_input = input("\nNumber of iterations (default 100): ").strip()
        try:
            num_iterations = int(iter_input) if iter_input else 100
        except ValueError:
            num_iterations = 100
        
        demo = GradientDescentDemo(
          learning_rate=learning_rate,
          num_iterations=num_iterations,
          train_intercept=train_intercept,
          batch_size=batch_size,
          num_samples=num_samples
        )
        demo.automatic_demo()


def run_demo_with_args(args):
    """Run demo with command line arguments"""
    # Determine mode
    if args.step_by_step:
        step_by_step = True
        num_iterations = args.iterations if args.iterations else 50
    else:  # Default to auto mode
        step_by_step = False
        num_iterations = args.iterations if args.iterations else 100
    
    # Set up parameters
    train_intercept = not args.no_intercept
    
    print(f"Running gradient descent demo with:")
    print(f"  Mode: {'Step-by-step' if step_by_step else 'Automatic'}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Samples: {args.samples}")
    print(f"  Train intercept: {train_intercept}")
    print(f"  Batch size: {args.batch if args.batch else 'Full batch'}")
    print()
    
    # Create and run demo
    demo = GradientDescentDemo(
        learning_rate=args.learning_rate,
        num_iterations=num_iterations,
        train_intercept=train_intercept,
        batch_size=args.batch,
        num_samples=args.samples
    )
    
    if step_by_step:
        demo.step_by_step_demo()
    else:
        demo.automatic_demo()


def run_demo():
    """Main entry point - handle both interactive and command-line modes"""
    import sys
    
    # If no arguments provided, run interactive mode
    if len(sys.argv) == 1:
        run_demo_interactive()
    else:
        # Parse and run with command line arguments
        args = parse_arguments()
        run_demo_with_args(args)


if __name__ == "__main__":
    run_demo()
