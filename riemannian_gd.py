import numpy as np

def project_to_sphere(x):
    """Project a vector onto the unit sphere"""
    return x / np.linalg.norm(x)

def riemannian_gradient_descent(f, grad_f, x0, lr=0.1, epochs=100):
    """
    Riemannian Gradient Descent on the unit sphere S^{n-1}
    """
    x = project_to_sphere(x0)
    trajectory = [x.copy()]
    
    for _ in range(epochs):
        g = grad_f(x)
        # Project gradient to tangent space
        g_tangent = g - np.dot(g, x) * x
        # Take gradient step
        x = x - lr * g_tangent
        # Retract back to sphere
        x = project_to_sphere(x)
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)

# Example: Minimize f(x) = -x[0] on sphere
def f(x):
    return -x[0]

def grad_f(x):
    g = np.zeros_like(x)
    g[0] = -1
    return g

if __name__ == "__main__":
    np.random.seed(42)
    x0 = np.random.randn(3)
    x_opt, traj = riemannian_gradient_descent(f, grad_f, x0)
    
    print("Optimal point:", x_opt)
    print("Final loss:", f(x_opt))
    print("Riemannian GD on sphere completed!")
