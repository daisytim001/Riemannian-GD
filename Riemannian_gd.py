import numpy as np
import matplotlib.pyplot as plt

# Simple Riemannian GD on the sphere S^{n-1}
def riemannian_gd(n=10, steps=200, lr=0.1):
    # Random start on the sphere
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    losses = []
    A = np.random.randn(n, n)  # Toy quadratic objective: f(x) = -x^T A x
    A = (A + A.T) / 2          # Make A symmetric
    
    for t in range(steps):
        # Euclidean gradient
        grad_euclid = -2 * A @ x
        
        # Project to tangent space of sphere: grad_R = grad - (x^T grad)x
        grad_R = grad_euclid - np.dot(x, grad_euclid) * x
        
        # Riemannian GD step + retract back to sphere
        x = x - lr * grad_R
        x = x / np.linalg.norm(x)
        
        loss = -x.T @ A @ x  # We minimize, so loss goes down
        losses.append(loss)

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Riemannian GD on Sphere")
    plt.savefig("loss_curve.png")
    print("Saved plot to loss_curve.png")
    return losses

if __name__ == "__main__":
    riemannian_gd()
