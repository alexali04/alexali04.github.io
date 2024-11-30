import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

x = 30.0 * torch.rand(100)
y = 3.0 * x + 40.0 + torch.randn(100) * torch.sqrt(torch.tensor(5.0))

X = torch.stack([torch.ones(100), x], dim=1)

w_MLE = torch.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ w_MLE

sigma_2_mle = ((y - y_pred).T @ (y - y_pred)) / (100 - 2)
slope_str = f"Estimated Slope, Intercept, Noise Variance: {w_MLE[1]:.4f}, {w_MLE[0]:.4f}, {sigma_2_mle:.4f}, True Noise Variance: {5.0:.4f}"
# Estimated Slope: 2.9656, Intercept: 40.5160, Noise Variance: 3.3836, True Noise Variance: 5.0000

plt.title(slope_str, fontsize=10, color="black")
plt.plot(x, y, "o", label="Noisy Targets", color="blue")
plt.plot(x, y_pred, "-*",label="Predictions", color="red")
plt.legend()
plt.show()

