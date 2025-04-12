import matplotlib.pyplot as plt
import numpy as np

draft_length = [5, 10, 20, 30, 40]
accept_length = [1.1467, 1.4066, 1.7574, 1.9173, 1.9409]

draft_length = draft_length[:-1]
accept_length = accept_length[:-1]

x = np.log(np.array(draft_length))
y = np.array(accept_length)
A = np.column_stack((x, np.ones_like(x)))
coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, y, rcond=None)
a, b = coefficients
z_fit = a * x + b

ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum((y - z_fit)**2)  # 直接计算残差平方和，而不是依赖residuals返回值
r_squared = 1 - (ss_residual / ss_total)
mae = np.mean(np.abs(y - z_fit))
rmse = np.sqrt(np.mean((y - z_fit)**2))

print(f"Equation: y = {a:.15f}x + {b:.15f}")
print(f"R²: {r_squared:.7f}")
print(f"MAE: {mae:.7f}")
print(f"RMSE: {rmse:.7f}")

plt.plot(draft_length, accept_length, label="Actual Values")
plt.scatter(draft_length, accept_length, label="Actual Values")
plt.plot(draft_length, z_fit, label="Fitted Values")
plt.scatter(draft_length, z_fit, label="Fitted Values")
plt.xlabel("Draft Length")
plt.ylabel("Accept Length")
plt.title("Draft Length vs Accept Length")
plt.xscale("log")
plt.legend()
plt.savefig("draft_length_vs_accept_length.png")
