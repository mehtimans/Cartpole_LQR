import numpy as np
from scipy.signal import savgol_coeffs
from collections import deque
import matplotlib.pyplot as plt

# Your encoder sampling rate
sampling_rate = 200 # Hz
dt = 1.0 / sampling_rate

# Low-delay settings (≤6 ms delay)
window_length = 11  # Odd, minimal delay (~6 ms)

# Savitzky-Golay coefficients for angular velocity (1st derivative)
coeffs = savgol_coeffs(window_length, polyorder=3 , deriv=1, delta=dt, use='conv')
print("Savitzky-Golay coefficients", coeffs)
print("used Savitzky-Golay coefficients", coeffs[::-1])


# Simulate encoder-like pole motion (swing-like)
t = np.arange(0, 5, dt)
true_angle = 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 2 * t)
noise = np.random.normal(0, 0.02, size=true_angle.shape)
noisy_angle = true_angle + noise

# Buffers
angle_buffer = deque(maxlen=window_length)
velocity_estimates = []
smoothed_angles = []

# Simulate real-time processing
for i in range(len(t)):
    angle_buffer.append(noisy_angle[i])

    if len(angle_buffer) == window_length:
        velocity = np.dot(coeffs[::-1], list(angle_buffer))  # Reverse coeffs for causal
        velocity_estimates.append(velocity)
        smoothed_angles.append(np.mean(angle_buffer))
    else:
        velocity_estimates.append(0.0)  # Initial dummy value
        smoothed_angles.append(noisy_angle[i])

# Compare with true angular velocity
true_velocity = (
    0.5 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t) +
    0.2 * 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * t)
)

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, noisy_angle, label='Noisy Encoder Angle', alpha=0.5)
plt.plot(t, smoothed_angles, label='Smoothed Angle (Moving Mean)', linewidth=2)
plt.title("Pole Angle (radians)")
plt.xlabel("Time (s)")
plt.ylabel("Angle")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, velocity_estimates, label='Estimated Angular Velocity (Low-Delay SG)', linewidth=2)
plt.plot(t, true_velocity, label='True Angular Velocity', linestyle='--')
plt.title("Pole Angular Velocity (radians/s)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
