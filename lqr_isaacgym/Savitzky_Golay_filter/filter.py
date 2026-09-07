import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Simulate encoder readings for the cart-pole pole angle (radians)
sampling_rate = 500  # Hz (adjust this to match your encoder's sampling rate)
dt = 1.0 / sampling_rate
t = np.arange(0, 5, dt)  # Simulate 5 seconds of motion

# Example: swing-up-like motion (non-sinusoidal)
pole_angle = 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 2 * t)
# You could replace this with your actual encoder angle measurements

# Optional: simulate sensor noise
noise = np.random.normal(0, 0.02, size=pole_angle.shape)
noisy_angle = pole_angle + noise

# Savitzky-Golay filter for velocity estimation
window_length = 51  # Must be odd, adjust based on motion speed and sampling rate
polyorder = 3

# Smoothed angle (for visualization)
smoothed_angle = savgol_filter(noisy_angle, window_length, polyorder)

# Estimate angular velocity
angular_velocity = savgol_filter(noisy_angle, window_length, polyorder, deriv=1, delta=dt)

# True angular velocity (for comparison, since we know simulated motion)
true_velocity = (
    0.5 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t) +
    0.2 * 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * t)
)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, noisy_angle, label='Noisy Encoder Angle', alpha=0.5)
plt.plot(t, smoothed_angle, label='Smoothed Angle', linewidth=2)
plt.title("Pole Angle (radians)")
plt.xlabel("Time (s)")
plt.ylabel("Angle")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, angular_velocity, label='Estimated Angular Velocity', linewidth=2)
plt.plot(t, true_velocity, label='True Angular Velocity (for comparison)', linestyle='--')
plt.title("Pole Angular Velocity (radians/s)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
