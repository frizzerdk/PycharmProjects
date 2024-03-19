import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import animation
import noise
from PIL import Image
# Create a vectorized version of the pnoise3 function
vec_pnoise3 = np.vectorize(noise.pnoise3)

# Create a figure and axes
fig, ax = plt.subplots()

# Initialize the noise array
noise_values = np.zeros((100, 100))

# Create the x and y coordinate arrays using the NumPy meshgrid function
x, y = np.meshgrid(range(100), range(100))

# Create the hue, saturation, and value arrays
hue = np.zeros((100, 100))
saturation = np.zeros((100, 100))
value = np.zeros((100, 100))

# Define the minimum and maximum hue values for the gradient
hue_min = 0.0
hue_max = 0.16
gradient_ratio = 0.7

# Function to update the plot at each frame
def update(frame):
    global noise_values

    # Generate new noise values using the x and y coordinate arrays
    noise_values = vec_pnoise3(x / 30, y / 30 + frame * 2, frame)

    # Adjust the range of the noise values to the range 0-1
    noise_values = (noise_values - np.min(noise_values)) / (np.max(noise_values) - np.min(noise_values))

    # Use the y coordinate of each pixel as the hue value
    hue[:, :] = y / 100

    # Add the gradient hue values to the noise hue values
    hue[:, :] = (hue[:, :] * gradient_ratio) + (noise_values * (1 - gradient_ratio))
    # Scale and shift the hue values using the hue_min and hue_max variables
    hue[:, :] = hue[:, :] * (hue_max - hue_min) + hue_min

    # Make the saturation and value relative to the hue value
    saturation_ratio = 0.1
    saturation[:, :] = saturation_ratio + (hue[:, :] / hue_max) * (1 - saturation_ratio)
    saturation[:, :] = 1 - saturation[:, :]
    value_ratio = 0.3
    value[:, :] = value_ratio + hue[:, :] / hue_max * (1 - value_ratio)

    # Adjust the saturation by multiplying by a constant value
    #saturation[:, :] = np.ones_like(hue[:, :]) * 1

    # Adjust the value by multiplying by a constant value
    #value[:, :] = np.ones_like(hue[:, :]) * 1

    # Convert the hue, saturation, and value arrays to RGB colors
    rgb = hsv_to_rgb(np.dstack((hue, saturation, value)))

    # Set the image data and redraw the plot
    ax.imshow(rgb)
    plt.draw()


# Create the animation using the update function
anim = animation.FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=50)
anim.save('basic_animation.mp4',  extra_args=['-vcodec', 'libx264'])
plt.show()
