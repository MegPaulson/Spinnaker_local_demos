
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# Generate sample data
num_points = 50
index_values = np.linspace(0, 4*np.pi, num_points)
y1_values = np.sin(index_values)
y2_values = np.cos(index_values)

# Create a DataFrame
data = pd.DataFrame({'Y1': y1_values, 'Y2': y2_values})

# Save data to CSV file
data.to_csv('sample_data.csv', index=False)

# wait for the file to be created
import time
time.sleep(1)

# Load CSV data
data = pd.read_csv('sample_data.csv')

# # # Create a figure and axis for plotting
# fig, ax = plt.subplots()
# line1, = ax.plot([], [], label='Series 1')
# line2, = ax.plot([], [], label='Series 2')
# scatter1 = ax.scatter([], [])
# scatter2 = ax.scatter([], [])

# # # Set labels and title
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_title('Time-lapse of Superimposed Plots')
# ax.legend()

# # set the x and y axis limits based on the data
# ax.set_xlim(0, 4*np.pi)
# ax.set_ylim(-1, 1)

# # use matplotlib.animation.FuncAnimation to create an animation
# def update(i):
#     line1.set_data(index_values[:i], y1_values[:i])
#     line2.set_data(index_values[:i], y2_values[:i])
#     scatter1.set_offsets(np.c_[index_values[:i], y1_values[:i]])
#     scatter2.set_offsets(np.c_[index_values[:i], y2_values[:i]])
#     return line1, line2, scatter1, scatter2

# ani = FuncAnimation(fig, update, frames=num_points, blit=True)

# # save the animation to a file
# ani.save('c:\\Acquisition_Data\\animation.mp4', writer='ffmpeg', fps=10)

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

# animate the generated data

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('c:\\Acquisition_Data\\animation.mp4', fourcc, 10.0, (640, 480))

for i in range(num_points):
    # Clear the current figure
    plt.clf()

    # Set the figure size to match the video resolution
    plt.gcf().set_size_inches(640 / plt.gcf().dpi, 480 / plt.gcf().dpi)

    #set the x and y axis limits based on the data
    plt.xlim(0, 4*np.pi)
    plt.ylim(-1, 1)

    # Plot the data up to the current point
    plt.plot(index_values[:i], y1_values[:i])
    plt.plot(index_values[:i], y2_values[:i])
    plt.scatter(index_values[:i], y1_values[:i])
    plt.scatter(index_values[:i], y2_values[:i])

    # Draw the figure and retrieve the pixel buffer
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))

    # Remove the alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Write the frame to the video file
    out.write(img)

# Release the VideoWriter
out.release()

# animate the images that correspond to the generated data
image_folder = r"c:\\Acquisition_Data\\vidsample"
video_name = image_folder + 'video.mp4'

# sort the numbered images in the folder
sortedimages = sorted(os.listdir(image_folder), key=lambda x: int(x.split('.')[0]))
images = [img for img in ((sortedimages)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Create a white background image with the desired resolution
background = np.ones((480, 640, 3), dtype=np.uint8) * 255

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(video_name, fourcc, 10, (640, 480))

for image in images:
    print(image)
    img = cv2.imread(os.path.join(image_folder, image))
    
    # Place the image in the center of the background image
    x_offset = (background.shape[1] - img.shape[1]) // 2
    y_offset = (background.shape[0] - img.shape[0]) // 2
    background[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    video.write(background)

cv2.destroyAllWindows()
# Release the VideoWriter
video.release()

# Combine the generated data video and the images video into one final video
data_video = cv2.VideoCapture('c:\\Acquisition_Data\\animation.mp4')
images_video = cv2.VideoCapture(video_name)

# Get the video properties
data_fps = data_video.get(cv2.CAP_PROP_FPS)
data_width = int(data_video.get(cv2.CAP_PROP_FRAME_WIDTH))
data_height = int(data_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
images_fps = images_video.get(cv2.CAP_PROP_FPS)
images_width = int(images_video.get(cv2.CAP_PROP_FRAME_WIDTH))
images_height = int(images_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object for the final video
final_video_name = 'c:\\Acquisition_Data\\final_video.mp4'
final_video = cv2.VideoWriter(final_video_name, fourcc, data_fps, (data_width, data_height + images_height))

# Read frames from both videos and write them to the final video
while True:
    # Read a frame from the data video
    ret_data, frame_data = data_video.read()
    if not ret_data:
        break

    # Read a frame from the images video
    ret_images, frame_images = images_video.read()
    if not ret_images:
        break

    # Combine the frames vertically
    combined_frame = np.vstack((frame_images, frame_data))

    # Write the combined frame to the final video
    final_video.write(combined_frame)

# Release the videos
data_video.release()
images_video.release()
final_video.release()


# Print the path of the final video
print(f"Final video saved at: {final_video_name}")



                
