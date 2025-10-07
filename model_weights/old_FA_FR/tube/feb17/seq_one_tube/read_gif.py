import numpy as np

# read gif 
import imageio
import os

FAFR_gif = "FAFRsplit10.gif"

# read gif
gif = imageio.get_reader(FAFR_gif)

# get the number of frames
num_frames = gif.get_length()

print("Number of frames: ", num_frames)

# make directory to store frames
if not os.path.exists("frames_3agents"):
    os.makedirs("frames_3agents")

# save each frame as an image
for i in range(num_frames):
    frame = gif.get_data(i)
    imageio.imwrite("frames_3agents/frame_{}.png".format(i), frame)

# select m frames to for figure
m = 4
frames = np.linspace(15, num_frames-5, m, dtype=int)

frame_size = gif.get_data(0).shape

# make a figure with m frames side by side
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, m, figsize=(20, 5))

# minimize white space between subplots

for i in range(m):
    ax[i].imshow(plt.imread("frames_3agents/frame_{}.png".format(frames[i])))
    ax[i].axis('off')
    
    # add border to each frame
    rect = patches.Rectangle((0,0), frame_size[1], frame_size[0], linewidth=2, edgecolor='k', facecolor='none')
    ax[i].add_patch(rect)


    # ax[i].set_title("Frame {}".format(frames[i]))
plt.subplots_adjust(wspace=-0.05, hspace=-0.08)

plt.tight_layout()
plt.savefig("frames_3agents.pdf")
plt.show()
