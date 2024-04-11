import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

folder_path = './example_data/posed_images/'  
frame_rate = 20  

image_files = [f for f in os.listdir(folder_path) if f.endswith('0.jpg')]
image_files = sorted(image_files)  
fig, ax = plt.subplots()

def update(frame):
    if frame < len(image_files):
        img_path = os.path.join(folder_path, image_files[frame])
        img = cv2.imread(img_path)
        ax.clear()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(image_files[frame])  
        global current_frame
        current_frame = frame  

        
ani = FuncAnimation(fig, update, frames=len(image_files), interval=1000 / frame_rate)

def on_space(event):
    global is_playing
    if event.key == ' ':
        if is_playing:
            ani.event_source.stop()
        else:
            ani.event_source.start()
        is_playing = not is_playing
    
def on_key(event):
    global current_frame
    if event.key == 'left':
        current_frame = max(current_frame - 1, 0)
    elif event.key == 'right':
        current_frame = min(current_frame + 1, len(image_files) - 1)
    update(current_frame)
    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_space)
fig.canvas.mpl_connect('key_press_event', on_key)
global is_playing
is_playing = True 

plt.show()
