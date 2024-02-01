import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置文件夹路径和帧率
folder_path = './example_data/posed_images/'  # 请替换为你的文件夹路径
frame_rate = 20  # 设置帧率（每秒显示的帧数）

# 获取文件夹中所有以.jpg为结尾的文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('0.jpg')]
image_files = sorted(image_files)  # 按文件名排序
# 创建Matplotlib的窗口
fig, ax = plt.subplots()

# 定义更新函数，用于显示下一帧图像
def update(frame):
    if frame < len(image_files):
        img_path = os.path.join(folder_path, image_files[frame])
        img = cv2.imread(img_path)
        ax.clear()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(image_files[frame])  # 显示文件名作为标题
        global current_frame
        current_frame = frame  # 更新当前帧数

        
# 使用FuncAnimation创建动画
ani = FuncAnimation(fig, update, frames=len(image_files), interval=1000 / frame_rate)

# 定义暂停/恢复函数
def on_space(event):
    global is_playing
    if event.key == ' ':
        if is_playing:
            ani.event_source.stop()
        else:
            ani.event_source.start()
        is_playing = not is_playing
    
# 定义左右箭头键事件处理器
def on_key(event):
    global current_frame
    if event.key == 'left':
        current_frame = max(current_frame - 1, 0)
    elif event.key == 'right':
        current_frame = min(current_frame + 1, len(image_files) - 1)
    update(current_frame)
    fig.canvas.draw()

# 绑定空格键和箭头键的事件处理器
fig.canvas.mpl_connect('key_press_event', on_space)
fig.canvas.mpl_connect('key_press_event', on_key)
global is_playing
is_playing = True  # 初始状态为播放

# 显示动画
plt.show()
