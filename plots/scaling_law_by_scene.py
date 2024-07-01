import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def adjust_rgb(hex, r=1.0, g=1.0, b=1.0):
    rgb = mcolors.hex2color(hex)
    return mcolors.to_hex((r * rgb[0], g * rgb[1], b * rgb[2]))

# 0.5, 0.75, 1.0 are not ready, they are made up now.
x_1 = [0.1, 0.2, 0.5, 1.0]  
y_1 = [8.70, 10.49, 16.01, 20.55]  # 15.35 should be 16.01. 

x_2 = [0.25, 0.5, 0.75, 1.0]  
y_2 = [9.59, 15.25, 17.82, 20.55]

sns.set(style='ticks')

fig, ax1 = plt.subplots(figsize=(6, 4.5))

color_1 = adjust_rgb('#F98C78', 1, 0.85, 0.85)
color_2 = adjust_rgb('#A9B8F9', 0.85, 0.85, 1)

sns.lineplot(x=x_1, y=y_1, ax=ax1, label='sample uniformly (AP@.25)', color=color_1, marker='s', linestyle='-', lw=3, markersize=8)
sns.lineplot(x=x_2, y=y_2, ax=ax1, label='sample by scene (AP@.25)', color=color_2, marker='o', linestyle='--', lw=3, markersize=8)
ax1.set_xlabel('Training Data Quantity', fontsize=18, labelpad=20)
ax1.set_ylabel('AP@0.25 (%)', fontsize=18, labelpad=20)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0, 30)

ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.tick_params(axis='both', which='minor', labelsize=16)
plt.title('Scaling Law compared with Sample Strategy', fontsize=20, pad=20)

plt.subplots_adjust(left=0.18, right=0.82, top=0.85, bottom=0.2)
plt.savefig('scaling_law_by_scene.png', dpi=300)
plt.show()
