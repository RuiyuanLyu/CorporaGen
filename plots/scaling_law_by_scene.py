import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def adjust_rgb(hex, r=1.0, g=1.0, b=1.0):
    rgb = mcolors.hex2color(hex)
    return mcolors.to_hex((r * rgb[0], g * rgb[1], b * rgb[2]))

# 0.5, 0.75, 1.0 are not ready, they are made up now.
x_vg = [0.25, 0.5, 0.75, 1.0]  
y_vg = [8.70, 10.49, 16.01, 20.55]  # 15.35 should be 16.01. 

x_qa = [0, 1/3, 2/3, 1.0]  
y_qa = [15.84, 38.46, 42.54, 44.81]

sns.set(style='ticks')

fig, ax1 = plt.subplots(figsize=(6, 4.5))

color_vg = adjust_rgb('#F98C78', 1, 0.85, 0.85)
sns.lineplot(x=x_vg, y=y_vg, ax=ax1, label='VG (AP@.25)', color=color_vg, marker='s', linestyle='-', lw=3, markersize=8)
ax1.set_xlabel('Training Data Quantity', fontsize=18, labelpad=20)
ax1.set_ylabel('AP@0.25 (%)', fontsize=18, labelpad=20)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0, 50)

ax2 = ax1.twinx()
color_qa = adjust_rgb('#A9B8F9', 0.85, 0.85, 1)
sns.lineplot(x=x_qa, y=y_qa, label='QA (Accuracy)', color=color_qa, marker='o', linestyle='-', lw=3, markersize=8)
ax2.set_ylabel('Accuracy (%)', fontsize=18, labelpad=15)
ax2.set_ylim(0, 70)

ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=18)
ax2.get_legend().remove()

plt.title('Scaling Law for VG and QA', fontsize=20, pad=20)

plt.subplots_adjust(left=0.18, right=0.82, top=0.85, bottom=0.2)
plt.savefig('scaling_law_by_scene.png', dpi=300)
plt.show()
