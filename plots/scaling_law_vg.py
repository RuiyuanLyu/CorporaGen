import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def adjust_rgb(hex, r=1.0, g=1.0, b=1.0):
    rgb = mcolors.hex2color(hex)
    return mcolors.to_hex((r * rgb[0], g * rgb[1], b * rgb[2]))

# 0.5, 0.75, 1.0 are not ready, they are made up now.
training_data = [0.1, 0.2, 0.5, 0.75, 1.0]  
ap25 = [8.70, 10.49, 12, 12.58, 16]  
rec25 = [39.24, 47.21, 55, 44.56, 73]  
ap50 = [2.51, 2.94, 3, 4.05, 5]    
rec50 = [17.18, 21.76, 25, 22.09, 45]     

sns.set(style='ticks')

fig, ax1 = plt.subplots(figsize=(6, 4))

ap25_color = adjust_rgb('#FFC3A7', g=0.85, b=0.85)
ap50_color = adjust_rgb('#FFD2A8', g=0.85, b=0.85)
ap25_color = adjust_rgb('#F98C78', g=0.85, b=0.85)
sns.lineplot(x=training_data, y=ap25, ax=ax1, label='AP@.25', color=ap25_color, marker='s', linestyle='-', lw=3, markersize=8)
sns.lineplot(x=training_data, y=ap50, ax=ax1, label='AP@.50', color=ap50_color, marker='s', linestyle='--', lw=3, markersize=8)
ax1.set_xlabel('Training Data Quantity', fontsize=14)
ax1.set_ylabel('Average Precision', fontsize=14)
ax1.set_xlim(0, 1.05)
ax1.set_ylim(0, 20)

ax2 = ax1.twinx()
rec25_color = adjust_rgb('#90BEF0', r=0.85, g=0.85)
rec50_color = adjust_rgb('#B6E8FF', r=0.85, g=0.85)
rec25_color = adjust_rgb('#A9B8F9', r=0.85, g=0.85)
sns.lineplot(x=training_data, y=rec25, ax=ax2, label='Recall@.25', color=rec25_color, marker='o', linestyle='-', lw=3, markersize=8)
sns.lineplot(x=training_data, y=rec50, ax=ax2, label='Recall@.50', color=rec50_color, marker='o', linestyle='--', lw=3, markersize=8)
ax2.set_ylabel('Recall', fontsize=14)
ax2.set_ylim(0, 100)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=12)
# ax1.legend([handles[0]] + [handles2[0]] + [handles2[1]] + [handles[1]],
#             [labels[0]] + [labels2[0]]  + [labels2[1]]  + [labels[1]],
#             loc="upper left")
ax2.get_legend().remove()

# plt.title('Performance Over Training Data Quantity', fontsize=16)
plt.savefig('scaling_law_vg.png', dpi=300, bbox_inches='tight')
plt.show()
