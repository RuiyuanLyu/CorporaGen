import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def adjust_rgb(hex, r=1.0, g=1.0, b=1.0):
    rgb = mcolors.hex2color(hex)
    return mcolors.to_hex((r * rgb[0], g * rgb[1], b * rgb[2]))

# 0.5, 0.75, 1.0 are not ready, they are made up now.
training_data = [0, 1/3, 2/3, 1.0]  
em = [8.31, 27.80, 30.23, 29.01]
bleu4 = [2.15, 8.54, 8.48, 8.63]
meteor = [4.44, 13.90, 14.76, 14.93]
rougel = [16.39, 41.33, 44.29, 42.62]
gpt4 = [15.84, 38.46, 42.54, 44.81]
sns.set(style='ticks')

fig, ax1 = plt.subplots(figsize=(6, 4))

sns.lineplot(x=training_data, y=em, ax=ax1, label='EM@1', color='red', marker='s', linestyle='-', lw=3, markersize=8)
sns.lineplot(x=training_data, y=bleu4, ax=ax1, label='BLEU-4', color='blue', marker='o', linestyle='--', lw=3, markersize=8)
sns.lineplot(x=training_data, y=meteor, ax=ax1, label='METEOR', color='green', marker='^', linestyle='-.', lw=3, markersize=8)
sns.lineplot(x=training_data, y=rougel, ax=ax1, label='ROUGE-L', color='purple', marker='d', linestyle=':', lw=3, markersize=8)
sns.lineplot(x=training_data, y=gpt4, ax=ax1, label='GPT-4', color='orange', marker='p', linestyle='-.', lw=3, markersize=8)
ax1.set_xlabel('Training Data Quantity', fontsize=14)
ax1.set_ylabel('Metric', fontsize=14)
ax1.set_xlim(-0.05, 1.05)

# plt.title('Performance Over Training Data Quantity', fontsize=16)
plt.savefig('scaling_law_ll3da.png', dpi=300)
plt.show()
