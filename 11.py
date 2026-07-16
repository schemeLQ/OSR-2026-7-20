import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 纵轴一行显示，横轴两行显示
y_labels = [
    '0 正常',
    '1 制冷剂泄漏',
    '2 冷凝器结垢',
    '3 冷却水回水温度异常',
    '4 传感器故障'
]

x_labels = [
    '0\n正常',
    '1\n制冷剂泄漏',
    '2\n冷凝器结垢',
    '3\n冷却水回水温度异常',
    '4\n传感器故障'
]

# 你之前的验证集混淆矩阵（样本数）
val_cm = np.array([
    [15821,    0,    0,    0,   62],
    [    0, 8072,    0,    0,    0],
    [    0,    0, 6329,    2,    0],
    [    0,    0,    0, 1368,    0],
    [  800,    0,    0,    0, 15083]
])

fig, ax = plt.subplots(figsize=(9, 7.5), dpi=160)
im = ax.imshow(val_cm, cmap='Blues')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('样本数', rotation=270, labelpad=15, fontsize=11)

ax.set_title('验证集混淆矩阵', fontsize=18, fontweight='bold', pad=12)
ax.set_xlabel('预测类别', fontsize=13)
ax.set_ylabel('真实类别', fontsize=13)

ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels, fontsize=10, rotation=0, ha='center')
ax.set_yticklabels(y_labels, fontsize=10)

# 网格线
ax.set_xticks(np.arange(-.5, len(x_labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(y_labels), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
ax.tick_params(which='minor', bottom=False, left=False)

# 标数字
max_val = val_cm.max()
for i in range(val_cm.shape[0]):
    for j in range(val_cm.shape[1]):
        color = 'white' if val_cm[i, j] > max_val * 0.45 else '#08306b'
        ax.text(j, i, str(val_cm[i, j]),
                ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('验证集混淆矩阵.png', bbox_inches='tight')
plt.show()