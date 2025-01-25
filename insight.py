import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set style biar tampilan plot lebih cakep
sns.set(style='whitegrid')

# Fungsi buat nambahin label di DALAM bar
def add_center_labels(ax, fmt='%.2f', fontcolor='white'):
    """
    ax: axes object dari matplotlib/seaborn
    fmt: format string untuk nilai label
    fontcolor: warna teks label
    """
    for container in ax.containers:
        ax.bar_label(
            container, 
            label_type='center',  # 'center' => label muncul di tengah bar
            fmt=fmt, 
            color=fontcolor,
            fontsize=9  # atur sesuai selera
        )

##############################################################################
# 1. Baca excel
##############################################################################
df = pd.read_excel('./logs-augmented/testing_accuracy_augmented.xlsx')

print("Kolom yang tersedia di DataFrame:")
print(df.columns)

##############################################################################
# 2. Waktu Training (average_training_time)
##############################################################################
# Kita bikin data agregat
train_time_by_model_bs_mp = (
    df.groupby(['model', 'batch_size', 'mixed_precision_mode'])['average_training_time']
      .mean()
      .reset_index()
)
print("\n=== Waktu Training (average_training_time) ===")
print(train_time_by_model_bs_mp)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=train_time_by_model_bs_mp,
    x='batch_size',
    y='average_training_time',
    hue='model',
    ci='sd',         # Tampilkan error bar (selisih standar deviasi)
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata Waktu Training per Epoch vs Batch Size (Per Model)")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 3. GPU VRAM usage (average_gpu_vram_usage)
##############################################################################
vram_by_model_bs_mp = (
    df.groupby(['model', 'batch_size', 'mixed_precision_mode'])['average_gpu_vram_usage']
      .mean()
      .reset_index()
)
print("\n=== GPU VRAM Usage (average_gpu_vram_usage) ===")
print(vram_by_model_bs_mp)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=vram_by_model_bs_mp,
    x='batch_size',
    y='average_gpu_vram_usage',
    hue='model',
    ci='sd',         
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata GPU VRAM Usage vs Batch Size (Per Model)")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 4. GPU Watt usage (average_gpu_watt_usage)
##############################################################################
watt_by_model_bs_mp = (
    df.groupby(['model', 'batch_size', 'mixed_precision_mode'])['average_gpu_watt_usage']
      .mean()
      .reset_index()
)
print("\n=== GPU Watt Usage (average_gpu_watt_usage) ===")
print(watt_by_model_bs_mp)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=watt_by_model_bs_mp,
    x='batch_size',
    y='average_gpu_watt_usage',
    hue='model',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata GPU Watt Usage vs Batch Size (Per Model)")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 5. Training accuracy (test_accuracy)
##############################################################################
acc_by_model_lr_drop = (
    df.groupby(['model', 'learning_rate', 'dropout_rate'])['test_accuracy']
      .mean()
      .reset_index()
)
print("\n=== Pengaruh faktor ke Test Accuracy (test_accuracy) ===")
print(acc_by_model_lr_drop)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=acc_by_model_lr_drop,
    x='dropout_rate',
    y='test_accuracy',
    hue='model',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata Test Accuracy vs Dropout (Per Model & Learning Rate)")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 6. Perbandingan antar model
##############################################################################
model_compare = (
    df.groupby('model')
      .agg({
          'test_accuracy': 'mean',
          'test_loss': 'mean',
          'average_gpu_vram_usage': 'mean',
          'average_gpu_watt_usage': 'mean',
          'average_training_time': 'mean'
      })
      .reset_index()
)
print("\n=== Perbandingan antar model ===")
print(model_compare)

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=model_compare, 
    x='model', 
    y='test_accuracy',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Comparison of Test Accuracy by Model")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 7. Perbandingan antar learning_rate
##############################################################################
lr_compare = (
    df.groupby('learning_rate')
      .agg({
          'test_accuracy': 'mean',
          'test_loss': 'mean',
          'average_training_time': 'mean'
      })
      .reset_index()
)
print("\n=== Perbandingan antar Learning Rate ===")
print(lr_compare)

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=lr_compare, 
    x='learning_rate', 
    y='test_accuracy',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata Test Accuracy vs Learning Rate")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 8. Perbandingan antar batch_size
##############################################################################
bs_compare = (
    df.groupby('batch_size')
      .agg({
          'test_accuracy': 'mean',
          'test_loss': 'mean',
          'average_training_time': 'mean',
          'average_gpu_vram_usage': 'mean'
      })
      .reset_index()
)
print("\n=== Perbandingan antar Batch Size ===")
print(bs_compare)

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=bs_compare, 
    x='batch_size', 
    y='test_accuracy',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata Test Accuracy vs Batch Size")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 9. Perbandingan antar mixed_precision_mode
##############################################################################
mp_compare = (
    df.groupby('mixed_precision_mode')
      .agg({
          'test_accuracy': 'mean',
          'test_loss': 'mean',
          'average_gpu_vram_usage': 'mean',
          'average_gpu_watt_usage': 'mean',
          'average_training_time': 'mean'
      })
      .reset_index()
)
print("\n=== Perbandingan antar Mixed Precision Mode ===")
print(mp_compare)

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=mp_compare, 
    x='mixed_precision_mode', 
    y='test_accuracy',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata Test Accuracy vs Mixed Precision Mode")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# 10. Perbandingan antar dropout_rate
##############################################################################
drop_compare = (
    df.groupby('dropout_rate')
      .agg({
          'test_accuracy': 'mean',
          'test_loss': 'mean',
          'average_training_time': 'mean'
      })
      .reset_index()
)
print("\n=== Perbandingan antar Dropout Rate ===")
print(drop_compare)

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=drop_compare, 
    x='dropout_rate', 
    y='test_accuracy',
    ci='sd',
    errwidth=0.7,
    capsize=0.2
)
plt.title("Rata-rata Test Accuracy vs Dropout Rate")
add_center_labels(ax, fmt='%.2f', fontcolor='white')
plt.show()

##############################################################################
# Selesai
##############################################################################
print("\nSelesai, bro! Cek output console dan plot.")
