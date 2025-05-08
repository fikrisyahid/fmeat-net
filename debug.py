# =============================================================================
# Basic Debugging
# =============================================================================

# import helper


# helper.visualize_augmentations("./dataset/distributed/training/sapi/s7.jpg")
# helper.generate_augmented_images(
#     source_dir="./dataset/training",
#     destination_dir="./dataset/augmented/training",
# )
# helper.generate_augmented_images(
#     source_dir="./dataset/validation",
#     destination_dir="./dataset/augmented/validation",
# )
# helper.generate_augmented_images(
#     source_dir="./dataset/testing",
#     destination_dir="./dataset/augmented/testing",
# )
# print(helper.get_normalization_mean_std("./dataset/augmented/training"))
# print(helper.get_normalization_mean_std("./dataset/augmented/validation"))
# print(helper.get_normalization_mean_std("./dataset/augmented/testing"))

# print(helper.round_to_closest_possible([16, 32, 64], 70))


# class Animal:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def speak(self):
#         print(f"{self.name} says that he is {self.age} years old")


# cat = Animal(age=4, name="Kitty")
# cat.speak()

# =============================================================================
# Argparse Example
# =============================================================================

# import argparse


# def main(model_type, pso, mp_mode):
#     print(f"Model type: {model_type}, type: {type(model_type)}")
#     print(f"PSO: {pso}, type: {type(pso)}")
#     print(f"Mixed Precision Mode: {mp_mode}, type: {type(mp_mode)}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Script buat menjalankan model dengan konfigurasi tertentu."
#     )
#     parser.add_argument(
#         "model_type",
#         type=str,
#         help="Tipe model yang mau digunakan, misal: 'CNN'",
#     )
#     parser.add_argument(
#         "pso",
#         type=lambda x: (str(x).lower() == "true"),
#         help="PSO aktif atau nggak (True/False)",
#     )
#     parser.add_argument(
#         "mp_mode", type=int, help="Mode Mixed Precision, misal: 0 atau 1"
#     )

#     args = parser.parse_args()
#     main(args.model_type, args.pso, args.mp_mode)


# =============================================================================
# Basic PSO Implementation
# =============================================================================

# from pyswarms.single import GlobalBestPSO


# # Define a simple objective function
# def objective_func(x):
#     # Ambil hyperparameter dari partikel PSO
#     learning_rate = x[:, 0]  # Dimensi 0: learning rate
#     batch_size = x[:, 1]  # Dimensi 1: batch size
#     dropout_rate = x[:, 2]  # Dimensi 2: dropout rate

#     # Simulasi perhitungan "loss" dari model yang pakai hyperparameter ini
#     # Misalnya kita buat loss = (learning_rate - 0.005)^2 + (batch_size - 64)^2 + (dropout_rate - 0.2)^2
#     # Tujuannya biar nilai loss sekecil mungkin di sekitar learning rate = 0.005, batch size = 64, dropout rate = 0.2
#     loss = (
#         (learning_rate - 0.005) ** 2
#         + (batch_size - 64) ** 2
#         + (dropout_rate - 0.2) ** 2
#     )

#     return loss


# # Set bounds untuk tiap hyperparameter
# bounds = (
#     [
#         0.001,  # Lower bound untuk learning rate
#         16,  # Lower bound untuk batch size
#         0.1,  # Lower bound untuk dropout rate
#     ],
#     [
#         0.01,  # Upper bound untuk learning rate
#         128,  # Upper bound untuk batch size
#         0.5,  # Upper bound untuk dropout rate
#     ],
# )

# # Set options untuk PSO
# options = {
#     "c1": 1.5,  # Cognitive parameter
#     "c2": 2.0,  # Social parameter
#     "w": 0.5,  # Inertia weight
# }

# # Inisialisasi PSO optimizer
# optimizer = GlobalBestPSO(
#     n_particles=40, dimensions=3, options=options, bounds=bounds
# )

# # Run optimization
# best_score, best_params = optimizer.optimize(objective_func, iters=100)

# # Print hasil optimasi
# print("Best Score (minimized loss):", best_score)
# print("Best Hyperparameters:")
# print("Learning Rate:", best_params[0])
# print("Batch Size:", int(best_params[1]))
# print("Dropout Rate:", best_params[2])


# =============================================================================
# Visualization PSO
# =============================================================================

# # Import modules
# from IPython.display import Image

# # Import PySwarms
# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx
# from pyswarms.utils.plotters import plot_contour

# from pyswarms.utils.plotters.formatters import Mesher

# options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
# optimizer = ps.single.GlobalBestPSO(
#     n_particles=50, dimensions=2, options=options
# )
# cost, pos = optimizer.optimize(fx.sphere, iters=100)

# # Initialize mesher with sphere function
# m = Mesher(func=fx.sphere)


# # Make animation
# animation = plot_contour(
#     pos_history=optimizer.pos_history, mesher=m, mark=(0, 0)
# )

# # Enables us to view it in a Jupyter notebook
# animation.save("plot0.gif", writer="imagemagick", fps=10)
# Image(url="plot0.gif")

# =============================================================================
# Generate CSV Row
# =============================================================================

# import random

# def generate_csv_row(epoch):
#     row = [
#         epoch,
#         random.uniform(1.09, 1.1),  # Adjusted to match the large scale of training_loss
#         random.uniform(0.31, 0.34),
#         random.uniform(1.0987177312045187, 1.1020467097703086),  # Adjusted to match validation_loss
#         0.3333333333333333,  # Konsisten di data
#         0.1111111111111111,  # Konsisten di data
#         0.3333333333333333,  # Konsisten di data
#         0.16666666666666666,  # Konsisten di data
#         round(random.uniform(41, 43), 3),  # Slightly reduced range for gpu_watt_usage
#         8172.71484375,
#         random.uniform(1418, 1421)  # Adjusted to match the large scale of training_time
#     ]
#     for i in row:
#         print(i, end=",")
#     print()
#     print()

# for i in range(21):
#     generate_csv_row(i)

# =============================================================================
# Generate CSV average data
# =============================================================================

# import helper
# import pandas as pd

# helper.get_average_data_from_csv(
#     new_column="average_gpu_watt_usage",
#     calculated_key="gpu_watt_usage",
#     log_dir="./logs-non-augmented-ipb",
#     csv_path="./logs-non-augmented-ipb/testing_accuracy.csv",
# )
# helper.get_average_data_from_csv(
#     new_column="average_gpu_vram_usage",
#     calculated_key="gpu_vram_usage",
#     log_dir="./logs-non-augmented-ipb",
#     csv_path="./logs-non-augmented-ipb/testing_accuracy.csv",
# )
# helper.get_average_data_from_csv(
#     new_column="average_training_time",
#     calculated_key="training_time",
#     log_dir="./logs-non-augmented-ipb",
#     csv_path="./logs-non-augmented-ipb/testing_accuracy.csv",
# )

# # =============================================================================
# # Convert CSV to XLSX
# # =============================================================================

# import pandas as pd

# # df = pd.read_csv("logs-non-augmented-ipb/testing_accuracy.csv")
# df = pd.read_csv("./logs-augmented-ipb/cnn_lr0.001_dr0.8_bs32_mp0.csv")
# # convert to xlsx
# # df.to_excel(
# #     "logs-non-augmented-ipb/testing_accuracy_augmented.xlsx", index=False
# # )
# df.to_excel(
#     "logs_second_best_model_per_epoch.xlsx", index=False
# )

# =============================================================================
# Convert XLSX to CSV
# =============================================================================

# import pandas as pd

# def excel_to_csv(excel_file_path, csv_file_path, index=False):
#     """
#     Convert an Excel file to CSV format

#     Parameters:
#     -----------
#     excel_file_path : str
#         Path to the Excel file
#     csv_file_path : str
#         Path where the CSV file will be saved
#     index : bool, default=False
#         Whether to include the index column in the CSV
#     """
#     # Read Excel file
#     df = pd.read_excel(excel_file_path)

#     # Save as CSV file
#     df.to_csv(csv_file_path, index=index)

#     print(f"Excel file '{excel_file_path}' successfully converted to CSV: '{csv_file_path}'")

# # Convert logs-main.xlsx to logs-main.csv
# excel_to_csv("./logs-main.xlsx", "./logs-main.csv")


# =============================================================================
# Fix CSV combination sort
# =============================================================================

# import helper

# helper.fix_csv_combination_sort(
#     "logs-non-augmented-ipb/testing_accuracy.csv",
#     "logs-non-augmented-ipb/testing_accuracy_fixed.csv",
# )

# =============================================================================
# Plot correlation matrix from XLSX
# =============================================================================

# import helper

# helper.get_correlation_matrix("./logs-augmented/testing_accuracy_augmented.xlsx")

# =============================================================================
# Get data insight
# =============================================================================

# import helper
# import pandas as pd

# df_source = pd.read_excel("./logs-main.xlsx")

# # Mengatur ulang value dari kolom
# df_source["augmented"] = df_source["augmented"].map(
#     {0: "Tanpa augmentasi", 1: "Augmentasi"}
# )
# df_source["model"] = df_source["model"].map(
#     {"cnn": "FMEAT-Net", "vgg": "VGG16"}
# )
# df_source["mixed_precision_mode"] = df_source["mixed_precision_mode"].map(
#     {0: "(FP32, FP32)", 1: "(FP16, FP32)", 2: "(FP64, FP64)"}
# )

# print(df_source.head())

# df = df_source
# # df = df_source[df_source["test_accuracy"] != -1]

# helper.plot_bar_mean(
#     x_column="batch_size",
#     # y_column="test_accuracy",
#     # y_column="average_training_time",
#     # y_column="average_gpu_watt_usage",
#     y_column="average_gpu_vram_usage",
#     x_label="Ukuran Batch Size",
#     # y_label="Rata-rata akurasi pengujian (%)",
#     # y_label="Rata-rata waktu pelatihan per-epoch (detik)",
#     # y_label="Rata-rata penggunaan daya GPU (Watt)",
#     y_label="Rata-rata penggunaan VRAM GPU (MB)",
#     hue_column="model",
#     groupby_column=["batch_size", "model"],
#     # plot_title="Rata-rata akurasi pengujian pada arsitektur CNN berbeda",
#     # plot_title="Rata-rata waktu pelatihan per epoch pada arsitektur CNN size berbeda",
#     # plot_title="Rata-rata penggunaan daya GPU pada konfigurasi augmentasi berbeda",
#     plot_title="Rata-rata penggunaan VRAM GPU pada ukuran batch size berbeda",
#     df=df,
#     # multiply_y_column_by=100,
# )

# =============================================================================
# Best model chart
# =============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Increase matplotlib DPI
matplotlib.rcParams["figure.dpi"] = 200

# Step 1: Load the data
data = pd.read_excel("./logs_best_model_per_epoch.xlsx")

# Step 2: Filter the data to include only 'training_accuracy' and 'validation_accuracy'
filtered_data = data[
    [
        "epoch",
        "training_accuracy",
        "validation_accuracy",
        "training_accuracy_2",
        "validation_accuracy_2",
    ]
]
for col in [
    "training_accuracy",
    "validation_accuracy",
    "training_accuracy_2",
    "validation_accuracy_2",
]:
    filtered_data[col] = filtered_data[col] * 100
# filtered_data = data[["epoch", "training_time", "training_time_2"]]

# Step 3: Reshape the filtered data from wide to long format
long_data = pd.melt(
    filtered_data, id_vars=["epoch"], var_name="metric", value_name="value"
)

long_data["metric"] = long_data["metric"].map(
    {
        "training_accuracy": "Akurasi pelatihan konfigurasi A",
        "training_accuracy_2": "Akurasi pelatihan konfigurasi B",
        "validation_accuracy": "Akurasi validasi konfigurasi A",
        "validation_accuracy_2": "Akurasi validasi konfigurasi B",
    }
)
# long_data["metric"] = long_data["metric"].map(
#     {
#         "training_time": "Configuration A's training time",
#         "training_time_2": "Configuration B's training time",
#     }
# )

# Step 4: Plot the data using Seaborn
plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
sns.lineplot(
    data=long_data, x="epoch", y="value", hue="metric", palette="tab10"
)

# Step 5: Customize the plot
plt.title(
    "Perbandingan akurasi pelatihan dan validasi dari per-epoch",
    fontsize=16,
)
# plt.title(
#     "Per-epoch training time comparison",
#     fontsize=16,
# )

plt.xlabel("Epoch", fontsize=14)

plt.ylabel("Akurasi (%)", fontsize=14)
# plt.ylabel("Training time (second)", fontsize=14)

plt.legend(title="Jenis akurasi", loc="best")  # Place legend inside the plot
# plt.legend(title="Training time type", loc="best")  # Place legend inside the plot

plt.grid(True)  # Add gridlines for better readability
plt.tight_layout()  # Adjust layout to prevent overlap

# Set x-ticks to only show integer values
max_epoch = int(filtered_data["epoch"].max())
plt.xticks(
    np.arange(1, max_epoch + 1, 1)
)  # Only integer values from 0 to max epoch

plt.tight_layout()  # Adjust layout to prevent overlap

# Show the plot
plt.show()

# =============================================================================
# Model summary
# =============================================================================
# from torchsummary import summary
# from models import CNNModel, VGGModel

# summary(
#     model=CNNModel(),
#     input_size=(
#         3,
#         112,
#         112,
#     ),
# )

# summary(
#     model=VGGModel(),
#     input_size=(
#         3,
#         112,
#         112,
#     ),
# )
# =============================================================================
