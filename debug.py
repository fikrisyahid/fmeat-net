# =============================================================================
# Basic Debugging
# =============================================================================

# import helper


# helper.visualize_augmentations("./dataset/training/sapi/s7.jpg")
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

# df = pd.read_csv("logs-non-augmented-ipb/testing_accuracy.csv")
# # convert to xlsx
# df.to_excel(
#     "logs-non-augmented-ipb/testing_accuracy_augmented.xlsx", index=False
# )


# =============================================================================
# Fix CSV combination sort
# =============================================================================

# import helper

# helper.fix_csv_combination_sort("logs/non-augmented/testing_accuracy.csv", "logs/non-augmented/testing_accuracy_fixed.csv")

# =============================================================================
# Plot correlation matrix from XLSX
# =============================================================================

# import helper

# helper.get_correlation_matrix("./logs-augmented/testing_accuracy_augmented.xlsx")

# =============================================================================
# Get data insight
# =============================================================================

import helper
import pandas as pd

df_source = pd.read_excel("./logs-main.xlsx")

# Filter data frame to only include rows that has environment column as lokal
df = df_source

# Apa saja yang mempengaruhi waktu training

# Model vs waktu training
helper.plot_bar_mean(
    x_column="model",
    y_column="average_training_time",
    x_label="Jenis Model",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["model"],
    plot_title="Rata-rata Waktu Pelatihan Model per Epoch",
    df=df,
)

# Learning rate vs waktu training
helper.plot_bar_mean(
    x_column="learning_rate",
    y_column="average_training_time",
    x_label="Learning Rate",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["model", "learning_rate"],
    plot_title="Rata-rata Waktu Pelatihan per Epoch vs Learning Rate (Per Model)",
    df=df,
)

# Dropout rate vs waktu training
helper.plot_bar_mean(
    x_column="dropout_rate",
    y_column="average_training_time",
    x_label="Persentase Dropout (%)",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["model", "dropout_rate"],
    plot_title="Rata-rata Waktu Pelatihan per Epoch vs Persentase Dropout (Per Model)",
    df=df,
)

# Batch size vs waktu training
helper.plot_bar_mean(
    x_column="batch_size",
    y_column="average_training_time",
    x_label="Batch Size",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["model", "batch_size"],
    plot_title="Rata-rata Waktu Pelatihan per Epoch vs Batch Size (Per Model)",
    df=df,
)

# Mode mixed precision vs waktu training
helper.plot_bar_mean(
    x_column="mixed_precision_mode",
    y_column="average_training_time",
    x_label="Mode Mixed Precision",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["model", "mixed_precision_mode"],
    plot_title="Rata-rata Waktu Pelatihan per Epoch vs Mode Mixed Precision (Per Model)",
    df=df,
)

df = df_source

# Perbandingan lokal dan IPB dalam waktu training
helper.plot_bar_mean(
    x_column="environment",
    y_column="average_training_time",
    x_label="Environment Pelatihan",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["environment", "model"],
    plot_title="Rata-rata Waktu Pelatihan per Epoch vs environment pelatihan",
    df=df,
)

# Perbandingan lokal dan IPB dalam watt usage
helper.plot_bar_mean(
    x_column="environment",
    y_column="average_gpu_watt_usage",
    x_label="Environment Pelatihan",
    y_label="Rata-rata penggunaan Watt GPU (Watt)",
    hue_column="model",
    groupby_column=["environment", "model"],
    plot_title="Rata-rata penggunaan Watt GPU vs environment pelatihan",
    df=df,
)

# Perbandingan lokal dan IPB dalam VRAM usage
helper.plot_bar_mean(
    x_column="environment",
    y_column="average_gpu_vram_usage",
    x_label="Environment Pelatihan",
    y_label="Rata-rata penggunaan VRAM GPU (MB)",
    hue_column="model",
    groupby_column=["environment", "model"],
    plot_title="Rata-rata penggunaan VRAM GPU vs environment pelatihan",
    df=df,
)

df = df_source[df_source["test_accuracy"] != -1]

# Perbandingan lokal dan IPB dalam akurasi testing
helper.plot_bar_mean(
    x_column="environment",
    y_column="test_accuracy",
    x_label="Environment Pelatihan",
    y_label="Akurasi Testing (%)",
    hue_column="model",
    groupby_column=["environment", "model"],
    plot_title="Rata-rata akurasi testing vs environment pelatihan",
    df=df,
)

df = df_source

# Perbandingan augmentasi dan non-augmentasi dalam waktu training
helper.plot_bar_mean(
    x_column="augmented",
    y_column="average_training_time",
    x_label="Konfigurasi Augmentasi",
    y_label="Rata-rata Waktu Pelatihan Model (s)",
    hue_column="model",
    groupby_column=["augmented", "model"],
    plot_title="Rata-rata waktu pelatihan model vs konfigurasi augmentasi",
    df=df,
)

df = df_source[df_source["test_accuracy"] != -1]

# Perbandingan augmentasi dan non-augmentasi dalam akurasi testing
helper.plot_bar_mean(
    x_column="augmented",
    y_column="test_accuracy",
    x_label="Konfigurasi Augmentasi",
    y_label="Rata-rata akurasi testing (%)",
    hue_column="model",
    groupby_column=["augmented", "model"],
    plot_title="Rata-rata akurasi testing model vs konfigurasi augmentasi",
    df=df,
)

# Perbandingan mixed precision dalam akurasi testing
helper.plot_bar_mean(
    x_column="mixed_precision_mode",
    y_column="test_accuracy",
    x_label="Mode Mixed Precision",
    y_label="Rata-rata akurasi testing (%)",
    hue_column="model",
    groupby_column=["mixed_precision_mode", "model"],
    plot_title="Rata-rata akurasi testing model vs mode mixed precision",
    df=df,
)

helper.get_correlation_matrix("./logs-main.xlsx")