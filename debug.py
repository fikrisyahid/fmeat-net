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

# import helper
# import pandas as pd

# helper.get_average_data_from_csv(new_column="average_gpu_watt_usage", calculated_key="gpu_watt_usage")
# helper.get_average_data_from_csv(new_column="average_gpu_vram_usage", calculated_key="gpu_vram_usage")
# helper.get_average_data_from_csv(new_column="average_training_time", calculated_key="training_time")


# df = pd.read_csv("logs/non-augmented/testing_accuracy.csv")
# # convert to xlsx
# df.to_excel("logs/non-augmented/testing_accuracy_non_augmented.xlsx", index=False)

# import helper

# helper.fix_csv_combination_sort("logs/non-augmented/testing_accuracy.csv", "logs/non-augmented/testing_accuracy_fixed.csv")

import helper

helper.get_correlation_matrix("./logs-augmented/testing_accuracy_augmented.xlsx")