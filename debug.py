# =============================================================================
# Basic Debugging
# =============================================================================

# import helper

# helper.visualize_augmentations("./dataset/training/sapi/s7.jpg")
# helper.generate_augmented_images(source_dir="./dataset/training", destination_dir="./dataset/augmented/training")
# helper.generate_augmented_images(source_dir="./dataset/validation", destination_dir="./dataset/augmented/validation")
# helper.generate_augmented_images(source_dir="./dataset/testing", destination_dir="./dataset/augmented/testing")
# print(helper.get_normalization_mean_std("./dataset/augmented/training"))
# print(helper.get_normalization_mean_std("./dataset/augmented/validation"))
# print(helper.get_normalization_mean_std("./dataset/augmented/testing"))

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
#         "model_type", type=str, help="Tipe model yang mau digunakan, misal: 'CNN'"
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

from pyswarms.single import GlobalBestPSO


# Define a simple objective function
def objective_func(x):
  # Ambil hyperparameter dari partikel PSO
  learning_rate = x[:, 0]  # Dimensi 0: learning rate
  batch_size = x[:, 1]  # Dimensi 1: batch size
  dropout_rate = x[:, 2]  # Dimensi 2: dropout rate

  # Simulasi perhitungan "loss" dari model yang pakai hyperparameter ini
  # Misalnya kita buat loss = (learning_rate - 0.005)^2 + (batch_size - 64)^2 + (dropout_rate - 0.2)^2
  # Tujuannya biar nilai loss sekecil mungkin di sekitar learning rate = 0.005, batch size = 64, dropout rate = 0.2
  loss = (learning_rate - 0.005) ** 2 + (batch_size - 64) ** 2 + (dropout_rate - 0.2) ** 2

  return loss


# Set bounds untuk tiap hyperparameter
bounds = (
  [
    0.001,  # Lower bound untuk learning rate
    16,  # Lower bound untuk batch size
    0.1,  # Lower bound untuk dropout rate
  ],
  [
    0.01,  # Upper bound untuk learning rate
    128,  # Upper bound untuk batch size
    0.5,  # Upper bound untuk dropout rate
  ],
)

# Set options untuk PSO
options = {
  "c1": 1.5,  # Cognitive parameter
  "c2": 2.0,  # Social parameter
  "w": 0.5,  # Inertia weight
}

# Inisialisasi PSO optimizer
optimizer = GlobalBestPSO(n_particles=40, dimensions=3, options=options, bounds=bounds)

# Run optimization
best_score, best_params = optimizer.optimize(objective_func, iters=100)

# Print hasil optimasi
print("Best Score (minimized loss):", best_score)
print("Best Hyperparameters:")
print("Learning Rate:", best_params[0])
print("Batch Size:", int(best_params[1]))
print("Dropout Rate:", best_params[2])


# =============================================================================
# Visualization
# =============================================================================

# # Import modules
# from IPython.display import Image

# # Import PySwarms
# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx
# from pyswarms.utils.plotters import plot_contour

# from pyswarms.utils.plotters.formatters import Mesher

# options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
# optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)
# cost, pos = optimizer.optimize(fx.sphere, iters=100)

# # Initialize mesher with sphere function
# m = Mesher(func=fx.sphere)


# # Make animation
# animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))

# # Enables us to view it in a Jupyter notebook
# animation.save("plot0.gif", writer="imagemagick", fps=10)
# Image(url="plot0.gif")
