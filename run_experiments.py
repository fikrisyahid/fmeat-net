import subprocess
import itertools
import os  # Import os module
import config  # Import config to access logging configurations

# Define the parameter ranges
model_types = ["cnn", "vgg"]
learning_rates = [0.01, 0.001, 0.0001]
dropout_rates = [0.2, 0.5, 0.8]
batch_sizes = [16, 32, 64]
mp_modes = [1, 2]  # Mixed precision modes

# Ensure the log directory exists
os.makedirs(config.LOG_FOLDER, exist_ok=True)

# Generate all combinations of parameters
combinations = list(
    itertools.product(
        model_types, learning_rates, dropout_rates, batch_sizes, mp_modes
    )
)

# Note: Combination is a list, it can be reversed by running combinations.reverse()
# in case you want to reverse the order of the combinations from the biggest possible value

print(f"Total combinations: {len(combinations)}")
print("-" * 80)
print(f"Combinations: {combinations}")
print("-" * 80)
print("Running experiments...")

iteration_log_file_path = (
    f"{config.LOG_FOLDER}/{config.ITERATION_LOG_FILE_NAME}"
)

for idx, (model_type, lr, dr, bs, mp_mode) in enumerate(combinations):
    # if (idx < 91):
    #     print(f"Skipping combination {idx + 1}/{len(combinations)}: Model={model_type}, LR={lr}, Dropout={dr}, Batch Size={bs}, MP Mode={mp_mode}")
    #     continue
    # if (model_type == "vgg" and mp_mode == 2 and (bs == 32 or bs == 64)):
    #     print(f"Skipping combination because not enough memory {idx + 1}/{len(combinations)}: Model={model_type}, LR={lr}, Dropout={dr}, Batch Size={bs}, MP Mode={mp_mode}")
    #     continue
    print(
        f"Running combination {idx + 1}/{len(combinations)}: Model={model_type}, LR={lr}, Dropout={dr}, Batch Size={bs}, MP Mode={mp_mode}"
    )
    command = [
        "python",
        "main.py",
        model_type,
        str(lr),
        str(dr),
        str(bs),
        str(mp_mode),
    ]

    # Run the command and handle exceptions
    try:
        subprocess.run(command, check=True)
        print(f"Combination {idx + 1} completed successfully.")
    except subprocess.CalledProcessError as e:
        error_message = f"Combination {idx + 1} failed with return code {e.returncode}. Error: {e}\n"
        print(error_message)
        # Construct the full path using f-string
        with open(iteration_log_file_path, "a") as log_file:
            log_file.write(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred while running combination {idx + 1}: {e}\n"
        print(error_message)
        with open(iteration_log_file_path, "a") as log_file:
            log_file.write(error_message)
    print("-" * 80)
