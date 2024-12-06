import argparse
from train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script untuk menjalankan model dengan konfigurasi tertentu.")
    parser.add_argument(
        "model_type",
        type=lambda x: (str(x).lower()),
        help="Tipe model yang akan digunakan: CNN atau VGG",
    )
    parser.add_argument(
        "pso",
        type=lambda x: (str(x).lower() == "true"),
        help="PSO aktif atau tidak: True atau False",
    )
    parser.add_argument("mp_mode", type=int, help="Mode Mixed Precision: 0, 1, atau 2")

    args = parser.parse_args()

    print(f"Model type: {args.model_type}, type: {type(args.model_type)}")
    print(f"PSO: {args.pso}, type: {type(args.pso)}")
    print(f"Mixed Precision Mode: {args.mp_mode}, type: {type(args.mp_mode)}")

    print("Running main function...")
    main(args.model_type, args.pso, args.mp_mode)
