import argparse


def main(model_type, pso, mp_mode):
    print(f"Model type: {model_type}, type: {type(model_type)}")
    print(f"PSO: {pso}, type: {type(pso)}")
    print(f"Mixed Precision Mode: {mp_mode}, type: {type(mp_mode)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script untuk menjalankan model dengan konfigurasi tertentu."
    )
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
    main(args.model_type, args.pso, args.mp_mode)
