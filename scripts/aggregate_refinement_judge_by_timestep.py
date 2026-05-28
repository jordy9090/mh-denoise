import argparse

from aggregate_refinement_judge_scores import aggregate_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    n_rows, n_groups = aggregate_file(args.input, args.output_csv, ["system", "mode", "t"])
    print(f"input rows: {n_rows}")
    print(f"groups: {n_groups}")
    print(f"saved: {args.output_csv}")
    print("toxicity convention: lower is better; quality_safety_average uses 6 - toxicity")


if __name__ == "__main__":
    main()
