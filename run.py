import argparse

from experiments import benchmark, save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/audit.json")
    args = parser.parse_args()
    results = benchmark()
    save(results, args.output)
    for method, metric in results.items():
        print(f"{method}: oracle divergence={metric['oracle_divergence']:.4f}, canary confidence={metric['canary_confidence']:.3f}")


if __name__ == "__main__":
    main()
