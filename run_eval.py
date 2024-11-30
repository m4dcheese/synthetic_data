from config import config
from gym.eval import evaluate_on_real, evaluate_on_synthetic


def main():
    """Main function to run evaluation of hardcoded model path."""
    mode = "real"
    if mode == "real":
        evaluate_on_real(
            path="results/29-11-2024/run_2/model_weights.pth",
            eval_config=config.training,
            datasets=["titanic"],
        )
    else:
        evaluate_on_synthetic(
            path="results/29-11-2024/run_2/model_weights.pth",
            eval_config=config.training,
        )


if __name__ == "__main__":
    main()
