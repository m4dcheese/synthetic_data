from config import config
from gym.eval import evaluate_on_real, evaluate_on_synthetic


def main():
    """Main function to run evaluation of hardcoded model path."""
    mode = "real"
    model = "results/30-11-2024/run_0/model_weights.pth"
    if mode == "real":
        evaluate_on_real(
            path=model,
            eval_config=config.training,
            datasets=["titanic", "student_depression"],
            only_quant=False,
        )
    else:
        evaluate_on_synthetic(
            path=model,
            eval_config=config.training,
            only_quant=False,
        )


if __name__ == "__main__":
    main()
