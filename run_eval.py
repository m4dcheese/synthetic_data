from config import config
from gym.eval import evaluate, evaluate_with_odeint


def main():
    """Main function to run evaluation of hardcoded model path."""
    # evaluate(path="model_weights.pth", eval_config=config.training)
    evaluate_with_odeint(
        path="results/14-11-2024/run_3/model_weights.pth", eval_config=config.training
    )


if __name__ == "__main__":
    main()
