from config import config
from gym.eval import evaluate


def main():
    """Main function to run evaluation of hardcoded model path."""
    evaluate(path="model_weights_linreg_nov26.pth", eval_config=config.training)


if __name__ == "__main__":
    main()
