import yaml


def main(x: int):
    # Load the config file
    config_file = yaml.safe_load(open("config.yml"))
    print(config_file["training"])


if __name__ == "__main__":
    x = 4
    main(x=x)
