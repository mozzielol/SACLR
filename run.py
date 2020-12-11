from models.saclr import SaCLR
import yaml
from util.dataloader import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SaCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
