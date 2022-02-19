import os
import gdown
from constants import DATA_DIR, WEIGHTS_DIR


def load_data(data_dir=DATA_DIR) -> None:
    gdown.cached_download(
        id='1KlOTV_lyv85EozJS9GZfZsMjTaVX_Smq',
        path=str(data_dir / 'dataset_for_moving.csv'),
        md5='cc1997c27e84f6a92ce90d9dc7d0bc16',
    )
    gdown.cached_download(
        id='1hB2IMk5oExIbIMUzUXYD6dUxSOwOX4Jr',
        path=str(data_dir / 'dataset_inventory_v2.csv'),
        md5='151532b6f190d9a79b309c9d457b004c',
    )
    if not os.path.exists(str(data_dir / 'sample_rgb_96')):
        gdown.cached_download(
            id='1Fnn0F3Y0D_3oFaeRQq4umKllXdt-OITd',
            path=str(data_dir / 'sample_rgb_96.zip'),
            md5='a99a6885ea4d3de29e5f31fffb9a4fd0',
            postprocess=gdown.extractall
        )

    if os.path.exists(str(data_dir / 'sample_rgb_96.zip')):
        os.remove(str(data_dir / 'sample_rgb_96.zip'))


def load_weights(weights_dir=WEIGHTS_DIR) -> None:
    gdown.cached_download(
        id='1NxlyI7Keh3ItvhZ1Yyh9gmM021mw5f-K',
        path=str(weights_dir / 'nostone_stone_classifier_v2.pth'),
        md5='11e2cebe581babc7dcfe53af6204eb22',
    )
    gdown.cached_download(
        id='19dpmEkmoyw1z9Y-Xm_BG8-ZrteNhnw5Q',
        path=str(weights_dir / 'resunet_v5.pth'),
        md5='c6bd42cbdc2951193f9e9213ac006217'
    )
    gdown.cached_download(
        id='1YRTX84ea-ley7o6CGqp3q9yDpNdczBRv',
        path=str(weights_dir / 'dqn_v7.pth'),
        md5='d14f43eb8f27e60cff7d58d164598527',
    )


if __name__ == '__main__':
    load_data(DATA_DIR)
    load_weights(WEIGHTS_DIR)
