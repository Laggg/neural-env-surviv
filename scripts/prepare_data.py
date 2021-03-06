import os
import gdown
from constants import DATA_DIR, WEIGHTS_DIR


def load_data(data_dir=DATA_DIR) -> None:
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
