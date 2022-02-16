import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from constants import DATA_DIR, WEIGHTS_DIR


def load_data(data_dir=DATA_DIR) -> None:
    gdd.download_file_from_google_drive(
        file_id='1KlOTV_lyv85EozJS9GZfZsMjTaVX_Smq',
        dest_path=str(data_dir / 'dataset_for_moving.csv')
    )
    gdd.download_file_from_google_drive(
        file_id='1hB2IMk5oExIbIMUzUXYD6dUxSOwOX4Jr',
        dest_path=str(data_dir / 'dataset_inventory_v2.csv')
    )
    gdd.download_file_from_google_drive(
        file_id='1Fnn0F3Y0D_3oFaeRQq4umKllXdt-OITd',
        dest_path=str(data_dir / 'sample_rgb_96.zip'),
        unzip=True
    )

    os.remove(str(data_dir / 'sample_rgb_96.zip'))


def load_weights(weights_dir=WEIGHTS_DIR) -> None:
    gdd.download_file_from_google_drive(
        file_id='1uevD6lfB1QuyxbCUOX6SivHiYRQ75_jZ',
        dest_path=str(weights_dir / 'nostone_stone_classifier.pth')
    )
    gdd.download_file_from_google_drive(
        file_id='19dpmEkmoyw1z9Y-Xm_BG8-ZrteNhnw5Q',
        dest_path=str(weights_dir / 'resunet_v5.pth')
    )


if __name__ == '__main__':
    load_data(DATA_DIR)
    load_weights(WEIGHTS_DIR)
