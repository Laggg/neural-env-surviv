from scripts.prepare_data import load_data, load_weights
from survivio_demo import demo_app


# main app
load_data()
load_weights()
demo_app()
