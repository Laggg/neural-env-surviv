from scripts.prepare_data import load_data, load_weights
from demo_app.survivio_demo import demo_app


# main app
load_data()
load_weights()
demo_app()
