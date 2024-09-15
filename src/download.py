
import pandas as pd
from utils import download_images

test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))


download_images(test['image_link'], '../images', allow_multiprocessing=False)
