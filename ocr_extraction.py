

from rapidocr_onnxruntime import RapidOCR
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np
import re
import requests
from PIL import Image


engine = RapidOCR(det_use_cuda=True)

def image_ocr(path):
    img = Image.open(requests.get(path, stream=True).raw)
    result,elapse = engine(img)
    _,text,_ = list(zip(*result))
    return " ".join(text)

if len(sys.argv) != 2:
    print("\n\n" + "=" * 3 + " There should have been exactly 2 arguments " + "=" * 3 + "\n\n")
    sys.exit()

if input("\n\nHave you ran the preprocessing ipynb file? If yes then type YES else go run it BC!!: ").lower() != "yes":
    print("=== SUDRA ===")
    sys.exit()

# get the file idx
file_part_idx = int(sys.argv[-1])

# if file not in partition range
if file_part_idx not in range(0, 13):
    print("\n\n" + "=" * 3 + " File idx not in range " + "=" * 3 + "\n\n")
    sys.exit()

# =========================================================================================================

# logs files
print("\n\n" + "=" * 3 + " Creating log files directory" + "=" * 3 + "\n\n")
logs_dir = "./logs"

def extract_the_last_index(file_path):
    '''
        gets the index of the last ocr extracted record
    '''
    file = open(file_path, 'r')
    
    lines = file.readlines()

    pattern = r'\[(.*?)\]'
    
    if lines:
        last_line = lines[-1].strip()
        matches = re.findall(pattern, last_line)
        if len(matches) == 1:
            return int(matches[0])
    return 0


# create logs dir
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"{logs_dir} created")


start_index = 0

# create log file 
log_file_path = os.path.join(logs_dir, f"logs_{file_part_idx}.txt")

if os.path.exists(log_file_path):
    # if log file exists open it in append and read mode
    # read the last record
    # and append new records saved
    print(f"{log_file_path} detected")
else:
    # w+ to create a file and have read and write
    print(f"The log file '{log_file_path}' did not exist and has been created.")

log_file = open(log_file_path, 'a+')


start_index = extract_the_last_index(log_file_path) 

print("\n\n" + "=" * 3 + f" Starting index from {start_index} " + "=" * 3 + "\n\n")

# =========================================================================================================

# read partitioned csv
partitions_dir = './partitions'
partitions_dir_with_ocr = './partitions'

# create partitions dir with ocr 
if not os.path.exists(partitions_dir_with_ocr):
    os.makedirs(partitions_dir_with_ocr)
    print(f"{partitions_dir_with_ocr} created")

print('\n\n' + '=' * 3 + f"All the partition dfs with extracted ocr test will be stored in dir {partitions_dir_with_ocr} " + "=" * 3 + '\n\n')

# =========================================================================================================

img_saved_dir_path = "./images"
df = pd.read_csv(os.path.join(partitions_dir, f"partition_{file_part_idx}.csv"))

# ocr_text column to hold the extracted ocr text values
#df['ocr_text'] = np.nan

# go through each row and append its ocr text in ocr_text column
try:
    for index, row in df.iloc[start_index:].iterrows():

        image_link = row['image_link']

        #file_path = os.path.join(img_saved_dir_path, file_name)

        ocr_text = image_ocr(image_link)

        df.at[index, 'ocr_text'] = ocr_text

        # write in log files
        log_file.write(f"OCR for [{index}] : image link: {image_link}\n")

        if index % 10 == 0:
            print(f"Saved till {index}")
            df.to_csv(os.path.join(partitions_dir_with_ocr, f"partition_{file_part_idx}.csv"), index=False)

except KeyboardInterrupt:
    log_file.close()
    df.to_csv(os.path.join(partitions_dir_with_ocr, f"partition_{file_part_idx}.csv"), index=False)
    print(f"\nCtrl+C detected. Saving log file and csv parititon_{file_part_idx}.csv till index {index}")


print('\n\n' + '=' * 3 + f"Log file closed" + "=" * 3 + '\n\n')
