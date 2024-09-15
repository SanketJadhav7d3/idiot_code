

from rapidocr_onnxruntime import RapidOCR
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np
import re


engine = RapidOCR(det_use_cuda=True)

def image_ocr(path):
    result,elapse = engine(path)
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

def extract_the_last_index(file):
    '''
        gets the index of the last ocr extracted record
    '''
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
    log_file = open(log_file_path, 'r+')
    print(f"{log_file_path} detected")
else:
    # w+ to create a file and have read and write
    log_file = open(log_file_path, 'w+')
    print(f"The log file '{log_file_path}' did not exist and has been created.")

start_index = extract_the_last_index(log_file)

print("\n\n" + "=" * 3 + f" Starting index from {start_index} " + "=" * 3 + "\n\n")

# =========================================================================================================

# read partitioned csv
partitions_dir = './partitions'
partitions_dir_with_ocr = './partitions-ocr'

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
for index, row in df.iloc[start_index:].iterrows():
    file_name = row['file_name']
    file_path = os.path.join(img_saved_dir_path, file_name)

    ocr_text = image_ocr(file_path)

    df.at[index, 'ocr_text'] = ocr_text

    # write in log files
    log_file.write("OCR for [" + str(index) + "]\n")

    if index % 10 == 0:
        print(f"Saved till {index}")
        df.to_csv(os.path.join(partitions_dir_with_ocr, f"partition_{file_part_idx}.csv"))

print(df)
