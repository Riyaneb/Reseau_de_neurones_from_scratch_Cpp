from datasets import load_dataset
import os
from pathlib import Path

def clean_txt(df_txt):
    serie = df_txt.astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.replace(r"\\", "", regex=True).str.strip()
    return serie


depot_name = "MaxLSB/LeCarnet"

dataset = load_dataset(depot_name)

df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()
df_val = dataset["validation"].to_pandas()

path_begin = Path("data/dataset_text/MaxLSB_LeCarnet_dataset")

os.makedirs(Path(path_begin / "train"), exist_ok=True)
os.makedirs(Path(path_begin / "mini_train"), exist_ok=True)
os.makedirs(Path(path_begin / "test"), exist_ok=True)
os.makedirs(Path(path_begin / "val"), exist_ok=True)

path_train = Path(path_begin / "train/MaxLSB_LeCarnet_train.txt")
path_mini_train = Path(path_begin / "mini_train/MaxLSB_LeCarnet_mini_train.txt")
path_test = Path(path_begin / "test/MaxLSB_LeCarnet_test.txt")
path_val = Path(path_begin / "val/MaxLSB_LeCarnet_val.txt")

serie_train = clean_txt(df_train["text"])
serie_test = clean_txt(df_test["text"])
serie_val = clean_txt(df_val["text"])

with open(path_train, "w", encoding="utf-8", newline="\n") as f:
    for txt in serie_train:
        f.write(txt)
        f.write("\n")

with open(path_train, "rb") as f:
    octets_txt = f.read(10000000)
    pos = octets_txt.rfind(b"\n")
    with open(path_mini_train, "wb") as f2 :
        f2.write(octets_txt[:pos+1])


with open(path_test, "w", encoding="utf-8", newline="\n") as f:
    for txt in serie_test:
        f.write(txt)
        f.write("\n")

with open(path_val, "w", encoding="utf-8", newline="\n") as f:
    for txt in serie_val:
        f.write(txt)
        f.write("\n")