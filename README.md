# teco-eeg-nlp

EEG-based binary sentiment classification on the Persian TeCo dataset (task 1).

This is the per-participant, EEG-only model: a small 2D CNN trained on the
word-level EEG (total reading time) of 11 participants and tested on the
held-out one, repeated for everyone (leave-one-participant-out). Sentences are
labelled positive / negative; neutral ones are dropped, which leaves 110
sentences (55 / 55).

## Data

The data is not in the repo. You need two folders somewhere on disk:

- the `TRT_Total` folder with the 12 `<Name>_trt_total.pickle` files
- the folder holding `Total_Sentiment_Raw_1.xlsx`, `_2`, `_3` (the labels)

The Persian sentences live inside the pickles, so the Excel files are only used
for the sentiment labels.

## Run

The easy way: open `run.ipynb`, run the cells, and paste in the two data
folders and an output folder when it asks.

From a terminal:

```
pip install -r requirements.txt
python src/per_participant_eeg.py \
    --trt-dir   /path/to/TRT_Total \
    --labels-dir /path/to/sentences/Task_1 \
    --out-dir   results
```

It prints macro-F1 for each participant and the mean, and writes
`results/per_participant_eeg.csv`.

## Colab

```
!git clone <repo-url>
%cd teco-eeg-nlp
!pip install -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
```

then open `run.ipynb` and point the prompts at the folders on your Drive.

## Notes

Folds are tiny (~22 test sentences), so the per-participant numbers move around
a bit between runs; compare the mean. `--seed` fixes the split and init.
