# Assignment 3 - AI imaging case study

Code for the nuclei image analysis assignment. The dataset is synthetic fluorescence
microscopy images of cell nuclei, 256x256 RGB, with masks and instance labels.
80 train / 20 val / 12 test.

## Install

Python 3.14, and Ollama must be installed and running.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3:4b
ollama pull llama3
```

Dataset: download nuclei_dataset.zip from
https://github.com/Nickolay-K/Assingnment-3-dataset
and unzip it into `data/raw/`, so you get `data/raw/nuclei_dataset/train/images/...`

## Run

From the project root, in this order:

```bash
python3 -m src.eda        # task 1 figures
python3 -m src.llm        # task 1 vlm description
python3 -m src.classical  # task 2 otsu + watershed
python3 -m src.train      # task 3 unet, ~2 min on M1 Pro
python3 -m src.pipeline   # task 4 full pipeline on the test images
```

src.llm and src.pipeline need Ollama running. The others do not.

Figures go to outputs/figures, tables to outputs/csv, model answers to outputs/json,
weights to outputs/models.

## Notebooks

The scripts above make the figures and the tables. The notebooks are where I compare things
and say what the numbers mean, so they are the easier place to read. They are saved with
their outputs, so you can look at them without running anything.

- notebooks/01_eda.ipynb - the data, the four density regimes, and a problem with the
  ground truth counts
- notebooks/02_vlm.ipynb - the vision model, naive prompt against structured prompt, and
  what it says about the corrupted images
- notebooks/03_classical.ipynb - otsu and watershed, choosing the seed spacing, and the
  description made only from numbers
- notebooks/04_unet_pipeline.ipynb - training, u-net against otsu, the loss comparison, the
  full pipeline, and what happens when the image is damaged

They need `sys.path.append("..")` at the top, which is already there.

## Code

- config.py - settings and paths
- src/data.py - file paths and png reading
- src/preprocess.py - grayscale, normalise, resize
- src/meta_data.py - metadata.csv
- src/prepare_torch.py - Dataset and augmentation
- src/data_loader.py - dataloaders
- src/eda.py - figures
- src/llm.py - Ollama calls and json parsing
- src/classical.py - otsu, watershed, regionprops
- src/unet.py - the unet from lab 4
- src/train.py - training and evaluation
- src/pipeline.py - mask to features to json to narrative
- prompts/ - prompt files

## Notes

I used gemma3:4b as the vision model instead of llama3.2-vision. I downloaded
llama3.2-vision but Ollama 0.32.14 fails to load it:

```
error loading model: unknown model architecture: 'mllama'
```

For counting I compare against the label maps, not the n_objects column in
metadata.csv. The dataset generator draws nuclei one over another, so a nucleus that
is fully covered still counts in metadata but has no pixels left in the image. 23 of
the 112 images are affected, mostly the clustered and dense ones.
