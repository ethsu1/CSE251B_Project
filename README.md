# Neural Style Transfer

## Setup

### Anime Data

Anime images are loaded from the 54 imgur albums shared in
[this](https://www.reddit.com/r/anime/comments/5vez7c/is_there_any_site_that_shares_anime_scenery/de206au?utm_source=share&utm_medium=web2x&context=3) Reddit comment.
Download with:
```
pip install -r requirements.txt
python download.py
```

This will create a new directory `anime_data` and download all of the raw images into it. Dataset is made up of png files. Final size is ~52Gb (mostly 1080x1920 png images).

### Real-World Data

Real-world images come from the [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset. From your local repository directory, run:
```
scripts/download_ade20k.sh
```

This will create a new directory `real_data` download and unzip the dataset move all training images into `real_data`, and then delete everything else that was downloaded (annotations, etc.). Dataset is made up of jpg files. Final size is ~800Mb.

### Index Files for Torch Dataloader

Make sure the previous two steps have been completed. To create an index txt file for each dataset for use in the PyTorch dataloader, run from your local repository:
```
scripts/make_index_files.sh
```

These will be stored in their respective directories, `anime_data` and `real_data`
