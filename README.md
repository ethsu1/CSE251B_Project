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

This will create a new directory `anime_data` and download all of the raw png images into it.

### Real-World Data

Real-world images come from the [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset. This can be downloaded and unzipped with:
```
./download_ade20k.sh
```

This will create a new directory `real_data` download and unzip the dataset, and then delete the .zip file.
