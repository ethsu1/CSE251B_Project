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

This will create a new directory `real_data`, download and unzip the dataset, move all training images into `real_data`, and then delete everything else that was downloaded (annotations, etc.). Dataset is made up of jpg files. Final size is ~800Mb.

We also set aside the first 1000 real data samples into a separate directory to perform testing of our style transfer methods. From your local repository directory, run:
```
scripts/create_real_content_set.sh
```

This creates a new directory `content_samples` and moves the first 1000 ADE20K samples to it.

### Index Files for Torch Dataloader

Make sure the previous two steps have been completed. To create an index txt file for each dataset for use in the PyTorch dataloader, run from your local repository:
```
scripts/make_index_files.sh
```

These will be stored in their respective directories, `anime_data` and `real_data`

## Training Real/Anime Classifier

Once the images in the previous section are downloaded we can finetune a ResNet18 network with weights pretrained on ImageNet with:
```
python train_anime_classifier.py
```

Options can be specified with the following arguments:
```
arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 4)
  --gpu GPU             index of gpus to use (for two, use --gpu 0,1)
  --epochs N            number of total epochs to run (default: 5)
  --arch ARCH           model architecture: (default: basic_fcn)
  -b N, --batch-size N  mini-batch size (default: 32)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 1e-3)
  --save-dir PATH       path to directory for saved outputs (default: extra/)
  --img-size N          dimension to resize images to (square, default: 256)
  --no-color            make all images monochrome

```

Note that, partly because the dataset contains ~35k images, it takes 6 minutes to go through a single training epoch (on a single 1080Ti), so adjust `--epochs` accordingly.

The model with lowest validation loss will be saved to the `extra/` directory unless a specific one is specified with `--save-dir`.

## Performing Evaluation w/ Real/Anime Classifier

Once the classifier is trained and the model has been saved somewhere, run:
```
python evaluate_images.py --model-path <MODEL_PATH> --image-dir <IMAGE_DIR>
```

where `<IMAGE_DIR>` is a path to the directory containing only images to be evaluated and `<MODEL_PATH>` is the path to the .pth.tar file saved in `train_anime_classifier.py`.

## Running Style Transfer Experiments on Individual Images
Different experiments can be run by changing `default.json` and then run:
```
python main.py
```

The following configuration variables are important for running individual experiments:

`random_noise`: input image to optimize
  - `True`: use random noise
  - `False`: use copy of content image

`model_name`: pretrained model on ImageNet
  - `resnet34`
  - `densenet121`
  - `vgg` (default)

`output_title`: filename of output NST image

`content_layers`: layer used to represent content (vgg only)

`style_layers`: layers used to represent style (vgg only)

`learning_rate`: learning rate used by optimizer

`num_epochs`: number of iterations to run

`content_weight`: how important content loss is (beta)

`style_weight`: how important style loss is (alpha)

`loss`: loss function
  - `l1`
  - `huber`
  - `l2` (default)

`optimizer`: optimization algorithm
  - `adam` 
  - `rmsprop`
  - `lbfgs` (default)

## Generating Style-transferred Images (for Test Data)
A large quantity of style-transferred images can be created by changing `default.json` and then run:
```
python create_test_nst.py
```
Within `output_dir_path`, a new subdirectory will be created having the name "`model_name`_ `loss`_`optimizer`". The style-transferred file names will be the concatenation of the content file name and the style file name.

In the event that the file crashes, it will record your progress such that you can start from where you left off. A file called `content_prog.txt` will be created that keeps track of the progress you have made via the name of the content files. You can specify to use this file by running:
```
python create_test_nst.py content_prog
```

The following configuration variables are important for generating style-transferred images:

`content_dir_path`: path of directory containing all the content images

`style_dir_path`: path of directory containing all the style images

`output_dir_path`: path of directory for output images

`model_name`: pre-trained model on ImageNet
  - `resnet34`
  - `densenet121`
  - `vgg` (default)

`content_layers`: layer used to represent content (vgg only)

`style_layers`: layers used to represent style (vgg only)

`learning_rate`: learning rate used by optimizer

`num_epochs`: number of iterations to run

`content_weight`: how important content loss is (beta)

`style_weight`: how important style loss is (alpha)

`loss`: loss function
  - `l1`
  - `huber`
  - `l2` (default)

`optimizer`: optimization algorithm
  - `adam` 
  - `rmsprop`
  - `lbfgs` (default)
