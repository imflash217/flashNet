"""
Raw Python version of jupyter notebooks from fast.ai course
"""

__author__ = "Vinay Kumar"

## Importing the necessary libraries
from fastai import *
from fastai.vision import *

def pets():
    ## Setting the hyperparams
    bs = 16     # batch-size
    num_epochs = 4

    ## Looking at the data
    """We are going to use 'Oxford-IIIT Pet Dataset' by O.M.Parkhi et al., 2012 which features
    12 cat breeds and 25 dog breeds. Our model will need to learn to differentiate between these
    37 breeds. According to the paper the best accuracy they could get in 2012 was 59.21%, using
    a complex model the was specific to pet-detection, with separate 'Image', 'Head', 'Body' models
    for the pet photos. Let's see how accurate we can be using Deep Learning!

    We are going to use `untar_data` function to which we must pass a URL argument & which will
    download and extract the data.
    """

    print(help(untar_data))
    path = untar_data(URLs.PETS)
    print(path)
    print(path.ls())
    path_anno = path/"annotations"
    path_img = path/"images"

    """The first thing we do when we approach a problem is to take a look at the data.
    We always need to understand very well what the problem is and what kind of data is available
    before we can figure out how to solve it. Taking a look at the data means:
    1. Understanding how the data directories are sturctured,
    2. What the labels are, and
    3. What the sample data look like.

    The main difference b/w handeling of image-classification datasets is the way labels are stored.
    In this particular Oxford-pets dataset; the labels are stored in the filenames themselves.
    We will need to extract them to be able to classify the images into correct categories.
    We'll use fastai's `ImageDataBunch.from_name_re` method to get the labels using a
    regular-expression.
    """

    fnames = get_image_files(path_img)
    print(fnames[:5])

    np.random.seed(2)
    re_pat = r"/([^/]+)_\d+.jpg$"

    data = ImageDataBunch.from_name_re(path_img, fnames, re_pat, ds_tfms=get_transforms(),
                                       size=224, bs=bs).normalize(imagenet_stats)
    data.show_batch(rows=3, figsize=(7,6))
    print(data.classes)
    print(len(data.classes), data.c)

    ## Training resnet34
    """Now we will train our model. We will use a CNN backbone and a fully connected head with a
    single hidden layer as a classifier.
    """

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    print(learn.model)
    learn.fit_one_cycle(num_epochs)
    learn.save("stage-1")

    ## Results
    """
    Let's see what results have we got.
    We'll see what are the categories that the model most confused with one another.
    We will try to see if what the model predicted was reasonable or not.
    Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily
    skewed i.e. the model makes the same mistakes over and over again but  it rarely confuses
    other categories. This suggests that it just finds it difficult to distinguish b/e some specific
    categories, which is a normal behaviour.
    """

    interp = ClassificationInterpretation.from_learner(learn)
    losses, idxs = interp.top_losses()
    print(len(data.valid_ds), len(losses), len(idxs))
    interp.plot_top_losses(9, figsize=(15, 11))
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    interp.most_confused(min_val=2)

if __name__ == "__main__":
    print("running main")
    pets()