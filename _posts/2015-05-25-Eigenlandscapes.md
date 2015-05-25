---
layout: post
title: Eigenlandscape art
---

What happens when we run singular value decomposition (SVD) on images?
In this post I'll show how to do SVD on images with python and
some of the interesting visual effects that result.
![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_25.jpg "Eigenlandscape")

## Eigenfaces
[Eigenfaces](http://en.wikipedia.org/wiki/Eigenface)
are visualisations of the eigenvectors of the
matrix that you get when you stack vectors representing 
faces together. They come up in the field of
automatic image and face recognition.

![Eigenfaces from scolarpedia](http://www.scholarpedia.org/w/images/thumb/6/65/Eigenfaces.jpg/250px-Eigenfaces.jpg "http://www.scholarpedia.org/article/Eigenfaces")

## Decomposition with `sklearn`
[Klara-Marie den Heijer](http://www.klaramariedenheijer.com)
and I wondered what would happen
when you applied the same transform to landscape photographs.

The first test was to take a bunch of hiking photos,
flatten them, and visualise the components.

```python
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage import io, transform
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler

# Read images as numpy arrays
file_names = glob.glob('Pieke/*.jpg')
images = [io.imread(file_name) for file_name in file_names]
# Scale to something smaller
scaled_images = [transform.resize(image, (155, 234)) for image in images]

# At this point each image is a numpy ndarray of 
# shape (155, 234, 3).
# Flatten so we have ndarrays of shape (108810,)
fimages = [image.flatten() for image in scaled_images]

# Now do the decomposition
pca = IncrementalPCA()
timages = pca.fit_transform(fimages)

# Scale each component so we can interpret 
# them as pixel intensities
mms = MinMaxScaler()
ef = mms.fit_transform(pca.components_)

# Plot the sixth component
plt.imshow(ef[6].reshape((155, 234, 3)))
```

The resulting image (shown below) reminds me a lot of some of 
[Ydi Coetsee's](http://www.ydicoetsee.com/) work.

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_6.jpg "Eigenlandscape")

Here is my favourite painting by Ydi:

![Painting](/images/2015-05-25-Eigenlandscapes/IMGP0539.jpg "Painting")

# Real landscapes as input
Taking about 20 of Klara's landscape photographs:

![Photo](/images/2015-05-25-Eigenlandscapes/IMG_9185.jpg "Photo")

and pushing them through 
the algorithm gives some fascinating results:

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_7.jpg "Eigenlandscape")

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_10.jpg "Eigenlandscape")

It seems that we also inadvertently discovered a way to 
automatically generate kitsch watercolours:

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_29.jpg "Eigenlandscape")

## Larger image set
On a slightly larger set (85 images), the most interesting images
are found in the first few eigenvectors (corresponding to the
largest eigenvalues):

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_6 2.jpg "Eigenlandscape")

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_9.jpg "Eigenlandscape")

The very last component (in the sets we tried)
always has a different visual quality,
reminding me a lot of some 
[impressionist](http://en.wikipedia.org/wiki/Houses_of_Parliament_series_(Monet)) paintings.

![Eigenlandscape](/images/2015-05-25-Eigenlandscapes/output_85.jpg "Eigenlandscape")

Note the horse silhouette in the center of the image above, and the
recurring telephone poles in this set - some of the
features are hard to dilute.

## Conclusion
There are a few more things I want to try, like 

- different ways to scale the intensities of the components,
- different (and possibly much larger) sets of images,
- passing the resulting images through the algorithm again (initially
 I thought you would get the same images back but after some quick
 experimentation this is not the case - and after repeating 100 times
 there is a lot of quality loss).

A quick google search only found [this](http://www.cs.colostate.edu/~idfah/main/publications/art) similar investigation - maybe I'm not using
the right terms?

Klara also plans to paint some of the images as part of
an ongoing study on something (ask her to explain).
