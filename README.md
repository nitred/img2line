# About
This is a python module to fit a line to an image of a line. It returns the x and y coordinates of the line found in the image. It also returns a scikit-learn model that can be used to predict new y coordinates from new x coordinates.

To understand the module better, checkout the Jupyter Notebook in the `notebooks` folder!

# Limitations
- The module can only detect dark lines in a light background.
- The first dark pixel from the bottom of every column in the image is taken into consideration.
- `y = model.predict(x)`, requires `x` to be a 2-dimension `numpy.ndarray`. If `x` is 1-dimension, then use `x = x[:, np.newaxis]` to convert it to 2-dimensions.

# Installation
## Prerequisites
- `python 2.7`
- `numpy`
- `scikit-learn`
- `scikit-image`
- `matplotlib`

`requirements.txt` is not provided because numpy is painful to install using pip.

## Regular Installation
```
git clone https://github.com/nitred/img2line.git
cd img2line
python setup.py install
```

## Development Installation (Anaconda2)
```
git clone https://github.com/nitred/img2line.git
cd img2line
enable anaconda-2 environment
conda env create -f environment.yml
source activate img2line
python setup.py develop
```

# Usage
## Get Coordinates
```
import img2lile
x, y = img2lile.get_line_from_image("/home/user/line.png", plot=True)
```

## Fit Line
```
import img2line
x, y, model = img2line.fit_line_to_image("/home/user/line.png", degree=3, plot=True)
```

## Using Model To Predict New y-coordinates
```
x_new = np.arange(100)[:, np.newaxis]
y_new = model.predict(x_new)
```
