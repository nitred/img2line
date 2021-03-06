# About
This is a python module to fit a line to an image of a line. It returns the x and y coordinates of the line found in the image. It also returns a scikit-learn model that can be used to predict new y coordinates from new x coordinates.

To understand the module better, checkout the Jupyter Notebook in the `notebooks` folder!

# Limitations
- The module can only detect dark lines in a light background.
- The first dark pixel from the bottom of every column in the image is taken into consideration.
- `y = model.predict(x)`, requires `x` to be a 2-dimension `numpy.ndarray`. If `x` is 1-dimension, then use `x = x[:, np.newaxis]` to convert it to 2-dimensions.

# Installation
## Pip Installation
```
pip install git+https://github.com/nitred/img2line.git --upgrade
```

## Regular Installation
```
git clone https://github.com/nitred/img2line.git
cd img2line
python setup.py install
```

## Development Installation (Anaconda3)
```bash
$ git clone https://github.com/nitred/img2line.git
$ cd img2line
enable anaconda3 environment
$ conda env create --force -f dev_environment.yml
$ source activate img2line
$ python setup.py develop
```

# Usage
## Get Coordinates
```python
import img2line
x, y = img2line.get_line_from_image("/home/user/line.png", plot=True)
```

## Fit Line
#### Polynomial Regression
```python
import img2line
x, y, model = img2line.fit_line_to_image("/home/user/line.png", degree=3, plot=True)
```

#### Gaussian Process Regression
```python
import img2line
x, y, model = img2line.fit_gpr_to_image("/home/user/line.png", plot=True)
```


## Using Model To Predict New Y-coordinates
```python
# obtain `model` from either `fit_line_to_image` or `fit_gpr_to_image`

x_new = np.arange(100)[:, np.newaxis]
y_new = model.predict(x_new)
```
