# Basic Two-Time corraltion analysis code

## Installation
Basic installation commands are:

```
git clone <repository>
cd xpcs_analysis_cpp
git submodule update
conda create env -f xpcs_<your os>.yml
python3 -m pip install .
```

### Requirements
- c++ 17
- anaconda python
- Eigen (included as submodule)
- pybind11 (included as submodule)

### Submodules
To get the required submodules after cloning the repository the package run:

```
git submodule update
```

The required modules will be download in the `extern` folder.

### Jupyter kernel
To set up a basic kernel, after the conda environment is set up, run:

```
python3 -m ipykernel install --user --name=xpcs-kernel
```

## Example

To use the package after it is installed in your environment, you can use:

```
from xpcs_analysis_py import generateTTC

```

The `generateTTC` function takes a 2D integer (specifically `uint16`) array as an input, which should have it's first dimension being the flattened pixels of the ROI, and the second the number of images in the dataset. So for example for a measurement of 100 frames using a detector of size 50x50, should be converted to an array of shape 2500x100.
