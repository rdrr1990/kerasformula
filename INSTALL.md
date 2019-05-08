# Installing kerasformula

This document provides install instructions to handle recent
version changes in both the relevant `R` and `Python` libraries.

Choose either `Python 3.x` (recommended) or `Python 2.7`. The version 
requirements are very **strict** on both `R` and the `Python` side. Though particular combinations of older libraries
still work, in general, upgrading everything is recommended.

(`Conda` environments likely still 
need to update the packages mentioned below but more detail will 
be provided on that and other installation routes.)


## Python3 Instructions

These instructions were confirmed using `Python 3.7.3` (on `Mac OSX Sierra 10.12.6`). Enter the following shell command:
```console
brew install python3
```
The following instructions are lightly adapted from [here](https://irudnyts.github.io/custom-set-up-of-keras-and-tensorflow-for-r-and-python/); if the above command doesn't work, see details there for background requirements.
```console
pip3 install tensorflow
pip3 install keras
```
To skip `Anaconda`'s bundling, which isn't necessary for `kerasformula`, the following packages are recommended too:
```console
pip3 install jupyter_client ipykernel numpy pandas matplotlib jupyter
```
Now open R.
```R
if(!require(keras)) install.packages("keras")
if(!require(kerasformula)) devtools::install_github("rdrr1990/kerasformula")

reticulate::use_python("/usr/local/bin/python3")
```
You can confirm the install worked as follows.
```R
library(kerasformula)
out <- kms(mpg~., mtcars, verbose=0)
```

### Troubleshooting Python3 Installation

If the above `kms` command throws an error, check the path for `python3`. In `R`:
```R
system("which python3")
```
Then use that path with the `reticulate::use_python` command shown above.

If that's not the issue, upgrade Python to be at least 3.7.3.

The version requirements on both the `R` and the `Python` side are very strict. Without current versions at least certain data objects in `R` will be mishandled by `Python`, throwing an error, even before the model is estimated in `Tensorflow`. 
These instructions have been tested on both `R 3.5.0` and `R 3.6.0`.
Here is the session info for the latter:

```R
> sessionInfo()
R version 3.6.0 (2019-04-26)
Platform: x86_64-apple-darwin16.7.0 (64-bit)
Running under: macOS Sierra 10.12.6

Matrix products: default
BLAS:   /Users/mohanty/Dropbox/R-3.6.0/lib/libRblas.dylib
LAPACK: /Users/mohanty/Dropbox/R-3.6.0/lib/libRlapack.dylib

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
[1] reticulate_1.12    kerasformula_1.7.0 Matrix_1.2-17      dplyr_0.8.0.1     
[5] keras_2.2.4.1     

loaded via a namespace (and not attached):
 [1] Rcpp_1.0.1        whisker_0.3-2     magrittr_1.5      tidyselect_0.2.5 
 [5] munsell_0.5.0     colorspace_1.4-1  lattice_0.20-38   R6_2.4.0         
 [9] rlang_0.3.4       plyr_1.8.4        grid_3.6.0        gtable_0.3.0     
[13] tfruns_1.4        lazyeval_0.2.2    assertthat_0.2.1  tibble_2.1.1     
[17] crayon_1.3.4      tensorflow_1.13.1 purrr_0.3.2       ggplot2_3.1.1    
[21] base64enc_0.1-3   zeallot_0.1.0     glue_1.3.1        compiler_3.6.0   
[25] pillar_1.3.1      generics_0.0.2    scales_1.0.0      jsonlite_1.6     
[29] pkgconfig_2.0.2  
```

## Python 2.7

In terminal, check to see if your version of `pip` is new enough to install packages.
```console
pip install utils np_utils
```
If that command throws an error about internet protocol security  ( [details on Stack]() ), upgrade pip as follows:
```console
curl https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```
Next, install the following libraries
```console
pip install utils np_utils
pip install --upgrade setuptools
pip install --upgrade tensorflow
pip install --upgrade keras
```


## Python 2.7 in a Virtual Environment

Here are instructions for `Python 2.7.10` in a virtual environment. 
(These instrucitons will accomplish what `keras::install_keras` aims to
by default. However, due to some of the issues discussed below, these
are recommended instead of that configuration function.)
These is the most complicated route, in part because the `Python 2` 
that ships with many Macs contains a no longer functioning version 
of pip. Upgrading Python with `brew` is recommended.
```console
brew upgrade python
```
Enter the followings shell commands to create a hidden folder where
the `R` `library(keras)` and `library(kerasformula)` will look for the `Python` 
copy of `keras`. Do not use the R function `keras::install_keras()`,
which creates a virtual environment with an outdated version of `pip`
that cannot complete the installation. 

```console
virtualenv .virtualenvs/r-tensorflow        
source .virtualenvs/r-tensorflow/bin/activate
```
Next, you likely need to upgrade `pip` since older versions of `pip` 
that come bundled with `Python2` are deemed insecure, preventing installation ( [details on Stack]() ).

```console
curl https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```
Next install the following packages...

```console
pip install --upgrade setuptools utils np_utils
pip install tensorflow
pip install keras
```
Check the path to `Python`, which you'll need in a moment.
```console
which python
```
Now, open `R`.
```R
if(!require(keras)) install.packages("keras")
if(!require(kerasformula)) devtools::install_github("rdrr1990/kerasformula")
```
Let `R` know about the version of `Python` you want:
```R
reticulate::use_python("/usr/bin/python")
```
You can confirm the install worked as follows.
```R
library(kerasformula)
out <- kms(mpg~., mtcars, verbose=0)
```

```R
if(!require(keras)) install.packages("keras")
if(!require(kerasformula)) devtools::install_github("rdrr1990/kerasformula")
```
You can confirm the install worked as follows.
```R
library(kerasformula)
out <- kms(mpg~., mtcars, verbose=0)
```
### Troubleshooting Python 2.7 virtual environment install

If the above `kms` command throws an error, 
check whether `keras` installed correctly.
```R
keras::is_keras_available()
```
If that returns `TRUE` but the `kerasformula` example above does not work, 
it is likely because either `Python` is outdated or some of the dependencies are.


The version requirements on both the `R` and the `Python` side are very strict. Without current versions at least certain data objects in `R` will be mishandled by `Python`, throwing an error, even before the model is estimated in `Tensorflow`. 
These instructions have been tested on both `R 3.5.0` and `R 3.6.0`.
Here is the session info for the latter:
```R
> sessionInfo()
R version 3.6.0 (2019-04-26)
Platform: x86_64-apple-darwin16.7.0 (64-bit)
Running under: macOS Sierra 10.12.6

Matrix products: default
BLAS:   /Users/mohanty/Dropbox/R-3.6.0/lib/libRblas.dylib
LAPACK: /Users/mohanty/Dropbox/R-3.6.0/lib/libRlapack.dylib

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
[1] kerasformula_1.7.0 Matrix_1.2-17      dplyr_0.8.0.1      keras_2.2.4.1     

loaded via a namespace (and not attached):
 [1] Rcpp_1.0.1        whisker_0.3-2     magrittr_1.5      tidyselect_0.2.5 
 [5] munsell_0.5.0     colorspace_1.4-1  lattice_0.20-38   R6_2.4.0         
 [9] rlang_0.3.4       plyr_1.8.4        grid_3.6.0        gtable_0.3.0     
[13] tfruns_1.4        lazyeval_0.2.2    assertthat_0.2.1  tibble_2.1.1     
[17] crayon_1.3.4      tensorflow_1.13.1 purrr_0.3.2       ggplot2_3.1.1    
[21] base64enc_0.1-3   zeallot_0.1.0     glue_1.3.1        compiler_3.6.0   
[25] pillar_1.3.1      generics_0.0.2    scales_1.0.0      reticulate_1.12  
[29] jsonlite_1.6      pkgconfig_2.0.2  
```
