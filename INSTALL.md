# Installing kerasformula

This document provides install instructions to handle recent
version changes in both the relevant `R` and `Python` libraries.
`Conda` environments likely still 
need to update the packages mentioned below but more detail will 
be provided on that and other installation routes.

## Python 2.7 Instructions

Here are instructions for `Python 2.7.10`. Upgrading Python with `brew` is recommended.

Enter the followings shell commands to create a hidden folder where
the `R` `library(keras)` and `library(kerasformula)` will look for the `Python` 
copy of `keras`. Do not use the R function `keras::install_keras()`,
which creates a virtual environment with an outdated version of `pip`
that cannot complete the installation. 

```
virtualenv .virtualenvs/r-tensorflow        
source .virtualenvs/r-tensorflow/bin/activate
```
Next, you likely need to upgrade `pip` since older versions of `pip` 
that come bundled with `Python2` are deemed insecure, preventing installation ( [details on Stack]() ).

```
curl https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```
Next install the following packages...

```
pip install --upgrade setuptools utils np_utils
pip install tensorflow
pip install keras
```
Now, open R. You can confirm the install worked as follows.
```
keras::is_keras_available()
# should return TRUE
```
Finally, test a model out.

```
library(kerasformula)
out <- kms(mpg~., mtcars, verbose=0)
```
The version requirements on both the `R` and the `Python` side are very strict. Without current versions at least certain data objects in `R` will be mishandled by `Python`, throwing an error, even before the model is estimated in `Tensorflow`. 
These instructions have been tested on both `R 3.5.0` and `R 3.6.0`.
Here is the session info for the latter:
```
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
``
