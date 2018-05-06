Reproducing results with kerasformula
================

There are several sources of uncertainty when estimating a neural net with `kerasformula`. Optionally, `kms` uses `R` to split training and test data. Optionally, Python's `numpy` further splits the training data so that some can be used for validation, epoch-by-epoch. Finally, parallel processing or GPUs may introduce additional noise as batches are fed through. To reproduce results exactly, use the following syntax:

``` r
library(kerasformula)
movies <- read.csv("http://s3.amazonaws.com/dcwoods2717/movies.csv")

out <- kms(log10(gross/budget) ~ . -title, movies, scale="z",
           seed = list(seed = 12345, disable_gpu = TRUE, disable_parallel_cpu = TRUE))
```

    ___________________________________________________________________________
    Layer (type)                     Output Shape                  Param #     
    ===========================================================================
    dense_1 (Dense)                  (None, 256)                   355328      
    ___________________________________________________________________________
    dropout_1 (Dropout)              (None, 256)                   0           
    ___________________________________________________________________________
    dense_2 (Dense)                  (None, 128)                   32896       
    ___________________________________________________________________________
    dropout_2 (Dropout)              (None, 128)                   0           
    ___________________________________________________________________________
    dense_3 (Dense)                  (None, 1)                     129         
    ===========================================================================
    Total params: 388,353
    Trainable params: 388,353
    Non-trainable params: 0
    ___________________________________________________________________________

We can confirm this works that worked as follows:

``` r
out2 <- kms(log10(gross/budget) ~ . -title, movies, scale="z",
           seed = list(seed = 12345, disable_gpu = TRUE, disable_parallel_cpu = TRUE))
```

    ___________________________________________________________________________
    Layer (type)                     Output Shape                  Param #     
    ===========================================================================
    dense_1 (Dense)                  (None, 256)                   355328      
    ___________________________________________________________________________
    dropout_1 (Dropout)              (None, 256)                   0           
    ___________________________________________________________________________
    dense_2 (Dense)                  (None, 128)                   32896       
    ___________________________________________________________________________
    dropout_2 (Dropout)              (None, 128)                   0           
    ___________________________________________________________________________
    dense_3 (Dense)                  (None, 1)                     129         
    ===========================================================================
    Total params: 388,353
    Trainable params: 388,353
    Non-trainable params: 0
    ___________________________________________________________________________

``` r
out$MSE_predictions
```

    [1] 0.6909273

``` r
out2$MSE_predictions
```

    [1] 0.6909273

``` r
identical(out$y_test, out2$y_test)
```

    [1] TRUE

``` r
identical(out$predictions, out2$predictions)
```

    [1] TRUE

For other cases, to assess degree of convergence...

``` r
cor(out$predictions, out2$predictions)
```

         [,1]
    [1,]    1

``` r
cor(out$predictions, out2$predictions, method="spearman")
```

         [,1]
    [1,]    1

``` r
cor(out$predictions, out2$predictions, method="kendal") # typically last to converge
```

         [,1]
    [1,]    1

or to visually inspect weights...

``` r
get_weights(out$model)       # not run
get_weights(out2$model)
summary(out$model)           # also printed before fitting unless verbose = 0
```

`kms` implements a wrapper for `keras::use_session_with_seed`, which should also be called *before* compiling a model that is to be passed as an argument to `kms` (for an example, see the bottom of the [vignette](https://github.com/rdrr1990/kerasformula/blob/master/examples/kerasformula_vignette.md)). See also [stack](https://stackoverflow.com/questions/42022950/) and [tf](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) docs. Thanks to @VladPerervenko for helpful [suggestions](https://github.com/rdrr1990/kerasformula/issues/1) on this topic (mistakes are of course all mine)!

This toy data set is also used to show how to build [regression](https://github.com/rdrr1990/kerasformula/blob/master/examples/movies/predicting_film_profits.md) and [classification](https://github.com/rdrr1990/kerasformula/blob/master/examples/movies/kms_with_aws_movie.md) models too.
