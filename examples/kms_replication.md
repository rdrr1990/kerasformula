Reproducing results with kerasformula
================

There are several sources of uncertainty when estimating a neural net with `kerasformula`. Optionally, `kms` uses `R` to split training and test data. Optionally, Python's `numpy` further splits the training data so that some can be used for validation, epoch-by-epoch. Finally, parallel processing or GPUs may introduce additional noise as batches are fed through. To reproduce results exactly, use the following syntax:

``` r
knitr::opts_chunk$set(message=FALSE, warning=FALSE, comment="")
```

``` r
library(kerasformula)
movies <- read.csv("http://s3.amazonaws.com/dcwoods2717/movies.csv")

out <- kms(log10(gross/budget) ~ . -title, movies, scale="z", batch_size = 1, Nepochs = 15,
           seed = list(seed = 12345, disable_gpu = TRUE, disable_parallel_cpu = TRUE))
```

We can confirm this works that worked as follows:

``` r
out2 <- kms(log10(gross/budget) ~ . -title, movies, scale="z", batch_size = 1, Nepochs = 15,
           seed = list(seed = 12345, disable_gpu = TRUE, disable_parallel_cpu = TRUE))

out$MSE_predictions
```

    [1] 0.7383057

``` r
out2$MSE_predictions
```

    [1] 0.7383057

``` r
identical(out$y_test, out2$y_test)
```

    [1] TRUE

``` r
identical(out$predictions, out2$predictions)
```

    [1] TRUE

`kms` implements a wrapper for `keras::use_session_with_seed`. See also [stack](https://stackoverflow.com/questions/42022950/) and [tf docs](https://www.tensorflow.org/api_docs/python/tf/set_random_seed). Thanks to @VladPerervenko for helpful [suggestions](https://github.com/rdrr1990/kerasformula/issues/1) on this topic (mistakes are of course all mine)! 

This toy data set is also used to show how to build [regression](https://github.com/rdrr1990/kerasformula/blob/master/examples/movies/predicting_film_profits.md) and [classification](https://github.com/rdrr1990/kerasformula/blob/master/examples/movies/kms_with_aws_movie.md) models too.
