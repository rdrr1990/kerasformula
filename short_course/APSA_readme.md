Building Neural Networks in R for Political Research
================
Pete Mohanty
8/14/2018

Political scientists are increasingly interested in machine learning approaches such as neural networks. Neural networks offer predictive accuracy in spite of complex data generating processes and may also aid researchers interested in examining the scope conditions of inferential claims. Until recently, the programming requirements for neural networks have been much steeper for neural networks than for statistical techniques like regression (perhaps not unlike the early days of Bayesian Markov Chain Monte Carlo) and many of the best techniques were limited to `Python`. This workshop introduces the theory behind neural networks and shows how to build them in `R` using the library `kerasformula`. The workshop will provide political examples such as Twitter data and Congressional forecasting. These examples will also serve to highlight the comparative strengths and weaknesses of neural networks in comparison with classical statistical approaches. The library `kerasformula` is a high-level interface for `Keras` and `Tensorflow` in `R` that allows researchers to fit a model in as little as one line of code and which allows for a high degree of customization (shape and depth of the network, loss and activation function, etc.). The workshop will be conducted in an ‘active learning’ paradigm whereby mini-lectures will alternate with hands-on coding activities. Participants will be encouraged to bring a sample of their own data and to build a working prototype by the end of the day. Some familiarity with `R` and `RStudio` is assumed but participants need not be advanced coders.

Data
====

Participants should have a sample of their own data in a `data.frame` which is clean enough to run a regression on. Alternatively, code will also be provided to quickly construct such a `data.frame` (similar to the data used in the slides).

Software
========

This course requires that that the `R` library `kerasformula` (version 1.5.0 or higher) be installed, as well as it's depedencies. How much fuss that is depends a bit on your computer (whether it's Windows or Mac, what you've already installed, and so on). Please note, due to various compability issues, (legacy) `Python 2.7` is recommended, not (current) `Python 3.x`.

-- **The Cloud** (fastest, simplest install). In your web browser, go to <https://rstudio.cloud> and make a free account and then click to start a new project and open `RStudio` in your browser. Proceed with **Mac Desktop** instructions

-- **Mac Desktop**

Open `R` or `RStudio` and enter the following into the `Console`:

``` r
install.packages("RCurl")           # may be needed to download data from GitHub
install.packages("kerasformula")
library(kerasformula)
install_keras()                     # run only once
```

`install_keras()` is run only once on each computer (including if you use `https://rstudio.cloud`). `install_keras()` also provides high performance computing options (`GPU`) which will be briefly discusssed in the course but

-- **Windows users** If you have not already installed `Python 2.7`, please do so from [here](https://www.python.org/downloads/). Then proceed with `Mac` instructions.

-- **Confirming** if all has gone well, you can now fit a neural net like so:

``` r
hello_world <- kms(mpg ~ weight + cyl, mtcars)
```

-- **Troubleshooting** If that did not work, it could be that one or another dependency failed to install. In particular, check to see whether the `R` libraries `tensorflow`, `keras`, and `reticulate` are installed; install individually as need be. If everything installed but you are seeing a lengthy error message in `Python` (complaining in part about `None` or `NoneType`), `R` is probably attempting to access `Tensorflow` via `Python 3.x`. Assuming it's installed, load the library `reticulate` and provide the path to your copy of `Python 2.7` to the `use_python()` function ([documentation](https://rstudio.github.io/reticulate/reference/use_python.html)).

Suggested Reading
=================

-   Hastie, Tibshirani, and Friedman. Chapter 11 of [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf).

-   François Chollet and JJ Allaire. [Deep Learning with R](https://www.manning.com/books/deep-learning-with-r). Manning Publications Co., 2018. ( `kerasformula` is a wrapper for `keras`, authored by Allaire; `kerasformula` helps users with many of the settings described in that work. That link has some free chapter downloads; Chollet's book, [Deep Learning with Python](http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf) contains the same content apart from the syntax.)

-   [Deep Learning](https://www.deeplearningbook.org/). 2016. Ian Goodfellow and Yoshua Bengio and Aaron Courville. MIT Press.

-   Pete Mohanty. 2018. [Analyzing rtweet Data with kerasformula](https://blogs.rstudio.com/tensorflow/posts/2018-01-24-analyzing-rtweet-data-with-kerasformula/) on *Tensorflow for R Blog*. January 18. (Note the syntax for the main function differs slightly in that, in the old version of `kms`, the user inputs a list `layers` which contains the number of `units`, `activation` function, etc. but now `units` and `activation` are no longer nested.)
