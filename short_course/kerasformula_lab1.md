Activity 1
================
Pete Mohanty
8/29/2018

**Activities**: You will be asked to complete several activities throughout the day. There are several questions that you should answer as you go. You may be asked some questions about concepts which haven't been introduced yet--that's fine, just do your best to make some notes and they'll be covered soon. Activities are best done with your neighbor but be sure to write your own code and make your own notes too. Examples are meant to run in under a minute; if they are taking much longer, stop and subset the data.

**Goal**: This is a short activity designed to get familar with the input and output of `kms` (which abbreviates `keras_model_sequential`).

**Data** Start by loading your own data or some of the Russian troll data. Here is some code that will check if it's available (in memory or on disk) and save a local copy to disk if not.

``` r
if(!exists("troll_tweets")){
  if("troll_tweets.csv" %in% dir()){
    troll_tweets <- read.csv("troll_tweets.csv")
  }else{
    troll_tweets <- read.csv("https://bit.ly/2Pz9Vvg", 
                         nrows = 25000, # comment out to save all to disk
                         stringsAsFactors = FALSE)
    write.csv(troll_tweets, file="troll_tweets.csv")
  }
}
```

**Q1** Provide a quick overview of the data frame. You may wish to use `summary`, `colnames`, or `glimpse` (`glimpse` is found in `library(dplyr)`).

**Q2** What is one variable that could be used for classification? Print a `table` of this variable.

**Q3** What is one variable that could be a regression outcome? Display a histogram (`hist`) of this variable.

**Task** Estimate a classification model using `kms` and answer the questions below about the output.

``` r
library(kerasformula)
library(ggplot2)

out <- kms(account_category ~ following + followers + language, units=3,
           data = troll_tweets, seed = 123)
```

**Q4** Look at the graph that was produced as the model estimated. Are there signs of overfitting (or underfitting)? How many epochs before validated loss stabilized?

**Q5** How many features are in the final model (what is `out$P`)?

**Q6** How does the model do out-of-sample in general? How does it do with rarer categories?

``` r
out$evaluations$acc                 # accuracy
mean(out$y_test == out$predictions) # same as above
out$confusion         # MCE abberviates 'most common error'
```

**Q7** Neural nets vary dramatically in shape and size. `kms` repeats inputs as need be based on `N_layers`. That means input can be either a vector or something of the appropriate that can be repeated. Change `Nlayers` and change another parameter like `units` and store the results of the new model as `out2`. You may wish to refer to the help (`?kms`) for details such as which inputs should be length `Nlayers` as opposed to `Nlayers - 1`. Which model fits better, `out` or `out2`? What are the trouble spots? You may wish to plot a comparison:

``` r
plot_confusion(out, out2)
```

**Q8** In general, practioners consider it important to scale the data. By default, `kerasformula` scales continuous variables on \[0, 1\]. But `kms(..., scale_continuous = "z")` standardizes (i.e., to Normal(0,1)) and `kms(..., scale_continuous = NULL)` leaves the data on its original scale. Which approach works best on this data?

``` r
plot_confusion(out, out_z, out) # can take as many as you please...
```

In any remaining time, check whether the results are stable by changing the seed.
