Lab 2: Designing Neural Nets
================
Pete Mohanty
8/29/2018

**Labs**: You will be asked to complete several activities throughout the day. There are several questions that you should answer as you go. You may be asked some questions about concepts which haven't been introduced yet--that's fine, just do your best to make some notes and they'll be covered soon. Activities are best done with your neighbor but be sure to write your own code and make your own notes too. Examples are meant to run in under a minute; if they are taking much longer, stop and subset the data.

**Goal**: Estimate several models altering the major elements of neural nets (model design).

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

Below find a neural net which achieves 98.3% accuracy out of sample.

``` r
library(kerasformula)
library(ggplot2)

out <- kms(account_category ~ following + followers + language + author + retweet, 
           units=3, 
           data = troll_tweets, seed = 123)
out$evaluations$acc
```

**Q1** Briefly describe the neural net by looking at `out$layers_overview` or `out$model`. How many layers are there? Which activation functions are used?

**Q2** Which optimizer is used? Which loss function? (`out$optimizer`, `out$loss`)

**Q3** What is the out-of-sample accuracy if you only run the model for 8 epochs?

**Q4** Estimate half a dozen or so models, each time changing one parameter, such as the number of layers, number of units per layer, the activation function(s), loss function, or optimizer. Compare out-of-sample accuracy and/or plot confusion matrices. Which are the top three?

``` r
plot_confusion(out, out2, out3) # can take as many as you please...
```

**Note** The above exercise is designed to highlight key elements of model design. K-folds cross-validation is arguably better suited to the task of model selection than CV. We will discuss kcv in a bit...
