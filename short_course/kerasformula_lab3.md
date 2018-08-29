Lab 3: Triage against Overfitting
================
Pete Mohanty
8/29/2018

**Labs**: You will be asked to complete several activities throughout the day. There are several questions that you should answer as you go. You may be asked some questions about concepts which haven't been introduced yet--that's fine, just do your best to make some notes and they'll be covered soon. Activities are best done with your neighbor but be sure to write your own code and make your own notes too. Examples are meant to run in under a minute; if they are taking much longer, stop and subset the data.

**Goal**: Manipulate the major available parameters that are meant to prevent overfitting.

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
tweets <- troll_tweets
tweets$kind <- tweets$account_category
```

Below find a neural net which achieves 98.3% accuracy out of sample with a small number of units (i.e., this is a model which does not appear to be overfitting).

``` r
library(kerasformula)
library(ggplot2)

out <- kms(account_category ~ following + followers + language + author + retweet, 
           units=3, 
           data = tweets, seed = 123)
```

To look at out-of-sample

``` r
out$evaluations$acc
```

To see the training/validation history and see whether the model is overfitting, underfitting, or striking an nice balance:

``` r
out$history$metrics$acc
out$history$metrics$val_acc
```

**Task** Start by estimating several models which manipulate the major levers against overfitting--portion of the data used for training, dropout rate, regularization. For each, make a note about what change you expect in terms of underfitting vs. overfitting.

**Task** Choose the top three models and perform k folds cross validation (ideally, this would be done on a fresh batch of data but let's not worry about that now.) Here is some code to get started ...

``` r
N_folds <- 5
folds <- sample(N_folds, nrow(tweets), replace=TRUE)
m1 <- list()

for(f in 1:N_folds){
  
  train <- paste0("train_f", f)
  m1[[train]] <- kms(account_category ~ following + followers, 
                     tweets[folds != f, ], verbose=0,
                     pTraining=1, validation_split=0,
                     units=3, Nepochs=8, seed=f)
  
  test <- paste0("test_f", f) 
  m1[[test]] <- predict(m1[[train]], tweets[folds == f, ])
}
```

Here is some more code that should help clean up the estimates once all three are there...

``` r
comparison <- data.frame(model = c(rep("model1", N_folds), 
                                   rep("model2", N_folds), 
                                   rep("model3", N_folds)),
                         fold = c(1:N_folds, 1:N_folds, 1:N_folds))

comparison$acc <- NULL

for(f in 1:N_folds){
  
  comparison$accuracy[f] <- m1[[paste0("test_f", f)]][["accuracy"]]
  comparison$accuracy[f + N_folds] <- m2[[paste0("test_f", f)]][["accuracy"]]
  comparison$accuracy[f + 2*N_folds] <- m3[[paste0("test_f", f)]][["accuracy"]]
  
}

ggplot(comparison) + aes(x=fold, y=accuracy, col=model) + geom_point() + theme_minimal() +
  labs(title="Model Comparison", subtitle="Out-of-Sample Fit Across k=5 Folds")
```
