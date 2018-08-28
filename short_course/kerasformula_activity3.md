Activity 3
================
Pete Mohanty
8/29/2018

**Activities**: You will be asked to complete several activities throughout the day. There are several questions that you should answer as you go. You may be asked some questions about concepts which haven't been introduced yet--that's fine, just do your best to make some notes and they'll be covered soon. Activities are best done with your neighbor but be sure to write your own code and make your own notes too. Examples are meant to run in under a minute; if they are taking much longer, stop and subset the data.

**Goal**: Fit a model classifying Congressional immigration votes using elements of the text as features.

**Data**: Use the data available on the course Github page from gathered with `library(Rvoteview)` (see lecture 1 for detail). You may of course choose to work your own data if its amenable.

``` r
library(kerasformula)
if("immigration_roll_call.RData" %in% dir()){
  load("immigration_roll_call.RData")
}else{
  load(url("https://bit.ly/2PtHGOG"))
}
```

The data, found in a nested structure called `rc`, comes in few a formats. The long format is most useful but is quite large so some care needs to be taken.

``` r
head(rc$votes.long)
```

              id icpsr     vname vote
    1 MP10199908 99908 RH1010873    1
    2 MH10115090 15090 RH1010873    9
    3 MH10110717 10717 RH1010873    1
    4 MH10115632 15632 RH1010873    6
    5 MH10111000 11000 RH1010873    6
    6 MH10114419 14419 RH1010873    9

``` r
dim(rc$votes.long)
```

    [1] 179241      4

The outcome is coded as follows:

``` r
rc$codes
```

    $yea
    [1] 1 2 3

    $nay
    [1] 4 5 6

    $notInLegis
    [1] 0

    $missing
    [1] 7 8 9

That means there a few ways to treat this as a classification problem (just don't forget `as.factor()`, show below, so the integer codes don't wind up being regressed on)... Run the code below to get a sense of the data...

``` r
rc$n    # obs on DV (legis x bill)
rc$m    # number of immigration bills voted on
dim(rc$vote.data)  # data about each bill
head(rc$vote.data)
```

For example, if we wanted to add congressional session to the data...

``` r
rc$votes.long$congress <- rc$vote.data$congress[match(rc$votes.long$vname, rc$vote.data$vname)]
```

Merging the whole data frames is not recommended, nor is estimating the whole thing on laptop...

``` r
seed <- 12345
set.seed(seed)
laptop_sample <- sample(nrow(rc$votes.long), 5000)
all_options <- kms(as.factor(vote) ~ id + vname + congress, 
         rc$votes.long[laptop_sample,], units=10, Nepochs = 5, 
         seed = seed, verbose = 0)
all_options$evaluations$acc
```

    [1] 0.5911824

``` r
yes_votes <- kms(vote %in% 1:3 ~ id + vname + congress, 
         rc$votes.long[laptop_sample,], units=10, Nepochs = 5, seed = seed, verbose=0)
yes_votes$evaluations$acc
```

    [1] 0.5931864

The vote descriptions are found here:

``` r
head(rc$vote.data$description)
```

    [1] "IMMIGRATION ACT OF 1990"                                                                      
    [2] "Immigration Act of 1995"                                                                      
    [3] "In the nature of a substitute."                                                               
    [4] "To provide temporary stay of deportation for certain eligible immigrants."                    
    [5] "To strike out the employment creation visa category."                                         
    [6] "To prevent the reduction of family preference immigration below the level set in current law."

``` r
rc$votes.long$description <- rc$vote.data$description[match(rc$votes.long$vname, rc$vote.data$vname)]
```

Those descriptions are now merged in to `rc$votes.long$decription`...

**Q1** Choose a couple of keywords you think may influence the outcome and estimate a model (your choice of whether the outcome is binary or multinomial). Does the addition offer improvements?

**Q2** Store your baseline formula (as a character string); call it `f`. (Do not include the additions from **Q1**.) Also, store a set of `keywords`; you may wish to use the code from lecture pasted below. Does this set of words offer improvements?

``` r
for(k in keywords)
  f <- paste0(f, " + ", "grepl(\'", k, "\', content)")
cat(f)
```

**Q3** Next, clean the bill descriptions, removing stop words and convert the words to ranks following the procedure found in lecture 3. For convenience, you may wish to use some of the code below.

``` r
tokenize <- function(txt, x, lang="english"){
  
  langs <- c("danish", "dutch", "english", 
             "finnish", "french", "german", 
             "hungarian", "italian", "norwegian", 
             "portuguese", "russian", "spanish", "swedish")

  if(length(txt) == 1){   
    
      tokens <- unlist(strsplit(tolower(txt), " "))
      keepers <- tokens[!grepl("@", tokens)]
      keepers <- keepers[!grepl("https", keepers)]
      keepers <- keepers[!grepl("#", keepers)]
      keepers <- removePunctuation(keepers)
      keepers <- keepers[nchar(keepers) > 0]
      
      w <- agrep(lang, langs) # approx grep
      
      if(length(w))
        keepers <- setdiff(keepers, stopwords(langs[w]))
      
      if(length(keepers)) return(keepers) else NA
      
  }else{
    
    out <- list()
    
    for(i in 1:length(txt))
      out[[i]] <- tokenize(txt[i], x, lang[i])
    
    return(out) 
  }
}
```

There's a bit more code in the slides but here are some more highlights...

``` r
tokens <- tokenize(rc$votes.long$description)
dictionary <- tokens %>% unlist %>% table %>% sort %>% names
ranks <- lapply(tokens, match, dictionary, nomatch=0L)
```

Now, decide how many of the words you wish to include (per observation) and estimate a new model (don't forget `pad_sequences()`).
