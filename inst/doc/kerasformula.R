## ---- echo = FALSE, messsage=FALSE, warning=FALSE------------------------
library(knitr)
opts_chunk$set(comment = "", message = FALSE, warning = FALSE)

## ---- eval = FALSE-------------------------------------------------------
#  install.packages("keras")
#  library(keras)
#  install_keras() # see https://keras.rstudio.com/ for details.

## ---- eval = FALSE-------------------------------------------------------
#  max_features <- 5000 # 5,000 words (ranked by popularity) found in movie reviews
#  maxlen <- 50  # Cut texts after 50 words (among top max_features most common words)
#  Nsample <- 1000
#  
#  cat('Loading data...\n')
#  imdb <- keras::dataset_imdb(num_words = max_features)
#  imdb_df <- as.data.frame(cbind(c(imdb$train$y, imdb$test$y),
#                                 pad_sequences(c(imdb$train$x, imdb$test$x))))
#  
#  set.seed(2017)   # can also set kms(..., seed = 2017)
#  
#  demo_sample <- sample(nrow(imdb_df), Nsample)
#  P <- ncol(imdb_df) - 1
#  colnames(imdb_df) <- c("y", paste0("x", 1:P))
#  
#  out_dense <- kms("y ~ .", data = imdb_df[demo_sample, ], Nepochs = 10,
#                   scale=NULL) # scale=NULL means leave data on original scale
#  
#  
#  plot(out_dense$history)  # incredibly useful
#  # choose Nepochs to maximize out of sample accuracy
#  
#  out_dense$confusion

## ---- eval=FALSE---------------------------------------------------------
#  cat('Test accuracy:', out_dense$evaluations$acc, "\n")

## ---- eval = FALSE-------------------------------------------------------
#  out_dense <- kms("y ~ .", data = imdb_df[demo_sample, ], Nepochs = 10, seed=123, scale=NULL,
#                   layers = list(units = c(512, 256, 128, NA),
#                                 activation = c("softmax", "relu", "relu", "softmax"),
#                                 dropout = c(0.75, 0.4, 0.3, NA),
#                                 use_bias = TRUE,
#                                 kernel_initializer = NULL,
#                                 kernel_regularizer = "regularizer_l1",
#                                 bias_regularizer = "regularizer_l1",
#                                 activity_regularizer = "regularizer_l1"
#                                 ))
#  out_dense$confusion

## ---- eval = FALSE-------------------------------------------------------
#  cat('Test accuracy:', out_dense$evaluations$acc, "\n")

## ---- eval = FALSE-------------------------------------------------------
#  use_session_with_seed(12345)
#  k <- keras_model_sequential()
#  k %>%
#    layer_embedding(input_dim = max_features, output_dim = 128) %>%
#    layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>%
#    layer_dense(units = 1, activation = 'sigmoid')
#  
#  k %>% compile(
#    loss = 'binary_crossentropy',
#    optimizer = 'adam',
#    metrics = c('accuracy')
#  )
#  out_lstm <- kms("y ~ .", imdb_df[demo_sample, ],
#                  keras_model_seq = k, Nepochs = 10, seed = 12345, scale = NULL)
#  out_lstm$confusion

## ---- eval=FALSE---------------------------------------------------------
#  cat('Test accuracy:', out_lstm$evaluations$acc, "\n")

## ---- eval=FALSE---------------------------------------------------------
#  
#  use_session_with_seed(12345)
#  
#  keras_model_sequential() %>%
#  
#    layer_embedding(input_dim = max_features, output_dim = 128) %>%
#  
#      layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>%
#  
#        layer_dense(units = 1, activation = 'sigmoid') %>%
#  
#          compile(loss = 'binary_crossentropy',
#                  optimizer = 'adam', metrics = c('accuracy')) %>%
#  
#              kms(input_formula = "y ~ .", data = imdb_df[demo_sample, ],
#                  Nepochs = 10, seed = 12345, scale = NULL) ->
#    out_lstm
#  
#  plot(out_lstm$history)

