#' kms
#' 
#' A regression-style function call for keras_model_sequential() which uses formulas and sparse matrices. A sequential model is a linear stack of layers.
#' 
#' @param input_formula an object of class "formula" (or one coerceable to a formula): a symbolic description of the keras inputs. "stars ~ mentions.tasty + mentions.fun". kms treats numeric data a continuous outcome for which a regression-style model is fit. To do classification,
#' @param data a data.frame.
#' @param keras_model_seq A compiled Keras sequential model. If non-NULL (NULL is the default), then bypasses the following `kms` parameters: layers, loss, metrics, and optimizer.
#' @param layers a list that creates a dense Keras model. Contains the number of units, activation type, and dropout rate. For classification, defaults to three layers: layers = list(units = c(256, 128, NA), activation = c("relu", "relu", "softmax"), dropout = c(0.4, 0.3, NA)). If the final element of units is NA (default), set to the number of unique elements in y. kms defines the number of layers as the length of the vector of activations. Inputs that appear once are repeated Nlayer times. See ?layer_dense or ?layer_dropout. For regression, activation = c("relu", "softmax", "linear"). For penalty terms, options must be precisely either "regularizer_l1", "regularizer_l2", or "regulizer_l1_l2". Also, "kernel_initalizer" defaults to "glorot_uniform" for classification and "glorot_normal" for regression (but either can be inputted with quotes).  
#' @param pTraining Proportion of the data to be used for training the model;  0 =< pTraining < 1. By default, pTraining == 0.8. Other observations used only postestimation (e.g., confusion matrix).
#' @param seed Integer or list containing seed to be passed to the sources of variation: R, Python's Numpy, and Tensorflow. If seed is NULL, automatically generated. Note setting seed ensures data will be partitioned in the same way but to ensure identical results, set disable_gpu = TRUE and disable_parallel_cpu = TRUE. Wrapper for use_session_with_seed(), which is to be called before compiling by the user if a compiled Keras model is passed into kms. See also see https://stackoverflow.com/questions/42022950/. 
#' @param validation_split Portion of data to be used for validating each epoch (i.e., portion of pTraining). To be passed to keras::fit. Default == 0.2. 
#' @param Nepochs Number of epochs; default == 15. To be passed to keras::fit.  
#' @param batch_size To be passed to keras::fit and keras::predict_classes. Default == 32. 
#' @param loss To be passed to keras::compile. Defaults to "binary_crossentropy", "categorical_crossentropy", or "mean_squared_error" based on input_formula and data.
#' @param metrics Additional metric(s) beyond the loss function to be passed to keras::compile. Defaults to "mean_absolute_error" and "mean_absolute_percentage_error" for continuous and c("accuracy") for binary/categorical (as well whether whether examples are correctly classified in one of the top five most popular categories or not if the number of categories K > 20).  
#' @param optimizer To be passed to keras::compile. Defaults to "optimizer_adam", an algorithm for first-order gradient-based optimization of stochastic objective functions introduced by Kingma and Ba (2015) here: https://arxiv.org/pdf/1412.6980v8.pdf.
#' @param scale_continuous Function to scale each non-binary column of the training data (and, if y is continuous, the outcome). The default 'scale_continuous = zero_one' places each non-binary column of the training model matrix on [0, 1]; 'scale_continuous = z' standardizes; 'scale_continuous = NULL' leaves the data on its original scale.
#' @param drop_intercept TRUE by default, may be required by implementation features.     
#' @param verbose 0 ot 1, to be passed to keras functions. Default == 1. 
#' @param ... Additional parameters to be passsed to Matrix::sparse.model.matrix.
#' @return kms_fit object. A list containing model, predictions, evaluations, as well as other details like how the data were split into testing and training.
#' @examples
#' if(is_keras_available()){
#' 
#'  mtcars$make <- unlist(lapply(strsplit(rownames(mtcars), " "), function(tokens) tokens[1]))
#'  company <- kms(make ~ ., mtcars, Nepochs = 1)
#'  # out of sample accuracy
#'  pCorrect <- mean(company$y_test == company$predictions)
#'  pCorrect
#'  company$confusion
#'  # plot(history$company) # helps pick Nepochs
#'  company <- kms(make ~ ., mtcars, Nepochs = 1, seed = 2018,
#'                layers = list(units = c(11, 9, NA), activation = c("relu", "relu", "softmax"),
#'                dropout = c(0.4, 0.3, NA)))
#'  # ?predict.kms_fit to see how to predict on newdata
#' }else{
#'    cat("Please run install_keras() before using kms(). ?install_keras for options like gpu.")
#' }
#'  
#' @author Pete Mohanty
#' @importFrom keras to_categorical keras_model_sequential layer_dense layer_dropout compile fit evaluate predict_classes is_keras_available
#' @importFrom Matrix sparse.model.matrix
#' @importFrom stats as.formula
#' @importFrom dplyr n_distinct %>%
#' 
#' @export
kms <- function(input_formula, data, keras_model_seq = NULL, 
                 layers = list(units = c(256, 128, NA), 
                               activation = c("relu", "relu", "softmax"),
                               dropout = c(0.4, 0.3, NA),
                               use_bias = TRUE,
                               kernel_initializer = NULL,
                               kernel_regularizer = "regularizer_l1",
                               bias_regularizer = "regularizer_l1",
                               activity_regularizer = "regularizer_l1"
                               ), 
                 pTraining = 0.8, validation_split = 0.2, Nepochs = 15, batch_size = 32, 
                 loss = NULL, metrics = NULL, optimizer = "optimizer_adam",
                 scale_continuous = "zero_one", drop_intercept=TRUE,
                 seed = list(seed = NULL, disable_gpu=FALSE, disable_parallel_cpu = FALSE), 
                 verbose = 1, ...){
  
  if(!is_keras_available())
    stop("Please run install_keras() before using kms(). ?install_keras for details on options like conda or gpu. Also helpful:\n\nhttps://tensorflow.rstudio.com/tensorflow/articles/installation.html")
   
  # if(!is.null(keras_model_seq) & (n_distinct(lapply(layers, length)) != 1))
  #  warning("\nThe number of units, activation functions, and dropout rates is not the same. Note the final number of units will be automatically determined by the data. Valid example:\n\nlayers = list(units = c(256, 128, NA),\n\t\tactivation = c('relu', 'relu', 'softmax'),\n\t\tdropout = c(0.4, 0.3, NA))")

  if(pTraining <= 0 || pTraining > 1) 
    stop("pTraining, the proportion of data used for training, must be between 0 and 1. See also help(\"predict.kms_fit\").")
  
  form <- formula(input_formula, data = data)
  if(form[[1]] != "~" || length(form) != 3) 
    stop("Expecting formula of the form\n\ny ~ x1 + x2 + x3\n\nwhere y, x1, x2... are found in (the data.frame) data.")
  
  data <- as.data.frame(data)
  X <- sparse.model.matrix(form, data = data, row.names = FALSE, ...)
  if(drop_intercept)
    X <- X[,-1]
  colnames_x <- colnames(X)
  P <- ncol(X)
  N <- nrow(X)
    
  if(!is.list(seed)){
    seed_list <- list(seed = NULL, disable_gpu=FALSE, disable_parallel_cpu = FALSE)
    if(is.numeric(seed))
      seed_list$seed <- seed
  }else{
      seed_list <- seed 
      # allow user to pass in integer which controls software but not hardware parameters too
      # see https://github.com/rdrr1990/kerasformula/blob/master/examples/kms_replication.md
    } 
  if(is.null(seed_list$seed)){
    
      a <- as.numeric(format(Sys.time(), "%OS"))
      b <- 10^6*as.numeric(format(Sys.time(), "%OS6"))
      seed_list$seed <- sample(a:b, size=1)
      
  } 
  
  if(is.null(keras_model_seq)){
    
    use_session_with_seed(seed = seed_list$seed, 
                          disable_gpu = seed_list$disable_gpu, 
                          disable_parallel_cpu = seed_list$disable_parallel_cpu, 
                          quiet = (verbose == 0)) 
    # calls set.seed() and Python equivalents...
    # seed intended to keep training / validation / test splits constant. 
    # additional parameters intended to remove simulation error
    # and ensure exact results...
    # see https://github.com/rdrr1990/kerasformula/blob/master/examples/kms_replication.md
    
  }else{
    set.seed(seed_list$seed)
    message("R seed set to ", seed_list$seed, " but for full reproducibility it is necessary to set the seed for the graph on the Python side too. kms cannot do this automatically when a compiled model is passed as argument. In R, call use_session_with_seed() before compiling. For detail:\n\nhttps://github.com/rdrr1990/kerasformula/blob/master/examples/kms_replication.md")
  }
    
  if(pTraining > 0){
    
    split <- sample(c("train", "test"), size = N, 
                    replace = TRUE, prob = c(pTraining, 1 - pTraining))
    
    X_train <- X[split == "train", ]
    X_test <- X[split == "test", ]
    
  }else{
   X_train <- X 
  }
  
  remove(X)
  
  y <- eval(form[[2]], envir = data)
  n_distinct_y <- n_distinct(y)
  
  if(is.numeric(y)){
      
      if(verbose > 0) 
        message("y does not appear to be categorical; proceeding with regression. To instead do classification, stop and do something like\n\n out <- kms(as.factor(y) ~ x1 + x2, ...)" )
      
      y_type <- "continuous"
      labs <- NULL
      
      if(is.null(loss))
        loss <- "mean_squared_error"
      
      if(is.null(metrics))
        metrics <- c("mean_absolute_error", "mean_absolute_percentage_error")
      
  }else{
        
    
      if(verbose > 0) 
        message("y appears categorical. Proceeding with classification.\n" )
      labs <- sort(unique(y))
      y_type <- if(n_distinct_y > 2) "multinomial" else "binary"
      
      if(is.null(loss)) 
        loss <- if(n_distinct_y == 2) "binary_crossentropy" else "categorical_crossentropy" 
      
      if(is.null(metrics))
        metrics <- c("accuracy")
      
      if(n_distinct_y > 20)
        metrics <- c(metrics, "top_k_categorical_accuracy")
  }

  
  if(y_type == "multinomial"){
    
    y_cat <- to_categorical(match(y, labs) - 1) # make parameter y.categorical (??)
    # match() - 1 for Python/C style indexing arrays, which starts at 0, must "undo"
    if(pTraining < 1){
      y_train <- y_cat[split == "train",]
      y_test <- y_cat[split == "test",]
    }else{
      y_train <- y_cat
    }
    remove(y_cat)
  }else{
    
    if(pTraining < 1){
      y_train <- as.numeric(y)[split == "train"]
      y_test <- as.numeric(y)[split == "test"]
    }else{
     y_train <- as.numeric(y) 
    }
    
  }
  
  train_scale <- NULL
  if(!is.null(scale_continuous)){
    
    cols_to_scale <- which(apply(X_train, 2, n_distinct) > 2)
    
    # information about scale of training data
    
    train_scale <- list()
    train_scale[["cols_to_scale"]] <- cols_to_scale
    train_scale[["X"]] <- matrix(nrow=2, ncol=length(cols_to_scale))
    colnames(train_scale$X) <- colnames_x[cols_to_scale]
    
    train_scale[["scale"]] <- scale_continuous
    
    if(scale_continuous == "zero_one"){
      
      stat1 <- min
      stat2 <- max
      transformation <- zero_one
      rownames(train_scale$X) <- c("min", "max")
      
    }else{
      
      stat1 <- mean
      stat2 <- sd
      rownames(train_scale$X) <- c("mean", "sd")
      transformation <- z
      
    }
    
    for(i in 1:length(cols_to_scale)){
      
      train_scale$X[1, i] <- stat1(X_train[ , cols_to_scale[i]])
      train_scale$X[2, i] <- stat2(X_train[ , cols_to_scale[i]])
      X_train[ , cols_to_scale[i]] <- transformation(X_train[ , cols_to_scale[i]])
    
      if(pTraining < 1){
        X_test[ , cols_to_scale[i]] <- transformation(X_test[ , cols_to_scale[i]], 
                                                      train_scale$X[1, i], 
                                                      train_scale$X[2, i])
        # place test data on scale observed in training data
      }
      
    }
    
    if(y_type == "continuous"){
      
      train_scale[["y"]] <- matrix(nrow=2, ncol=1)
      rownames(train_scale$y) <- rownames(train_scale$X) 
      train_scale$y[1] <- stat1(y_train)
      train_scale$y[2] <- stat2(y_train)
      y_train <- transformation(y_train)
      
      if(pTraining < 1)
        y_test <- transformation(y_test, train_scale$y[1], train_scale$y[2])
    }
    
  }
  
  if(is.null(keras_model_seq)){
    
    Nlayers <- length(layers$activation)
    
    if(is.null(layers$kernel_initializer))
      layers$kernel_initializer <- if(y_type == "continuous") "glorot_normal" else "glorot_uniform"
    
    for(i in 1:length(layers)){
      if(length(layers[[i]]) == 1){
        layers[[i]] <- rep(layers[[i]], Nlayers)
      }
    }
    
    if(is.na(layers$units[Nlayers]))
      layers$units[Nlayers] <- max(1, ncol(y_train))
    
    if(y_type == "continuous")
      layers$activation[Nlayers] <- "linear"
    
    penalty <- function(reg_type){
      if(is.null(reg_type)) NULL else do.call(reg_type, list(0.01))
    }
    
    keras_model_seq <- keras_model_sequential() 
    for(i in 1:Nlayers){
      keras_model_seq <- if(i == 1){
        layer_dense(keras_model_seq, units = layers$units[i], 
                    activation = layers$activation[i], input_shape = c(P), 
                    use_bias = layers$use_bias[i], 
                    kernel_initializer = layers$kernel_initializer[i],
                    kernel_regularizer = penalty(layers$kernel_regularizer[i]),
                    bias_regularizer = penalty(layers$bias_regularizer[i]),
                    activity_regularizer = penalty(layers$activity_regularizer[i])
                    )
      }else{
        layer_dense(keras_model_seq, units = layers$units[i], activation = layers$activation[i], use_bias = layers$use_bias[i])
      }
      if(i != Nlayers)
        model <- layer_dropout(keras_model_seq, rate = layers$rate[i], seed = seed_list$seed)
    }
    
    keras_model_seq %>% compile(
      loss = loss,
      optimizer = do.call(optimizer, args = list()),
      metrics = metrics
    )
    
  }

  history <- fit(keras_model_seq, X_train, y_train, 
    epochs = Nepochs, 
    batch_size = batch_size, 
    validation_split = validation_split,
    verbose = verbose, 
    view_metrics = verbose > 0
  )
  
  object <- list(history = history,
                 input_formula = form, model = keras_model_seq, 
                 loss = loss, optimizer = optimizer, metrics = metrics,
                 N = N, P = P, K = n_distinct_y,
                 y_test = if(pTraining == 1) NULL else y[split == "test"],
                 # avoid y_test <- y_cat[split == "test", ]
                 y_type = y_type,
                 y_labels = labs, colnames_x = colnames_x, 
                 seed = seed, split = split, 
                 train_scale = train_scale)

  if(pTraining < 1){
    
    object[["evaluations"]] <- evaluate(keras_model_seq, X_test, y_test)
    
    if(y_type == "continuous"){
      
      y_fit <- predict(keras_model_seq, X_test, 
                       batch_size = batch_size, verbose = verbose)
      
      object[["MSE_predictions"]] <- mean((y_fit - y_test)^2)
      object[["MAE_predictions"]] <- mean(abs(y_fit - y_test))
      object[["R2_predictions"]] <- cor(y_fit, y_test)^2
      object[["cor_kendals"]] <- cor(y_fit, y_test, method="kendal") # guard against broken clock predictions
      object[["predictions"]] <- y_fit
      
    }else{

      y_fit <- labs[1 + predict_classes(keras_model_seq, X_test, 
                                        batch_size = batch_size, verbose = verbose)]
      # indices + 1 to get back to R/Fortran land...
      
      object[["predictions"]] <- y_fit
      object[["confusion"]] <- confusion(object)
      
    }    
    
  }
  
  class(object) <- "kms_fit"
  return(object)
  
}

##
## helper functions
## 

zero_one <- function(x, x_min = NULL, x_max = NULL){

# places on [0, 1], optionally based on x_min, x_max of the training data
    
  if(is.null(x_min) | is.null(x_max)){
    return((x - min(x, na.rm = TRUE))/(max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
  }else{
    return((x - x_min)/(x_max - x_min))
  }
  
}

z <- function(x, x_mean = NULL, x_sd = NULL){
  
# standardizes, optionally based on training data
  
  if(is.null(x_mean) | is.null(x_sd)){
    (x - mean(x, na.rm=TRUE))/sd(x, na.rm=TRUE)
  }else{
    (x - x_mean)/x_sd
  }
  
}



