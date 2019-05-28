#' kms
#' 
#' A regression-style function call for keras_model_sequential() which uses formulas and, optionally, sparse matrices. A sequential model is a linear stack of layers.
#' 
#' @param input_formula an object of class "formula" (or one coerceable to a formula): a symbolic description of the keras inputs. "mpg ~ cylinders". kms treats numeric data with more than two distinct values a continuous outcome for which a regression-style model is fit. Factors and character variables are classified; to force classification, "as.factor(cyl) ~ .". 
#' @param data a data.frame.
#' @param keras_model_seq A compiled Keras sequential model. If non-NULL (NULL is the default), then bypasses the following `kms` parameters: N_layers, units, activation, dropout, use_bias, kernel_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, loss, metrics, and optimizer.
#' @param N_layers How many layers in the model? Default == 3. Subsequent parameters (units, activation, dropout, use_bias, kernel_initializer, kernel_regularizer, bias_regularizer, and activity_regularizer) may be inputted as vectors that are of length N_layers (or N_layers - 1 for units and dropout). The length of those vectors may also be length 1 or a multiple of N_layers (or N_layers - 1 for units and dropout). 
#' @param units How many units in each layer? The final number of units will be added based on whether regression or classification is being done. Should be length 1, length N_layers - 1, or something that can be repeated to form a length N_layers - 1 vector. Default is c(256, 128).  
#' @param activation Activation function for each layer, starting with the input. Default: c("relu", "relu", "softmax"). Should be length 1, length N_layers, or something that can be repeated to form a length N_layers vector.
#' @param dropout Dropout rate for each layer, starting with the input. Not applicable to final layer. Default: c(0.4, 0.3). Should be length 1, length N_layers - 1, or something that can be repeated to form a length N_layers - 1 vector.
#' @param use_bias See ?keras::use_bias. Default: TRUE. Should be length 1, length N_layers, or something that can be repeated to form a length N_layers vector.
#' @param kernel_initializer Defaults to "glorot_uniform" for classification and "glorot_normal" for regression (but either can be inputted). Should be length 1, length N_layers, or something that can be repeated to form a length N_layers vector.
#' @param kernel_regularizer Must be precisely either "regularizer_l1", "regularizer_l2", or "regulizer_l1_l2". Default: "regularizer_l1". Should be length 1, length N_layers, or something that can be repeated to form a length N_layers vector.
#' @param bias_regularizer Must be precisely either "regularizer_l1", "regularizer_l2", or "regulizer_l1_l2". Default: "regularizer_l1". Should be length 1, length N_layers, or something that can be repeated to form a length N_layers vector.
#' @param activity_regularizer Must be precisely either "regularizer_l1", "regularizer_l2", or "regulizer_l1_l2". Default: "regularizer_l1". Should be length 1, length N_layers, or something that can be repeated to form a length N_layers vector.   
#' @param embedding If TRUE, the first layer will be an embedding with the number of output dimensions determined by `units` (so to speak, that means there will really be N_layers + 1). Note input `kernel_regularizer` is passed on as the `embedding_regularizer`. Note pad_sequences() may be used as part of the input_formula and you may wish to set scale_continuous to NULL. See ?layer_embedding.
#' @param pTraining Proportion of the data to be used for training the model;  0 =< pTraining < 1. By default, pTraining == 0.8. Other observations used only postestimation (e.g., confusion matrix).
#' @param validation_split Portion of data to be used for validating each epoch (i.e., portion of pTraining). To be passed to keras::fit. Default == 0.2. 
#' @param Nepochs Number of epochs; default == 15. To be passed to keras::fit.  
#' @param batch_size Default batch size is 32 unless emedding == TRUE in which case batch size is 1. (Smaller eases memory issues but may affect ability of optimizer to find global minimum). To be passed to several functions library(keras) functions like fit(), predict_classes(), and layer_embedding(). If embedding==TRUE, number of training obs must be a multiple of batch size. 
#' @param loss To be passed to keras::compile. Defaults to "binary_crossentropy", "categorical_crossentropy", or "mean_squared_error" based on input_formula and data.
#' @param metrics Additional metric(s) beyond the loss function to be passed to keras::compile. Defaults to "mean_absolute_error" and "mean_absolute_percentage_error" for continuous and c("accuracy") for binary/categorical (as well whether whether examples are correctly classified in one of the top five most popular categories or not if the number of categories K > 20).  
#' @param optimizer To be passed to keras::compile. Defaults to "optimizer_adam", an algorithm for first-order gradient-based optimization of stochastic objective functions introduced by Kingma and Ba (2015) here: https://arxiv.org/pdf/1412.6980v8.pdf.
#' @param optimizer_args Advanced optional arguments such as learning rate, decay, and momentum to be passed to via a named list. See library(keras) help for the arguments each optimizer accepts. For example, ?optimizer_adam accepts optimizer_adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = NULL, decay = 0, amsgrad = FALSE, clipnorm = NULL, clipvalue = NULL).
#' @param scale_continuous How to scale each non-binary column of the training data (and, if y is continuous, the outcome). The default 'scale_continuous = 'zero_one'' places each non-binary column of the training model matrix on [0, 1]; 'scale_continuous = z' standardizes; 'scale_continuous = NULL' leaves the data on its original scale.
#' @param sparse_data Default == FALSE. If TRUE, X is constructed by sparse.model.matrix() instead of model.matrix(). Recommended to improve memory usage if there are a large number of categorical variables or a few categorical variables with a large number of levels. May compromise speed, particularly if X is mostly numeric.
#' @param drop_intercept TRUE by default.     
#' @param seed Integer or list containing seed to be passed to the sources of variation: R, Python's Numpy, and Tensorflow. If seed is NULL, automatically generated. Note setting seed ensures data will be partitioned in the same way but to ensure identical results, set disable_gpu = TRUE and disable_parallel_cpu = TRUE. Wrapper for use_session_with_seed(), which is to be called before compiling by the user if a compiled Keras model is passed into kms. See also see https://stackoverflow.com/questions/42022950/. 
#' @param verbose Default == 1. Setting to 0 disables progress bar and epoch-by-epoch plots (disabling them is recommended for knitting RMarkdowns if X11 not installed).
#' @param ... Additional parameters to be passsed to Matrix::sparse.model.matrix.
#' @return kms_fit object. A list containing model, predictions, evaluations, as well as other details like how the data were split into testing and training. To extract or save weights, see https://tensorflow.rstudio.com/keras/reference/save_model_hdf5.html 
#' @examples
#' if(is_keras_available()){
#' 
#'  mtcars$make <- unlist(lapply(strsplit(rownames(mtcars), " "), function(tokens) tokens[1]))
#'  company <- kms(make ~ ., mtcars, Nepochs = 1, verbose=0)
#'  # out of sample accuracy
#'  pCorrect <- mean(company$y_test == company$predictions)
#'  pCorrect
#'  company$confusion
#'  # plot(history$company) # helps pick Nepochs
#'  # below
#'  # find the default settings for layers
#'  company <- kms(make ~ ., mtcars,
#'                 units = c(256, 128), 
#'                 activation = c("relu", "relu", "softmax"),
#'                 dropout = 0.4,
#'                 use_bias = TRUE,
#'                 kernel_initializer = NULL,
#'                 kernel_regularizer = "regularizer_l1",
#'                 bias_regularizer = "regularizer_l1",
#'                 activity_regularizer = "regularizer_l1",
#'                 Nepochs = 1, verbose=0
#'                 )
#'                 
#'  # example with learning rate               
#'  
#'  company <- kms(make ~ ., mtcars, units = c(10,10), optimizer_args = list(lr = 0.03))                             
#'  # see help file for each optimizer for advanced options.
#'  # ?optimizer_adam to see options for default optimizer
#'  
#'                                
#'  # ?predict.kms_fit to see how to predict on newdata
#' }else{
#'    cat("Please run install_keras() before using kms(). ?install_keras for options like gpu.")
#' }
#'  
#' @author Pete Mohanty
#' @importFrom keras to_categorical keras_model_sequential layer_dense layer_dropout compile fit evaluate predict_classes is_keras_available get_weights save_model_hdf5 save_model_weights_hdf5 use_session_with_seed layer_embedding layer_flatten
#' @importFrom Matrix sparse.model.matrix
#' @importFrom stats as.formula cor formula model.matrix predict sd
#' @importFrom dplyr n_distinct %>%
#' @importFrom ggplot2 ggplot aes geom_histogram ggtitle
#' @importFrom utils object.size 
#' 
#' @export
kms <- function(input_formula, data, keras_model_seq = NULL, 
                N_layers = 3,
                units = c(256, 128), 
                activation = c("relu", "relu", "softmax"),
                dropout = 0.4,
                use_bias = TRUE,
                kernel_initializer = NULL,
                kernel_regularizer = "regularizer_l1",
                bias_regularizer = "regularizer_l1",
                activity_regularizer = "regularizer_l1",
                embedding = FALSE,
                pTraining = 0.8, validation_split = 0.2, Nepochs = 15, batch_size = NULL, 
                loss = NULL, metrics = NULL, 
                optimizer = "optimizer_adam",
                optimizer_args = list(), # named list based on e.g. optimizer_adam
                scale_continuous = "zero_one", drop_intercept=TRUE,
                sparse_data = FALSE,
                seed = list(seed = NULL, disable_gpu=FALSE, disable_parallel_cpu = FALSE), 
                verbose = 1, ...){
  
  if(!is_keras_available())
    stop("Please run install_keras() before using kms(). ?install_keras for details on options like conda or gpu. Also helpful:\n\nhttps://tensorflow.rstudio.com/tensorflow/articles/installation.html")
   
  # if(!is.null(keras_model_seq) & (n_distinct(lapply(layers, length)) != 1))
  #  warning("\nThe number of units, activation functions, and dropout rates is not the same. Note the final number of units will be automatically determined by the data. Valid example:\n\N_layers = list(units = c(256, 128, NA),\n\t\tactivation = c('relu', 'relu', 'softmax'),\n\t\tdropout = c(0.4, 0.3, NA))")

  if(pTraining <= 0 || pTraining > 1) 
    stop("pTraining, the proportion of data used for training, must be between 0 and 1. See also help(\"predict.kms_fit\").")
  
  
  form <- formula(input_formula, data = data)
  if(form[[1]] != "~" || length(form) != 3) 
    stop("Expecting formula of the form\n\ny ~ x1 + x2 + x3\n\nwhere y, x1, x2... are found in (the data.frame) data.")
  
  data <- as.data.frame(data)
  X <- if(sparse_data) sparse.model.matrix(form, data = data, row.names = FALSE, ...) else model.matrix(form, data = data, row.names = FALSE, ...) 
  
  if(verbose > 0)
    if(as.numeric(unlist(strsplit(format(object.size(X), unit="Mb"), " "))[1]))
      message("Model Matrix size: ", format(object.size(X), unit="Gb"), "\n")
  
  if(drop_intercept)
    X <- X[,-1]
  colnames_x <- colnames(X)
  P <- ncol(X)
  N <- nrow(X)
  
  batch_size <- if(embedding) 1 else 32 # handles nuissance ... 
  
  if(verbose > 0) message("N: ", N, ", P: ", P, "\n\n")
    
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
    
      seed_list$seed <- sample(2^30, 1)
      # py Seed must be between 0 and 2**32 - 1 but avoiding R integer coercion issues with larger than 2^30    
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
    if(verbose > 0) message("R seed set to ", seed_list$seed, " but for full reproducibility it is necessary to set the seed for the graph on the Python side too. kms cannot do this automatically when a compiled model is passed as argument. In R, call use_session_with_seed() before compiling. For detail:\n\nhttps://github.com/rdrr1990/kerasformula/blob/master/examples/kms_replication.md")
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
  
  if(is.numeric(y) & n_distinct_y > 2){
      
      if(verbose > 0) 
        message("y does not appear to be categorical; proceeding with regression. To instead do classification, stop and do something like\n\n out <- kms(as.factor(y) ~ x1 + x2, ...)" )
      
      y_type <- "continuous"
      labs <- NULL
      
      if(is.null(loss))
        loss <- "mean_squared_error"
      
      if(is.null(metrics))
        metrics <- c("mean_absolute_error", "mean_absolute_percentage_error")
      
  }else{
        
      labs <- sort(unique(y))
      y_type <- if(n_distinct_y > 2) "multinomial" else "binary"
      if(verbose > 0) 
        message("y appears categorical. Proceeding with ", y_type, " classification.\n" )
      
      
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
    
    # binary case
    if(is.character(y))
      y <- as.factor(y)
    
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
    
    if(length(cols_to_scale) > 0){
      
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
  
  layers <- NULL
  
  if(is.null(keras_model_seq)){
    
    if(is.null(kernel_initializer))
      kernel_initializer <- if(y_type == "continuous") "glorot_normal" else "glorot_uniform"
    
    layers <- data.frame(row.names = paste0("layer", 1:N_layers))
    layers$use_bias <- use_bias
    layers$kernel_initializer <- kernel_initializer
    layers$kernel_regularizer <- kernel_regularizer
    layers$bias_regularizer <- bias_regularizer
    layers$activity_regularizer <- activity_regularizer
    layers$units <- 1
    layers$units[N_layers] <- max(1, ncol(y_train))
    layers$units[-N_layers] <- units
    layers$activation <- activation
    if(y_type == "continuous")
      layers$activation[N_layers] <- "linear"
    layers$dropout <- 0
    layers$dropout[-N_layers] <- dropout 
    layers$embedding <- NA
    layers$embedding[1] <- embedding

    if(verbose > 0) print(layers)
  
    keras_model_seq <- keras_model_sequential() 
    
    if(embedding){
      
      layer_embedding(keras_model_seq, 
                      input_dim = (max(X_train) + 1),
                      output_dim = layers$units[1],
                      input_length = P,
                      embeddings_regularizer = penalty(layers$kernel_regularizer[1]),
                      activity_regularizer = penalty(layers$activity_regularizer[1]),
                      batch_size = batch_size
                      )
      layer_flatten(keras_model_seq)
      layer_dropout(keras_model_seq, layers$dropout[1])
      
    }else{
      
        layer_dense(keras_model_seq, 
                    input_shape = c(P), 
                    units = layers$units[1], 
                    activation = layers$activation[1], 
                    use_bias = layers$use_bias[1], 
                    kernel_initializer = layers$kernel_initializer[1],
                    kernel_regularizer = penalty(layers$kernel_regularizer[1]),
                    bias_regularizer = penalty(layers$bias_regularizer[1]),
                    activity_regularizer = penalty(layers$activity_regularizer[1])) 
    }
    
    start <- if(embedding) 2 else 1
    
    for(i in start:N_layers){
      
        layer_dense(keras_model_seq, 
                    units = layers$units[i], 
                    activation = layers$activation[i], 
                    use_bias = layers$use_bias[i], 
                    kernel_initializer = layers$kernel_initializer[i],
                    kernel_regularizer = penalty(layers$kernel_regularizer[i]),
                    bias_regularizer = penalty(layers$bias_regularizer[i]),
                    activity_regularizer = penalty(layers$activity_regularizer[i])) 

       if(!is.na(layers$dropout[i]))
          if(layers$dropout[i] > 0)
            layer_dropout(keras_model_seq, rate = layers$dropout[i], seed = seed_list$seed)
    }
    
    keras_model_seq %>% compile(
      loss = loss,
      optimizer = do.call(optimizer, args = optimizer_args),
      metrics = metrics
    )
    
  }
  
  if(verbose > 0) summary(keras_model_seq)

  history <- fit(keras_model_seq, X_train, y_train, 
    epochs = Nepochs, 
    batch_size = batch_size, 
    validation_split = validation_split,
    verbose = verbose, 
    view_metrics = verbose > 0
  )
  
  object <- list(history = history,
                 input_formula = form, model = keras_model_seq, layers_overview = layers,
                 loss = loss, optimizer = optimizer, metrics = metrics,
                 N = N, P = P, K = n_distinct_y,
                 y_test = if(pTraining == 1) NULL else y[split == "test"],
                 # avoid y_test <- y_cat[split == "test", ]
                 y_type = y_type, sparse_data = sparse_data,
                 y_labels = labs, colnames_x = colnames_x, 
                 seed = seed, split = split, 
                 train_scale = train_scale)

  if(pTraining < 1){
    
    object[["evaluations"]] <- evaluate(keras_model_seq, X_test, y_test, batch_size = batch_size)
    
    if(y_type == "continuous"){
      
      y_fit <- predict(keras_model_seq, X_test, 
                       batch_size = batch_size, verbose = verbose)
      
      object[["MSE_predictions"]] <- mean((y_fit - y_test)^2)
      object[["MAE_predictions"]] <- mean(abs(y_fit - y_test))
      object[["R2_predictions"]] <- cor(y_fit, y_test)^2
      object[["cor_kendals"]] <- cor(y_fit, y_test, method="kendal") # guard against broken clock predictions
      object[["predictions"]] <- y_fit
      
      est <- data.frame(y = c(y_test, y_fit),
                        type = c(rep("y_test", length(y_test)), rep("predictions", length(y_fit))))
      if(verbose > 0) 
        ggplot(est, aes(x=~y, fill=~type)) + geom_histogram() + ggtitle("Holdout Data vs. Predictions")
      
      
      
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

penalty <- function(regularization_type, alpha = 0.01){
  if(is.null(regularization_type)) NULL else do.call(regularization_type, list(alpha))
}


