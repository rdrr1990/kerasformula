#' kms
#' 
#' A regression-style function call for keras_model_sequential() which uses formulas and sparse matrices. A sequential model is a linear stack of layers.
#' 
#' @param input_formula an object of class "formula" (or one coerceable to a formula): a symbolic description of the keras inputs. The outcome, y, is assumed to be categorical, e.g. "stars ~ mentions.tasty + mentions.fun".
#' @param data a data.frame.
#' @param keras_model_seq A compiled Keras sequential model. If non-NULL (NULL is the default), then bypasses the following `kms` parameters: layers, loss, metrics, and optimizer.
#' @param layers a list that creates a dense Keras model. Contains the number of units, activation type, and dropout rate. Example with three layers: layers = list(units = c(256, 128, NA), activation = c("relu", "relu", "softmax"), dropout = c(0.4, 0.3, NA)). If the final element of units is NA (default), set to the number of unique elements in y. See ?layer_dense or ?layer_dropout. 
#' @param pTraining Proportion of the data to be used for training the model;  0 =< pTraining < 1. By default, pTraining == 0.8. Other observations used only postestimation (e.g., confusion matrix).
#' @param seed seed to passed to set.seed for partitioning data. If NULL (default), automatically generated.
#' @param validation_split Portion of data to be used for validating each epoch (i.e., portion of pTraining). To be passed to keras::fit. Default == 0.2. 
#' @param Nepochs Number of epochs. To be passed to keras::fit. Default == 25.  
#' @param batch_size To be passed to keras::fit and keras::predict_classes. Default == 32. 
#' @param loss To be passed to keras::compile. Defaults to "binary_crossentropy" or "categorical_crossentropy" based on the number of distinct elements of y.
#' @param metrics To be passed to keras::compile. Default == c("accuracy").
#' @param optimizer To be passed to keras::compile. Default == "optimizer_rmsprop".
#' @param verbose 0 ot 1, to be passed to keras functions. Default == 0. 
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
                 layers = list(units = c(256, 128, NA), activation = c("relu", "relu", "softmax"),
                               dropout = c(0.4, 0.3, NA)), 
                 pTraining = 0.8, seed = NULL, validation_split = 0.2, 
                 Nepochs = 25, batch_size = 32, loss = NULL, metrics = c("accuracy"),
                 optimizer = "optimizer_rmsprop", verbose = 0, ...){
  
  if(!is_keras_available())
    stop("Please run install_keras() before using kms(). ?install_keras for options and details like setting up gpu.")
   
  if(pTraining <= 0 || pTraining >= 1) 
    stop("pTraining, the proportion of data used for training, must be between 0 and 1.")
  
  form <- as.formula(input_formula)
  if(form[[1]] != "~" || length(form) != 3) 
    stop("Expecting formula of the form\n\ny ~ x1 + x2 + x3\n\nwhere y, x1, x2... are found in (the data.frame) data.")
  
  x_tmp <- sparse.model.matrix(form, data = data, ...)
  colnames_x <- colnames(x_tmp)
  P <- ncol(x_tmp)
  N <- nrow(x_tmp)
  
  if(pTraining > 0){
    
    if(is.null(seed)) 
      seed <- sample(as.numeric(format(Sys.time(), "%OS")):10^6*as.numeric(format(Sys.time(), "%OS6")), 
                     size=1)
    set.seed(seed)
    
    split <- sample(c("train", "test"), size = N, 
                    replace = TRUE, prob = c(pTraining, 1 - pTraining))
    
    x_train <- x_tmp[split == "train", ]
    x_test <- x_tmp[split == "test", ]
    
  }else{
   x_train <- x_tmp 
  }
  remove(x_tmp)
  
  y <- eval(form[[2]], envir = data)
  n_distinct_y <- n_distinct(y)
  
  if(is.numeric(y)) 
    if((n_distinct_y == length(y) | min(y) < 0 | 
        sum(y %% 1) > length(y) * .Machine$double.eps))
      warning("y does not appear to be categorical.\n\n" )
  
  labs <- sort(unique(y))
  
  if(n_distinct_y > 2){
    
    y_cat <- to_categorical(match(y, labs) - 1) # make parameter y.categorical (??)
    # -1 for Python/C style indexing arrays, which starts at 0, must "undo"
    if(pTraining > 0){
      y_train <- y_cat[split == "train",]
      y_test <- y_cat[split == "test",]
    }else{
      y_train <- y_cat
    }
  }else{
    if(pTraining > 0){
      y_train <- y[split == "train"]
      y_test <- y[split == "test"]
    }else{
     y_train <- y_cat 
    }
  }
  remove(y_cat)
  
  if(is.null(keras_model_seq)){
    
    if(is.na(layers$units[length(layers$units)]))
      layers$units[length(layers$units)] <- max(1, ncol(y_train))
    
    Nlayers <- length(layers$units)
    
    keras_model_seq <- keras_model_sequential() 
    for(i in 1:Nlayers){
      keras_model_seq <- if(i == 1){
        layer_dense(keras_model_seq, units = layers$units[i], activation = layers$activation[i], input_shape = c(P))
      }else{
        layer_dense(keras_model_seq, units = layers$units[i], activation = layers$activation[i])
      }
      if(i != Nlayers)
        model <- layer_dropout(keras_model_seq, rate = layers$rate[i])
    }
    
    if(is.null(loss)) 
      loss <- if(n_distinct(y) == 2) "binary_crossentropy" else "categorical_crossentropy" 
    
    keras_model_seq %>% compile(
      loss = loss,
      optimizer = do.call(optimizer, args = list()),
      metrics = metrics
    )
    
  }

  history <- keras_model_seq %>% fit(x_train, y_train, 
    epochs = Nepochs, 
    batch_size = batch_size, 
    validation_split = validation_split,
    verbose = verbose
  )
  
  object <- list(history = history,
                 input_formula = form, model = keras_model_seq, 
                 loss = loss, optimizer = optimizer, metrics = metrics,
                 N = N, P = P, K = n_distinct_y,
                 y_test = y[split == "test"], y_labels = labs, colnames_x = colnames_x,
                 seed = seed, split = split)
  
  if(pTraining > 0){

    evals <- keras_model_seq %>% evaluate(x_test, y_test)
    # 1 + to get back to R/Fortran land... 
    y_fit <- labs[1 + predict_classes(keras_model_seq, x_test, batch_size = batch_size, verbose = verbose)]
    object[["evaluations"]] = evals 
    object[["predictions"]] = y_fit
    object[["confusion"]] <- confusion(object)
    
  }
  
  class(object) <- "kms_fit"
  return(object)
  
}




