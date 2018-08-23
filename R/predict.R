#' predict.kms_fit
#' 
#' predict function for kms_fit object. Places test data on same scale that the training data were by kms(). Wrapper for keras::predict_classes(). Creates a sparse model matrix with the same columns as the training data, some of which may be 0.
#' 
#' @param object output from kms()
#' @param newdata new data. Performs merge so that X_test has the same columns as the object created by kms_fit using the user-provided input formula. y_test is also generated from that formula.
#' @param batch_size To be passed to keras::predict_classes. Default == 32.
#' @param verbose 0 ot 1, to be passed to keras::predict_classes. Default == 0.
#' @param ... additional parameters to build the sparse matrix X_test.
#' @return list containing predictions, y_test, confusion matrix.
#' @examples 
#' if(is_keras_available()){
#' 
#'  mtcars$make <- unlist(lapply(strsplit(rownames(mtcars), " "), function(tokens) tokens[1]))
#'  company <- kms(make ~ ., mtcars[3:32, ], Nepochs = 2, verbose=0)
#'  forecast <- predict(company, mtcars[1:2, ])
#'  forecast$confusion
#'  
#'  # example where y_test is unavailable
#'  
#'  trained <- kms(log(mpg) ~ ., mtcars[4:32,], Nepochs=1, verbose=0)
#'  X_test <- subset(mtcars[1:3,], select = -mpg)  
#'  predictions <- predict(trained, X_test)
#'  
#' }else{
#'    cat("Please run install_keras() before using kms(). ?install_keras for options like gpu.")
#' }
#' @author Pete Mohanty
#' @importFrom Matrix Matrix
#' @method predict kms_fit
#' @export
predict.kms_fit <- function (object, newdata, batch_size = 32, verbose=0, ...) {
  
  if (class(object) != "kms_fit") {
    warning("Object not of class 'kms_fit'")
    UseMethod("predict")
    return(invisible(NULL))
  }
  
  if(!is_keras_available())
    stop("Please run install_keras() before using this predict method. ?install_keras for options and details (e.g. to use gpu).")
  
  newdata <- as.data.frame(newdata)
  
  y_in_newdata <- length(setdiff(all.vars(object$input_formula[[2]]), colnames(newdata))) == 0
  y_test <- if(y_in_newdata) eval(object$input_formula[[2]], envir = newdata) else NULL
  
  if(is.null(y_test)){
    if(verbose > 0)
      message("Unable to construct y_test from newdata.\n")
  }else{
    
    if(object$y_type != "continuous"){
      y_test_labels <- unique(y_test)
      if(mean(y_test_labels %in% object$y_labels) != 1)
        message("newdata contains outcomes not present in training data.\nCompare object$y_labels (from the trained object) to fit$y_test_labels.")
    }
        
  }
  
  test_formula <- if(is.null(y_test)) as.formula(paste(object$input_formula[[1]], object$input_formula[[3]])) else object$input_formula
  
  
  if(object$sparse_data){
    newdata_tmp <- sparse.model.matrix(test_formula, data = newdata, row.names = FALSE, ...)
    X_test <- Matrix(0, nrow = nrow(newdata), ncol = object$P, sparse = TRUE, ...)
  }else{
    newdata_tmp <- model.matrix(test_formula, data = newdata, row.names = FALSE, ...)
    X_test <- matrix(0, nrow = nrow(newdata), ncol = object$P, ...)
  }
  
  colnames(X_test) <- object$colnames_x

  cols <- match(colnames(newdata_tmp), object$colnames_x)
  cols <- cols[!is.na(cols)]
  if(length(cols) == 0)
    stop("newdata does not contain any columns with the same name as the training data.")
  X_test[ , cols] <- newdata_tmp[ , which(colnames(newdata_tmp) %in% object$colnames_x)]
  remove(newdata_tmp)
  
  if(!is.null(object$train_scale)){
    
    transformation <- if(object$train_scale$scale == "zero_one") zero_one else z
    
    # only continuous variables are scaled but
    # different levels may be observed on categorical variables in test and training
    # making the column numbers in X_train meaningless...
    
    nfo <- as.data.frame(object$train_scale$X)
    
    for(colname in colnames(object$train_scale$X)){
      
      test_col <- match(colname, colnames(X_test))
      X_test[, test_col] <- transformation(X_test[ , test_col], nfo[[colname]][1], nfo[[colname]][2])
      
    }
    
    if(!is.null(y_test) & object$y_type == "continuous")
      y_test <- transformation(y_test, object$train_scale$y[1], object$train_scale$y[2])
    
  }
        
  if(is.null(object$y_type)) # legacy with kerasformula 0.1.0
    object$y_type <- if(object$K == 2) "binary" else "multinomial"
  
  if(object$y_type == "continuous"){
    
    y_fit <- predict(object$model, X_test, 
                     batch_size = batch_size, verbose = verbose)
    
  }else{
    
    # 1 + to get back to R/Fortran land... 
    y_fit <- object$y_labels[1 + predict_classes(object$model, X_test, 
                                                 batch_size = batch_size, verbose = verbose)]
  }
  
  fit <- list(fit = y_fit, y_test = y_test)
  
  if(!is.null(y_test)){
    
    if(object$y_type == "continuous"){
      
      fit[["MSE_predictions"]] <- mean((y_fit - y_test)^2)
      fit[["MAE_predictions"]] <- mean(abs(y_fit - y_test))
      fit[["R2_predictions"]] <- cor(y_fit, y_test)^2
      fit[["cor_kendals"]] <- cor(y_fit, y_test, method="kendal") # guard against broken clock predictions
      
    }else{
      
      fit[["y_test_labels"]] <- y_test_labels 
      fit[["confusion"]] <- confusion(y_test = y_test, predictions = y_fit)
      fit[["accuracy"]] <- mean(y_fit == y_test)
      
    }
    
  }
  
  
  
  return(fit)
    
}