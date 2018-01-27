#' predict.kms_fit
#' 
#' predict function for kms_fit object. Wrapper for keras::predict_classes(). Creates a sparse model matrix with the same columns as the training data, some of which may be 0.
#' 
#' @param object output from kms()
#' @param newdata new data. Performs merge so that x_test has the same columns as the object created by kms_fit using the user-provided input formula. y_test is also generated from that formula.
#' @param batch_size To be passed to keras::predict_classes. Default == 32.
#' @param verbose 0 ot 1, to be passed to keras::predict_classes. Default == 0.
#' @param ... additional parameters to build the sparse matrix x_test.
#' @return list containing predictions, y_test, confusion matrix.
#' @examples 
#' if(is_keras_available()){
#' 
#'  mtcars$make <- unlist(lapply(strsplit(rownames(mtcars), " "), function(tokens) tokens[1]))
#'  company <- kms(make ~ ., mtcars[3:32, ], Nepochs = 2)
#'  forecast <- predict(company, mtcars[1:2, ])
#'  forecast$confusion
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
  
  newdata_tmp <- sparse.model.matrix(object$input_formula, data = newdata, row.names = FALSE, ...)
  x_test <- Matrix(0, nrow = nrow(newdata), ncol = object$P, sparse = TRUE, ...)
  colnames(x_test) <- object$colnames_x

  cols <- match(colnames(newdata_tmp), object$colnames_x)
  cols <- cols[!is.na(cols)]
  if(length(cols) == 0)
    stop("newdata does not contain any columns with the same name as the training data.")
  x_test[ , cols] <- newdata_tmp[ , which(colnames(newdata_tmp) %in% object$colnames_x)]
  remove(newdata_tmp)
  
  y_test <- eval(object$input_formula[[2]], envir = newdata)
  y_test_labels <- unique(y_test)
  if(mean(y_test_labels %in% object$y_labels) != 1)
    warning("newdata contains outcomes not present in training data.\nCompare object$y_labels (from the trained object) to fit$y_test_labels.")

  # 1 + to get back to R/Fortran land... 
  y_fit <- object$y_labels[1 + predict_classes(object$model, x_test, 
                                               batch_size = batch_size, verbose = verbose)]
  fit <- list(fit = y_fit, y_test = y_test, y_test_labels = y_test_labels, 
              confusion = confusion(y_test = y_test, predictions = y_fit),
              accuracy = mean(y_fit == y_test))
  
  return(fit)
    
}