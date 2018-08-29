#' confusion
#' 
#' Confusion matrix or (for larger number of levels) confusion table.
#' 
#' @param object Optional fit object. confusion() assumes object contains holdout/vaidation data as `y_test` and the forecasts/classifications as `predictions` but alternative variable names can be specified with the input arguments by those names.
#' @param y_test A vector of holdout/validation data or the name in object (if fit object provided but alternative variable name required).
#' @param predictions A vector predictions or the name in object (if fit object provided but alternative variable name required).
#' @param return_xtab Logical. If TRUE, returns confusion matrix, which is a crosstable with correct predictions on the diagonal (if all levels are predicted at least once). If FALSE, returns data.frame with columns for percent correct, most common misclassification, second most common misclassification, and other predictions. Only defaults to crosstable-style if y_test has fewer than six levels.
#' @param digits Number of digits for proportions when return_xtab=FALSE; if NULL, no rounding is performed.
#' @return confusion matrix or table as specified by return_xtab.
#' @examples
#' mtcars$make <- unlist(lapply(strsplit(rownames(mtcars), " "), function(tokens) tokens[1]))
#' company <- if(is_keras_available()){
#'                kms(make ~ ., mtcars, Nepochs=1, verbose=0)
#'            }else{
#'                  list(y_test = mtcars$make[1:5], 
#'                  predictions = sample(mtcars$make, 5))
#'                  }
#' confusion(company)     # same as above confusion$company if is_keras_available() == TRUE
#' confusion(company, return_xtab = FALSE) # focus on pCorrect, most common errors
#' @export
confusion <- function(object = NULL, y_test = NULL, predictions = NULL, return_xtab = NULL, digits=3){
  
  obj <- data.frame(y_test = if(is.null(object)) y_test else object[[if(is.null(y_test)) "y_test" else y_test]],
                    predictions = if(is.null(object)) predictions else object[[if(is.null(predictions)) "predictions" else predictions]],
                    stringsAsFactors = FALSE)

  return_xtab <- if(is.null(return_xtab)) n_distinct(obj$y_test) < 6 else return_xtab 
  
  if(return_xtab){
    
    cf <- table(obj$y_test, obj$predictions)
    colnames(cf) <- paste0(colnames(cf), "_pred")
    rownames(cf) <- paste0(rownames(cf), "_obs")
    return(cf)
    
  }else{
    
    obj[["correct"]] <- obj$y_test == obj$predictions
    cf <- data.frame(label = unique(obj$y_test)) 
    # confusion 
    
    cf[["N"]] <- NA
    cf[["pCorrect"]] <- NA
    cf[["MCE"]] <- NA # Most Common Error
    cf[["pMCE"]] <- 0 # proportion that are MCE
    cf[["MCE2"]] <- NA # second most common error
    cf[["pMCE2"]] <- 0 
    cf[["pOther"]] <- 0
    
    for(i in 1:nrow(cf)){
      
      lab_i <- obj$y_test == cf$label[i]
      cf$N[i] <- Nlab_i <- sum(lab_i)
      
      cf$pCorrect[i] <- mean(obj$y_test[lab_i] == obj$predictions[lab_i])
      
      tab <- sort(table(obj$predictions[lab_i]), decreasing = TRUE)
      tab <- tab[-which(names(tab) == cf$label[i])]
      
      if(cf$pCorrect[i] != 1 && length(tab) > 0){
        
        cf$MCE[i] <- names(tab)[1]
        cf$pMCE[i] <- tab[1]/Nlab_i
        
        if(cf$pCorrect[i] + cf$pMCE[i] != 1){
          
          cf$MCE2[i] <- names(tab)[2]
          cf$pMCE2[i] <- tab[2]/Nlab_i
          cf$pOther[i] <- 1 - (cf$pCorrect[i] + cf$pMCE[i] + cf$pMCE2[i])
          
        }
    
      }
            
    }
    
    if(!is.null(digits)){
      cf$pCorrect <- round(cf$pCorrect, digits=digits) 
      cf$pMCE <-  round(cf$pMCE, digits=digits)
      cf$pMCE2 <- round(cf$pMCE2, digits=digits) 
      cf$pOther <- round(cf$pOther, digits=digits)
    }
   return(cf) 
  }
    
}  

#' plot_confusion
#' 
#' @param ... kms_fit objects. (For each, object$y_test must be binary or categorical.)
#' @param display Logical: display ggplot comparing confusion matrices? (Default TRUE.)
#' @param return_ggplot Default FALSE (if TRUE, returns the ggplot object for further customization, etc.).
#' @param title ggplot title
#' @param subtitle ggplot subtitle
#' @param position Position adjustment, either as a string, or the result of a call to a position adjustment function
#' @param alpha Transparency of points, between 0 and 1
#' @return (optional) ggplot. set return_ggplot=TRUE
#' @examples 
#' 
#' if(is_keras_available()){
#' 
#'    model_tanh <- kms(Species ~ ., iris, 
#'                      activation = "tanh", Nepochs=5, 
#'                      units=4, seed=1, verbose=0)
#'    model_softmax <- kms(Species ~ ., iris, 
#'                         activation = "softmax", Nepochs=5, 
#'                         units=4, seed=1, verbose=0)
#'    model_relu <- kms(Species ~ ., iris, 
#'                      activation = "relu", Nepochs=5, 
#'                      units=4, seed=1, verbose=0)
#'                      
#'    plot_confusion(model_tanh, model_softmax, model_relu, 
#'                   title="Species", 
#'                   subtitle="Activation Function Comparison")
#'    
#' }
#' @importFrom ggplot2 element_text geom_point labs theme theme_minimal ylim
#' @export
plot_confusion <- function(..., display = TRUE, return_ggplot = FALSE, title="", subtitle="", position="identity", alpha = 1){
  
  args <- list(...)
  if(unique(lapply(args, class)) != "kms_fit")
    stop("All objects must be kms_fit (i.e., output from kerasformula::kms()).")
  
  model <- as.character(as.list(substitute(list(...)))[-1L])   
  y_type <- c()
  
  confusions <- list()
  for(i in 1:length(args)){
    
    confusions[[i]] <- confusion(args[[i]], return_xtab = FALSE)
    confusions[[i]][["Model"]] <- model[i]
    y_type[i] <- args[[i]][["y_type"]]
    
  }  
  
  if("continuous" %in% unique(y_type))
    stop("plot_confusion() is intended for categorical variables.")
  
  cf <- do.call(rbind, confusions)
  
  # circumventing CRAN check 
  label <- pCorrect <- Model <- N <- NULL
  
  g <- ggplot(cf, aes(x =label, y =pCorrect, col=Model, size=N)) + theme_minimal() + 
    geom_point(position = position, alpha = alpha) + theme(axis.text.x = element_text(angle = 70, hjust = 1)) + 
    ylim(c(0,1)) + labs(y = "Proportion Correct\n(out of sample)", x="Model Comparison",
                        title=title, subtitle=subtitle) 
  
  if(display) print(g)
  if(return_ggplot) return(g) 
  
}

