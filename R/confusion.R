#' confusion
#' 
#' Confusion matrix or (for larger number of levels) confusion table.
#' 
#' @param object Optional fit object. confusion() assumes object contains holdout/vaidation data as `y_test` and the forecasts/classifications as `predictions` but alternative variable names can be specified with the input arguments by those names.
#' @param y_test A vector of holdout/validation data or the name in object (if fit object provided but alternative variable name required).
#' @param predictions A vector predictions or the name in object (if fit object provided but alternative variable name required).
#' @param return_xtab Logical. If TRUE, returns confusion matrix, which is a crosstable with correct predictions on the diagonal (if all levels are predicted at least once). If FALSE, returns (rectangular) table with columns for percent correct, most common misclassification, second most common misclassification, and other predictions. Defaults to TRUE (crosstable-style) only if number of levels < 6.
#' @param digits Number of digits for proportions when return_xtab=FALSE; if NULL, no rounding is performed.
#' @return confusion matrix or table as specified by return_xtab.
#' #' @examples
#' mtcars$make <- unlist(lapply(strsplit(rownames(mtcars), " "), function(tokens) tokens[1]))
#' company <- if(is_keras_available()) kms(make ~ ., mtcars) else list(y_test = mtcars$make[1:5], predictions = sample(mtcars$make, 5))
#' confusion(company)     # same as above confusion$company if is_keras_available() == TRUE
#' confusion(company, return_xtab = FALSE) # focus on pCorrect, most common errors
#' @export
confusion <- function(object = NULL, y_test = NULL, predictions = NULL, return_xtab = NULL, digits=3){
  
  obj <- data.frame(y_test = if(is.null(object)) y_test else object[[if(is.null(y_test)) "y_test" else y_test]],
                    predictions = if(is.null(object)) predictions else object[[if(is.null(predictions)) "predictions" else predictions]],
                    stringsAsFactors = FALSE)

  return_xtab <- if(is.null(return_xtab)) n_distinct(obj$predictions) < 6 else return_xtab 
  
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


