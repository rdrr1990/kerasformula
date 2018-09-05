#' kms_kcv
#' 
#' k_folds cross-validation. Except for pTraining and validation split (replaced by k_folds), all inputs are the same as kms(). See ?kms
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
#' @param k_folds Number of folds. For example, if k_folds == 5 (default), the data are split into 80\% training, 20\% testing (five times).
#' @param Nepochs Number of epochs; default == 15. To be passed to keras::fit.  
#' @param batch_size Default batch size is 32 unless emedding == TRUE in which case batch size is 1. (Smaller eases memory issues but may affect ability of optimizer to find global minimum). To be passed to several functions library(keras) functions like fit(), predict_classes(), and layer_embedding(). If embedding==TRUE, number of training obs must be a multiple of batch size. 
#' @param loss To be passed to keras::compile. Defaults to "binary_crossentropy", "categorical_crossentropy", or "mean_squared_error" based on input_formula and data.
#' @param metrics Additional metric(s) beyond the loss function to be passed to keras::compile. Defaults to "mean_absolute_error" and "mean_absolute_percentage_error" for continuous and c("accuracy") for binary/categorical (as well whether whether examples are correctly classified in one of the top five most popular categories or not if the number of categories K > 20).  
#' @param optimizer To be passed to keras::compile. Defaults to "optimizer_adam", an algorithm for first-order gradient-based optimization of stochastic objective functions introduced by Kingma and Ba (2015) here: https://arxiv.org/pdf/1412.6980v8.pdf.
#' @param scale_continuous How to scale each non-binary column of the training data (and, if y is continuous, the outcome). The default 'scale_continuous = 'zero_one'' places each non-binary column of the training model matrix on [0, 1]; 'scale_continuous = z' standardizes; 'scale_continuous = NULL' leaves the data on its original scale.
#' @param sparse_data Default == FALSE. If TRUE, X is constructed by sparse.model.matrix() instead of model.matrix(). Recommended to improve memory usage if there are a large number of categorical variables or a few categorical variables with a large number of levels. May compromise speed, particularly if X is mostly numeric.
#' @param drop_intercept TRUE by default.
#' @param seed Integer vector of length k_folds or list containing k_folds-length seed vector to be passed to the sources of variation: R, Python's Numpy, and Tensorflow. If seed is NULL, automatically generated. Note setting seed ensures data will be partitioned in the same way but to ensure identical results, set disable_gpu = TRUE and disable_parallel_cpu = TRUE. Wrapper for use_session_with_seed(), which is to be called before compiling by the user if a compiled Keras model is passed into kms. See also see https://stackoverflow.com/questions/42022950/. 
#' @param verbose Default == 1. Setting to 0 disables progress bar and epoch-by-epoch plots (disabling them is recommended for knitting RMarkdowns if X11 not installed).
#' @param ... Additional parameters to be passsed to Matrix::sparse.model.matrix.
#' @return An kms_kcv_fit object; nested list containing train and test estimates produced by kms() and predict.kms(), respectively.
#' @examples
#' if(is_keras_available()){
#' 
#'     kcv_out <- kms_kcv(Species ~ ., iris, Nepochs=1, verbose=0)
#'     kcv_out$train_f1$history # nested object, train and test 
#'     kcv_out$test_f3$accuracy # for each fold f = 1, 2, ... 
#'     
#'     
#' }else{
#'    cat("Please run install_keras() before using kms(). ?install_keras for options like gpu.")
#' }
#' @author Pete Mohanty
#' @export
kms_kcv <- function(input_formula, data, keras_model_seq = NULL, 
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
                    k_folds = 5, 
                    Nepochs = 15, batch_size = NULL, 
                    loss = NULL, metrics = NULL, optimizer = "optimizer_adam",
                    scale_continuous = "zero_one", drop_intercept=TRUE,
                    sparse_data = FALSE,
                    seed = list(seed = NULL, disable_gpu=FALSE, disable_parallel_cpu = FALSE), 
                    verbose = 1, ...){

  out <- list()
  out[["folds"]] <- sample(k_folds, nrow(data), replace=TRUE)
  out[["k_folds"]] <- k_folds
  class(out) <- "kms_kcv_fit"
  
  if(!is.list(seed)){
    seed_list <- list(seed = NULL, disable_gpu=FALSE, disable_parallel_cpu = FALSE)
    if(is.numeric(seed)){
      if(length(seed) == k_folds){
        seed_list$seed <- seed
      }else{
        seed_list$seed <- seed[1] + 0:(k_folds - 1)
      }
    }
      
  }else{
    seed_list <- seed 
    # allow user to pass in integer which controls software but not hardware parameters too
    # see https://github.com/rdrr1990/kerasformula/blob/master/examples/kms_replication.md
  } 
  if(is.null(seed_list$seed)){
    
    seed_list$seed <- sample(2^30, size = k_folds) 
    # py Seed must be between 0 and 2**32 - 1 but avoiding R integer coercion issues with larger than 2^30
    
  } 
  
  if(verbose)
    cat("starting k folds cross validation... \n\n\n\n\n")
  
  for(f in 1:k_folds){
    
    tmp_seed <- seed_list
    tmp_seed$seed <- tmp_seed$seed[f]
    
    out[[paste0("train_f", f)]] <- kms(input_formula = input_formula, 
                                       data = data[out$folds != f, ], 
                                       keras_model_seq = keras_model_seq, 
                                       N_layers = N_layers, 
                                       units = units, 
                                       activation = activation, 
                                       dropout = dropout, 
                                       use_bias = use_bias, 
                                       kernel_initializer = kernel_initializer, 
                                       kernel_regularizer = kernel_regularizer, 
                                       bias_regularizer = bias_regularizer, 
                                       activity_regularizer = activity_regularizer, 
                                       embedding = embedding, 
                                       pTraining = 1,
                                       validation_split = 0,
                                       Nepochs = Nepochs, 
                                       batch_size = batch_size, 
                                       loss = loss, 
                                       metrics = metrics, 
                                       optimizer = optimizer, 
                                       scale_continuous = scale_continuous, 
                                       drop_intercept = drop_intercept, 
                                       sparse_data = sparse_data, 
                                       seed = tmp_seed, 
                                       verbose = verbose)
                                       # args(...)) #, ...)
    
    if(verbose)
      cat("\n\nFinished training on fold", f, "\n")
  
    out[[paste0("test_f", f) ]] <- predict(out[[paste0("train_f", f)]],
                                           data[out$folds == f, ], 
                                           batch_size = if(is.null(batch_size)) 32 else batch_size)
    if(verbose)
      cat("Finished testing on fold", f, "\n\n\n")
    
  }
  return(out)
}





