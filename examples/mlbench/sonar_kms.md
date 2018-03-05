kerasformula on mlbench data
================

Here is an example from `mlbench`. Thanks to Michael Gallagher for suggesting these data!

``` r
library(kerasformula)
library(mlbench)
data(Sonar)

for(v in 1:60)
  Sonar[,v] <- as.numeric(Sonar[, v])

table(Sonar$Class)
```


      M   R 
    111  97 

``` r
class_dense <- kms(Class ~ ., Sonar)
class_dense$evaluations$acc
```

    [1] 0.5

Here is another example using `lstm` (which is typically used on larger datasets). Note that `input_dimension` should be `P`, the number of columns in the model matrix (which was already constructed in the previous example).

``` r
class_dense$P
```

    [1] 61

``` r
k <- keras_model_sequential()
k %>%
  layer_embedding(input_dim = class_dense$P, output_dim = 50) %>% 
  layer_lstm(units = 32, dropout = 0.4, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, # number of levels observed on y or just 1 if binary  
              activation = 'sigmoid')

k %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'nadam',
  metrics = c('accuracy')
)

class_lstm <- kms(Class ~ ., Sonar, k)
class_lstm$evaluations$acc
```

    [1] 0.5652174
