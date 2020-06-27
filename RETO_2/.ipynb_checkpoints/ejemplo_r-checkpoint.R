library(tidyverse)
library(caret)
library(MLmetrics)
library(fastDummies)

df = read.csv('DATA_RETO_2.csv', header = TRUE)

Y = df$cumulative_cases
X = df %>% select(-cumulative_cases)

evaluate <- function(X, Y){
  # NO ALTERAR ESTA FUNCION
  set.seed(2020)
  n_folds = 3
  folds <- sample(cut(seq(1,nrow(X)),breaks=n_folds,labels=FALSE))
  scores <- double(length = n_folds)
  for(i in 1:n_folds){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    
    X_train <- X[-testIndexes, ]
    Y_train <- Y[-testIndexes]
    X_test <- X[testIndexes, ]
    Y_test <- Y[-testIndexes]
    
    model <- model_fit(X_train, Y_train)
    predictions = model_predict(X_test, model)
    scores[i] <- MLmetrics::MAE(y_pred = predictions,
                   y_true = Y_test)
  }
  return(mean(scores))
}

# Aqui escribes el resto de tu codigo

X <- X %>% select(-country, -country_code, -iso, -date)

# Para limpiar Valores Ausentes
for(i in colnames(X)){
  if(is.numeric(X[[i]])){
    X[[i]] <- replace_na(X[[i]], replace = mean(X[[i]], na.rm = TRUE))
  }
}

# Para codificar columnas categoricas 
X <- fastDummies::dummy_cols(X, remove_selected_columns = TRUE)

model_fit <- function(X,Y){
  # Programa tu modelo aqui
  modelo_de_ejemplo <- caret::train(x = X, y = Y, method = 'lm')
}

model_predict <- function(X, model){
  # Crea un vector de predicciones aqui
    # Por ejemplo, para predecir siempre la media:
    return(predict(model, X))
}

evaluate(X,Y)

## CASO LOG SCALING

model_fit_log <- function(X,Y){
  # Programa tu modelo aqui
  modelo_de_ejemplo <- caret::train(x = X, y = log1p(Y), method = 'lm')
}

model_predict_log <- function(X, model){
  # Crea un vector de predicciones aqui
    # Por ejemplo, para predecir siempre la media:
    return(expm1(predict(model, X)))
}

evaluate(X,Y)