library(tidyverse)
library(keras)
rm(list = ls())
use_python("~/opt/anaconda3/bin/python")
use_condaenv(conda = "~/opt/anaconda3/bin/conda")


gather_matrix <- function(mat){
  list(
    row = row(mat),# rownames(mat)[row(mat)] %||% row(mat),
    col = col(mat),# colnames(mat)[col(mat)] %||% col(mat),
    value = mat) %>%
  map_dfc(as.vector)
}

n <- 400
dim <- 2
library(igraph)
set.seed(1234)
B <- matrix(c(.4, .2, .2, .4), 2)
g <- sample_sbm(n, B, c(n/2, n/2))[]
gdf <- g %>% gather_matrix() %>%
  filter(row<col) %>% mutate(row = row-1, col = col-1)

ase <- function(g, dim){
  usv <- irlba::partial_eigen(g, dim)
  with(usv, vectors %*% diag(sqrt(values)))
}

sig_ase <- function(shape, dtype){
  x <- ase(g, dim)
  p <- x %*% t(x) %>% pmin(.99) %>% pmax(.01) %>% matrix(n)
  l <- matrix(log(p/(1-p)), n)
  ase(l, dim)
}

get_ase_p <- function(g,dim){
  
  x <- ase(g, dim)
  p <- matrix(pmax(.01, pmin(.99, x %*% t(x))),n)
}

train_data <- list(matrix(gdf$row,ncol=1), matrix(gdf$col, ncol=1))
train_labels <- matrix(gdf$value,ncol=1)
# ================================
# inputs
# ================================

input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

# embedding matrix for mean vectors
embedding_mu <- layer_embedding(
  input_dim = n,
  output_dim = dim,
  embeddings_initializer = sig_ase,
  input_length = 1, 
  name = "embedding_mu"
)

# select target mu from the mu embedding matrix
target_vector_mu <- input_target %>%
  embedding_mu() %>% 
  layer_flatten()

# select context mu from the mu embedding matrix
context_vector_mu <- input_context %>%
  embedding_mu() %>%
  layer_flatten()

dotprod <- layer_dot(inputs = list(target_vector_mu, 
                                   context_vector_mu), 1L)


# output <- layer_activation(dotprod, activation = "sigmoid",
#                       trainable = FALSE)
output <- layer_dense(dotprod, units = 1, name="sig" activation = "sigmoid")

# ================================
# model compile
# ================================
model <- keras_model(list(input_target, input_context), output)
model %>% compile(
  loss = "binary_crossentropy", 
  optimizer = "adam",
  metric = "accuracy")

summary(model)

history <- model %>% fit(
  train_data,
  train_labels,
  epochs = 100,
  batch_size = 128,
  verbose=0
)

(model$predict(list(matrix(c(0,n/2),ncol=1), matrix(c(1,1), ncol=1))))
plot(history)

xhat <- model$get_layer("embedding_mu")$get_weights()[[1]]

param <- model$get_layer("sig")$get_weights()
a0 <- as.double(param[[2]])
a1 <- as.double(param[[1]])

phat <- 1/(1 + exp(-(a0 + a1*xhat %*% t(xhat))))

gdf %>% mutate(pred = phat[upper.tri(phat)])

gdf %>% mutate(pred = phat[upper.tri(phat)]) %>%
  mutate(b = ifelse(row < n / 2 & col < n / 2, "11",
              ifelse(row >= n/2 & col >= n/2, "22", "12"))) %>%
  ggplot(aes(x=pred, fill=b))+geom_density(alpha = .3)
