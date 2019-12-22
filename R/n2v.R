#' @export
get_n2v_model <- function(n, dim, init_x=NULL){

  as_embedding_initializer <- function(x=NULL){
    if(is.null(x)){
      initializer_random_uniform
    }
    function(shape, dtype){
      k_cast(k_reshape(x, shape), dtype)
    }
  }

  input_target <- layer_input(shape = 1)
  input_context <- layer_input(shape = 1)

  # embedding matrix for mean vectors
  embedding_mu <- layer_embedding(
    input_dim = n,
    output_dim = dim,
    embeddings_initializer = as_embedding_initializer(init_x),
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
  output <- layer_dense(dotprod, units = 1, name="sig", activation = "sigmoid")

  # ================================
  # model compile
  # ================================
  model <- keras_model(list(input_target, input_context), output)
}


n2v_training_data <- function(g, w, k = 1, directed = FALSE, df = FALSE){
  d <- Matrix::rowSums(g[]) + 1
  d <- d/sqrt(sum(d))
  w_plus <- ergo_weights_pos(g,w)
  gdf <- w_plus %>% gather_matrix() %>%
    filter(row < col, value != 0) %>%
    rename(from = row, to = col, weight = value) %>% mutate(edge = 1)

  gdf <- gdf %>%
    mutate(weight = w_plus[cbind(from, to)]) %>%
    bind_rows(list(from = 1:n, to = 1:n) %>%
                cross_df(.filter = `>=`) %>%
                mutate(edge = 0,
                       weight = k * w * d[from] * d[to])) %>%
    mutate(from = from - 1, to = to-1)


  shape <- c(nrow(gdf),1)
  if(df){
    gdf
  } else {
    list(
      data = list(k_reshape(gdf$from, shape),
                  k_reshape(gdf$to, shape)),
      labels = k_reshape(gdf$edge, shape),
      weights = k_reshape(gdf$weight, shape))
  }
}



ergo_weights_pos <- function(g, w, epsilon = NULL){
  g <- g[]
  n <- nrow(g)
  if(is.null(epsilon)){
    epsilon <- 1 / n
  }
  g <- g + epsilon
  dinv <- Matrix::Diagonal(n, 1/Matrix::rowSums(g))
  t <- dinv %*% g

  seq_len(w-1) %>% reduce(~{g + .x %*% t}, .init = g)
}


#' @export
n2v_fit <- function(model, train, epochs = 50, batch_size = 2048){
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metric = "mean_squared_error")




  history <- model %>% fit(
    train$data,
    train$labels,
    epochs = epochs,
    batch_size = batch_size,
    verbose=0,
    weights = train$weights
  )
}

#' @export
n2v <- function(g, dim, w, init = NULL){
  if(is.null(init)){
    x <- gbilinear_embed_spectral(g[], dim, tol = .001)
  }


  train <- n2v_training_data(g, w)
  # gdf <-  n2v_training_data(g, w, df = TRUE)
  model <- get_n2v_model(n, dim, x)
  history <- n2v_fit(model, train)
  embed <- model$get_layer("embedding_mu")$get_weights()[[1]]
  attr(embed, "history") <- history
  embed
}
