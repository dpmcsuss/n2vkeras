
library(tidyverse)

#' @export
ase_embed <- function(g, dim){
  if(dim < nrow(g[])){
    usv <- irlba::partial_eigen(g[], dim)
  } else {
    usv <- eigen(g)
  }
  with(usv, vectors %*% diag(sqrt(values)))

}

#' @export
low_rank_approx <- function(g, rank){
  irlba::irlba(g, rank) %>% with(u %*% diag(d) %*% t(v))
}

#' @title Embed based on link function via Spectral Methods
#' @description Estimates an embedding based on a bilinear link function.
#'
#' The function first computes a low rank approximation of , then computes
#' @
#'
#' @export
gbilinear_embed_spectral <- function(g, dim, f = log_odds, dim_init=dim, ...){
  n <- nrow(g[])
  g[] %>%
    low_rank_approx(dim_init) %>%
    f(...) %>%
    # matrix(n, n) %>%
    ase_embed(dim)
}

#' @export
log_odds <- function(x, tol = 0){
  if(tol == 0 && any(x<0 | x>1)){
    warning("Some probabilities are <0 or >1. Use tol>0 to avoid infinite values.")
  }
  x %>% pmin(1-tol) %>% pmax(tol) %>%
    (rlang::as_function(~log(.x/(1-.x))))
}



#' @export
gather_matrix <- function(mat, sym = TRUE){
  mat_df <- list(
    row = row(mat),# rownames(mat)[row(mat)] %||% row(mat),
    col = col(mat),# colnames(mat)[col(mat)] %||% col(mat),
    value = mat) %>%
    map_df(as.vector)
}

#' @export
logit <- function(x){
  1 / (1 + exp(-x))
}

