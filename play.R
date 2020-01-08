library(tidyverse)
library(igraph)
library(keras)
devtools::load_all()
rm(list = ls())
use_python("~/opt/anaconda3/bin/python")
use_condaenv(conda = "~/opt/anaconda3/bin/conda")

# Make the graph
n <- 100
set.seed(1234)
B <- matrix(c(.6, .2,
              .2, .6), 2)
g <- sample_sbm(n, B, c(n/2, n/2))

# Set parameters
dim <- 2
w <- 8
x <- gbilinear_embed_spectral(g[], dim, log_odds, tol = .001)
n2v_res <- n2v(g, dim, w, init = x)
xhat <- n2v_res$embed
model <- n2v_res$model
as_tibble(rbind(xhat, x),.name_repair = "unique") %>%
  mutate(block = rep(rep(c("a","b"), each = n/2),2),
         which = rep(c("res", "init"), each = n)) %>%
  ggplot(aes(x=...1, y=...2, color = block, shape = which, alpha = which)) + geom_point()

as_tibble(xhat-x,.name_repair = "unique") %>%
  mutate(block = rep(c("a","b"), each = n/2)) %>%
  ggplot(aes(x=...1, y=...2, color = block)) + geom_point()
# param <- model$get_layer("sig")$get_weights()
# a0 <- as.double(param[[2]])
# a1 <- as.double(param[[1]])

phat <- 1/(1 + exp(-(a0 + a1*xhat %*% t(xhat))))

g[] %>% gather_matrix() %>% filter(row<col) %>% mutate(pred = phat[upper.tri(phat)]) %>%
  mutate(b = ifelse(row < n / 2 & col < n / 2, "11",
              ifelse(row >= n/2 & col >= n/2, "22", "12"))) %>%
  ggplot(aes(x=pred, fill=b))+geom_density(alpha = .3)
