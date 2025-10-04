library(statmod)
library(cmdstanr)
library(posterior)
library(dplyr)

# ---- locate stan file (works from project root or SFA_Stan/R) ---------------
stan_path <- '/home/rstudio/SFA_Stan/stan/sfa_ibfa.stan'
mod <- cmdstan_model(stan_path)
mod$exe_file()
file.exists(mod$exe_file())  # should print TRUE if STAN file compile

set.seed(1)

# -----------------------------------------------------------------------------
# Your SFA-style simulation (rows = items, cols = respondents)
# -----------------------------------------------------------------------------
n_votes <- 20    # total items (we'll use 50 votes + 150 words)
n_words <- 50    # total items (we'll use 50 votes + 150 words)

legislators <- 20   # respondents/legislators (N)

# True dimension weights (first 3 dominant)
d_sim <- rep(0, legislators); d_sim[1:3] <- c(2, 1.5, 1) * 3

# Latent structure: Theta.sim is (items x respondents)
ideal_points <- matrix(rnorm(legislators^2,  sd = .5), nrow= legislators)
word_locations <- matrix(rnorm(legislators*n_words,  sd = .5), nrow = legislators)
vote_locations <- matrix(rnorm(legislators*n_votes,  sd = .5), nrow = legislators)

theta_votes <- ideal_points%*%diag(d_sim)%*%(vote_locations) +
      rep(1,legislators)%*%t(rnorm(n_votes, sd=.25)) +
      rnorm(legislators, sd = .25)%*%t(rep(1,n_votes)) 


theta_words <- ideal_points%*%diag(d_sim)%*%(word_locations) +
  rep(1,legislators)%*%t(rnorm(n_words, sd=.25)) +
  rnorm(legislators, sd = .25)%*%t(rep(1,n_words)) 

# Split into votes (first 25 rows) and counts (second 25 rows)
votes_mat <- apply(pnorm(theta_votes/3), c(1, 2), function(p) rbinom(1, 1, p))   # 25 x 500
count_mat <- apply(pnorm(theta_words/3), c(1, 2), function(p) rpois(1, 20*p))    # 25 x 500

## Working into STAN
# votes_mat : N x n_votes   (0/1)
# count_mat : N x n_words   (counts)
N        <- legislators
J        <- n_votes
L        <- n_words

# 1) Convert counts -> GLOBAL ordinal categories Y in {0..C}
C_cap <- 100L
count_mat[count_mat > C_cap] <- C_cap

# Map unique positives to consecutive 1..U while zeros stay 0
counts_vec_0 <- as.vector(count_mat)
# factors turn unique values into 1..U; we subtract 1 so zeros become 0
counts_vec_0 <- as.numeric(as.factor(counts_vec_0)) - 1L
counts_vec_0[counts_vec_0 < 0] <- 0L
counts_vec_0[counts_vec_0 > C_cap] <- C_cap

# reshape to N x L (your count_mat is already N x L, so 'byrow=FALSE' is right)
Y <- matrix(counts_vec_0, nrow = N, ncol = L, byrow = FALSE)
storage.mode(Y) <- "integer"
C <- max(Y)  # top category actually used

# 2) Votes matrix X is already N x J
X <- votes_mat
storage.mode(X) <- "integer"
stopifnot(all(X %in% 0:1))

# 3) Tokens (doc length offset) per legislator
tokens <- pmax(1, rowSums(count_mat))  # N-length, strictly > 0

# 4) Empirical CDF
F_y <- numeric(C)
yv <- as.vector(Y)
for (c in 1:C) F_y[c] <- mean(yv <= (c - 1L))

# clamp to (0,1) and make strictly increasing (avoid equal τ's)
eps <- 1e-6
F_y <- pmin(1 - eps, pmax(eps, F_y))   # in (eps, 1-eps)

# tiny ramp makes neighbors strictly increasing: critical for log_diff_exp
ramp <- 1e-8
F_y <- cummax(F_y)                       # non-decreasing
F_y <- pmin(1 - (C*ramp + eps), F_y)     # leave headroom
F_y <- F_y + ramp * (1:C)                # strictly increasing
F_dir <- 1L    

# 5) Gauss–Hermite nodes/weights
gh <- statmod::gauss.quad(n = 20, kind = "hermite")

# 6) Stan data (shared-only first: K_s=3; you can raise later)
stan_data <- list(
  N = N, J = J, L = L, C = C,
  X = X, Y = Y, tokens = as.numeric(tokens),
  K_s = 5, K_v = 0, K_w = 0,
  Mgh = length(gh$nodes),
  gh_nodes = as.numeric(gh$nodes),
  gh_weights = as.numeric(gh$weights),
  F_y = as.vector(F_y),
  F_dir = as.integer(F_dir)
)

# 7) Safe initials (keep cutpoints well-spaced; predictors tame)
make_inits <- function(sd) {
  out <- list(
    # cutpoint warp
    beta0 = 0, beta1_pos = 0.8, beta2 = 1.0, zeta = 0,
    sigma_alphaW = 0.3, sigma_bV = 0.6, sigma_gammaV = 0.4,
    
    # latent traits (shared)
    theta      = matrix(0, sd$N, sd$K_s),
    gammaV_raw = rep(0, sd$N),
    
    # vote loadings (shared)
    z_a_s_vote       = matrix(0, sd$K_s, sd$J),
    lambda_a_s_vote  = matrix(0.4, sd$K_s, sd$J),
    tau_a_s_vote     = 0.4,
    c2_a_s_vote      = 4,
    
    # word loadings (shared)
    z_beta_s_word      = matrix(0, sd$K_s, sd$L),
    lambda_beta_s_word = matrix(0.4, sd$K_s, sd$L),
    tau_beta_s_word    = 0.4,
    c2_beta_s_word     = 4,
    
    # specific-block scalars (declared even if K_v/K_w = 0)
    tau_a_v_vote    = if (sd$K_v > 0) 0.3 else 1.0,
    c2_a_v_vote     = 4,
    tau_beta_w_word = if (sd$K_w > 0) 0.3 else 1.0,
    c2_beta_w_word  = 4
  )
  
  # add vote-specific only if K_v > 0
  if (sd$K_v > 0) {
    out$u_vote          <- matrix(0, sd$N, sd$K_v)
    out$z_a_v_vote      <- matrix(0, sd$K_v, sd$J)
    out$lambda_a_v_vote <- matrix(0.4, sd$K_v, sd$J)
  }
  
  # add word-specific only if K_w > 0
  if (sd$K_w > 0) {
    out$u_word            <- matrix(0, sd$N, sd$K_w)
    out$z_beta_w_word     <- matrix(0, sd$K_w, sd$L)
    out$lambda_beta_w_word<- matrix(0.4, sd$K_w, sd$L)
  }
  
  out
}


fit <- mod$sample(
  data = stan_data,
  init = function(chain_id) make_inits(stan_data),
  seed = 123,
  chains = 2, parallel_chains = 2,
  iter_warmup = 800, iter_sampling = 800,
  adapt_delta = 0.95, max_treedepth = 12,
  refresh = 100
)


# 8) Compile & sample
mod <- cmdstan_model(stan_path)  # already compiled per your earlier call
fit <- mod$sample(
  data = stan_data,
  init = function(chain_id) make_inits(stan_data),
  seed = 123,
  chains = 2, parallel_chains = 2,
  iter_warmup = 800, iter_sampling = 800,
  adapt_delta = 0.95, max_treedepth = 12,
  refresh = 100
)

## Save
file_rds <- '/home/rstudio/SFA_Stan/output/sfa_ibfa_20leg.stan'
fit$save_object(file = file_rds)
# 9) Quick diagnostics (Rhat/ESS)
sum_all <- fit$summary()
bad <- subset(sum_all, rhat > 1.05)
if (nrow(bad)) {
  message("Potential issues:")
  print(bad[, c("variable","rhat","ess_bulk","ess_tail")][1:min(30,nrow(bad)), ])
} else {
  message("All parameters pass basic Rhat/ESS thresholds.")
}

# 10) Recover the FIRST SHARED DIMENSION theta[,1] with stable sign + 95% CI
library(posterior)
vars_theta1 <- paste0("theta[", 1:N, ",1]")
dm <- as_draws_matrix(fit$draws(variables = vars_theta1))  # draws x N

# anchor sign: use the largest |value| in the first draw
first_draw <- dm[1, ]
i_ref <- which.max(abs(first_draw))
s_ref <- ifelse(sign(first_draw[i_ref]) == 0, 1, sign(first_draw[i_ref]))
# flip each draw if the anchor's sign disagrees
s_draw <- sign(dm[, i_ref]); s_draw[s_draw == 0] <- 1
s_draw <- s_draw * s_ref
dm_aligned <- dm * matrix(s_draw, nrow(dm), N, byrow = FALSE)

# summarize
theta1_mean <- colMeans(dm_aligned)
theta1_ci   <- apply(dm_aligned, 2, quantile, probs = c(0.025, 0.975))
theta1_summary <- data.frame(
  legislator = 1:N,
  mean = theta1_mean,
  lo95 = theta1_ci[1, ],
  hi95 = theta1_ci[2, ]
)

head(theta1_summary)

cor(theta1_mean, ideal_points[,1:5])
cor(theta1_mean, ideal_points[,11:15])
