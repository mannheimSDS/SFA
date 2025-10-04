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
n.sim <- 50    # total items (we'll use 25 votes + 25 words)
k.sim <- 500   # respondents/legislators (N)

# True dimension weights (first 3 dominant)
d.sim <- rep(0, n.sim); d.sim[1:3] <- c(2, 1.5, 1) * 3

# Latent structure: Theta.sim is (items x respondents)
U.sim <- matrix(rnorm(n.sim^2,  sd = .5), n.sim, n.sim)
V.sim <- matrix(rnorm(k.sim*n.sim, sd = .5), k.sim, n.sim)
Theta.sim <- U.sim %*% diag(d.sim) %*% t(V.sim)  # (n.sim x k.sim)

# Item and respondent random offsets (as in your code)
item_fx <- rep(1, n.sim) %*% t(rnorm(k.sim, sd = .5))          # (n x k)
pers_fx <- rnorm(n.sim, sd = .5) %*% t(rep(1, k.sim))          # (n x k)

probs.sim <- pnorm(-1 + Theta.sim + item_fx + pers_fx)         # (n x k)

# Split into votes (first 25 rows) and counts (second 25 rows)
votes.mat <- apply(probs.sim[1:25, ], c(1, 2), function(p) rbinom(1, 1, p))   # 25 x 500
count.mat <- apply(probs.sim[26:50,], c(1, 2), function(p) rpois(1, 20*p))    # 25 x 500

# -----------------------------------------------------------------------------
# Convert counts -> GLOBAL ranks 0..C for the ordinal probit words battery
# -----------------------------------------------------------------------------
C <- 100  # top category

mat_convert <- sort(unique(as.vector(count.mat)))
mat_convert <- mat_convert[mat_convert>0]
mat_convert <- cbind(1:length(mat_convert),mat_convert)
for(i in 1:nrow(mat_convert)) count.mat[count.mat==mat_convert[i,2]] <- mat_convert[i,1]
count.mat[count.mat > C] <- C

counts_vec_0 <- as.vector(count.mat)
counts_vec_0 <- as.numeric(as.factor(counts_vec_0))-1
counts_vec_0[counts_vec_0>C] <- C
unique_vec  <- unique(counts_vec_0[counts_vec_0>0])

# Reshape back to items x N, then transpose to N x L for Stan
L <- nrow(count.mat)
N <- ncol(count.mat)
Y <- t(matrix(counts_vec, nrow = L, ncol = N))
storage.mode(Y) <- "integer"                                    # N x L

# Choose C for Stan = max observed category (<= C_cap and <= length(uvals))
C_data <- max(counts_vec)
if (C_data < 1L) C_data <- 1L 

# Votes need to be N x J as well
X <- t(votes.mat)                                      # 500 x 25

# Document lengths (tokens): use total raw counts per respondent (must be >0)
tokens <- pmax(1L, colSums(count.mat))                 # length N

# Empirical CDF
# Inputs
C <- max(Y)                 # categories are {0,..,C}
raw_counts <- as.vector(count.mat)   # all word-count cells

# Empirical cdf over c-1, for c = 0..C-1:
# F_y[c+1] = P(T <= c) with F_y[1] = P(T <= 0) = 1
F_y <- numeric(C)
F_y[1] <- mean(raw_counts <= -1)  # = 0
if (C > 1) {
  for (c in 1:(C-1)) F_y[c+1] <- mean(raw_counts <= (c-1))
}
F_y <- pmin(1 - 1e-6, pmax(1e-6, F_y))
F_dir <- +1L




# -----------------------------------------------------------------------------
# Build Stan data (shared-only test: K_v = K_w = 0)
# -----------------------------------------------------------------------------
N  <- k.sim
J  <- 25
L  <- 25
K_s <- 3         # feel free to change; shared-only still okay
K_v <- 0
K_w <- 0

# Gauss–Hermite nodes/weights (for item-effect marginalization)
Mgh <- 20
gh  <- statmod::gauss.quad(n = Mgh, kind = "hermite")
gh_nodes   <- as.numeric(gh$nodes)
gh_weights <- as.numeric(gh$weights)

## Sanity check....
stopifnot(all(X %in% 0:1))
stopifnot(is.integer(X))
stopifnot(is.integer(Y))
stopifnot(min(Y) >= 0, max(Y) <= C)
stopifnot(all(tokens > 0))

## initializations
make_inits <- function(sd) {
  list(
    beta0 = 0,
    beta1_pos = 0.8,   # spacing scale
    beta2 = 1.0,       # ~linear in F_y
    zeta = 0.0,
    sigma_alphaW = 0.3,
    sigma_bV = 0.6,
    sigma_gammaV = 0.4,
    theta = matrix(0, sd$N, sd$K_s),
    gammaV_raw = rep(0, sd$N),
    z_a_s_vote = matrix(0, sd$K_s, sd$J),
    lambda_a_s_vote = matrix(0.4, sd$K_s, sd$J),
    tau_a_s_vote = 0.4, c2_a_s_vote = 4,
    z_beta_s_word = matrix(0, sd$K_s, sd$L),
    lambda_beta_s_word = matrix(0.4, sd$K_s, sd$L),
    tau_beta_s_word = 0.4, c2_beta_s_word = 4
    # add the *_vote/_word specific blocks if Kv/Kw > 0
  )
}

##
stan_data <- list(
  N = N, J = ncol(X), L = ncol(Y), C = C_data,
  X = X, Y = Y, tokens = as.numeric(tokens),
  K_s = 3, K_v = 0, K_w = 0,
  Mgh = Mgh, gh_nodes = gh_nodes, gh_weights = gh_weights
)

#
stan_data$F_y   <- as.vector(F_y)  # length C
stan_data$F_dir <- as.integer(F_dir)

# -----------------------------------------------------------------------------
# Compile & sample
# -----------------------------------------------------------------------------
mod <- cmdstan_model(stan_path)

fit <- mod$sample(data = stan_data, init = function(chain_id) make_inits(stan_data),
                  seed=123, chains=2, parallel_chains = 2, iter_warmup=600, iter_sampling=600,
                  adapt_delta=0.95, max_treedepth=12)

run_dir <- "/home/rstudio/SFA_Stan/output"
fit$save_object(file = file.path(run_dir, "test_sfa_zip.rds"))
# Quick peek at key parameters (if your stan has generated quantities, grab those too)
vars_basic <- c("zeta","sigma_bV","sigma_alphaW")
print(fit$summary(variables = vars_basic), n = length(vars_basic))

## Recover ideal points


# 2) IDEAL POINTS: pull, summarize, compare to truth if available --------------
ip_vars <- c("ip_vote_shared","ip_vote_total",
             "ip_word_shared","ip_word_total","ip_word_total_w_offset",
             "ip_shared_joint")

ip_draws <- fit$draws(variables = ip_vars)
ip_sum <- summarise_draws(ip_draws, mean, sd,
                          ~quantile(.x, probs = c(.05,.95)))
