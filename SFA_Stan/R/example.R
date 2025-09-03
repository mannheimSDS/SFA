# R/example.R
# IBFA: probit-IRT (votes) + hurdle-NB-IRT (words) with battery-wise horseshoe
# Uses cmdstanr. Install toolchain once: install.packages("cmdstanr"); cmdstanr::install_cmdstan()

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
  library(bayesplot)
})

# ---- Paths (relative to the project root) ----
root_dir  <- getwd()
stan_file <- gsub("/SFA_Stan","/SFA_Stan/stan/ibfa_hurdle_probit_hs.stan", root_dir)
#stan_file <- file.path(root_dir, "stan", "ibfa_hurdle_probit_hs.stan")
stopifnot(file.exists(stan_file))

out_dir   <- file.path(root_dir, "output", "fits")
ppc_dir   <- file.path(root_dir, "output", "ppc")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ppc_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(123)

# ---- Simulate data consistent with the Stan model ----
# N respondents, J vote items, L word items
N  <- 80
J  <- 25
L  <- 30

# Factor dimensions (start simple: shared-only; set Kv/Kw > 0 if you want)
K_s <- 2
K_v <- 0
K_w <- 0

# Shared abilities
theta <- matrix(rnorm(N * K_s), nrow = N, ncol = K_s)

# ----- Vote block (probit IRT) -----
# Item params for simulation (independent from priors in Stan)
alphaV_true <- rnorm(J, 0, 1.0)    # intercepts
bV_true     <- rnorm(J, 0, 1.0)    # difficulties
a_s_vote_true <- matrix(rnorm(K_s * J, 0, 0.8), nrow = K_s, ncol = J)

etaV <- sweep(theta %*% a_s_vote_true, 2, bV_true, FUN = "-")
etaV <- sweep(etaV, 2, alphaV_true, FUN = "+")
pV   <- pnorm(etaV)
X    <- matrix(rbinom(N * J, size = 1, prob = as.vector(pV)), nrow = N, ncol = J)

# ----- Word block (hurdle NB with token offset) -----
tokens <- pmax(1L, rpois(N, lambda = 400))   # doc length / exposure (>0)

# Appearance (probit gate)
alphaH_true <- rnorm(L, 0, 1.0)
c_s_word_true <- matrix(rnorm(K_s * L, 0, 0.7), nrow = K_s, ncol = L)
zeta <- sweep(theta %*% c_s_word_true, 2, alphaH_true, FUN = "+")
rho  <- pnorm(zeta)  # Pr(Y>0)

# Intensity (NB-2 mean with offset)
alphaM_true <- rnorm(L, 0, 0.8)
b_s_word_true <- matrix(rnorm(K_s * L, 0, 0.7), nrow = K_s, ncol = L)
nu  <- sweep(theta %*% b_s_word_true, 2, alphaM_true, FUN = "+") +
  matrix(log(tokens), nrow = N, ncol = L)
mu  <- exp(nu)
kappa_true <- rgamma(L, shape = 2, rate = 1) + 1  # >0

# Draw hurdle counts (NB truncated at zero when the gate is open)
Y <- matrix(0L, nrow = N, ncol = L)
Z <- matrix(rbinom(N * L, 1, as.vector(rho)), nrow = N, ncol = L)
for (i in 1:N) for (l in 1:L) {
  if (Z[i, l] == 1) {
    # Draw until positive to emulate NB^+ (zero-truncated NB)
    y <- 0L
    # Safety loop (NB rarely returns 0 when mu is reasonable; this is fine for a sim)
    while (y == 0L) {
      y <- rnbinom(1, size = kappa_true[l], mu = mu[i, l])
    }
    Y[i, l] <- y
  }
}

# ---- Build Stan data list ----
stan_data <- list(
  N = N, J = J, L = L,
  X = X,
  Y = Y,
  tokens = as.numeric(tokens),
  K_s = K_s, K_v = K_v, K_w = K_w
)

# ---- Compile Stan model ----
mod <- cmdstan_model(stan_file)

# ---- Sample ----
fit <- mod$sample(
  data = stan_data,
  seed = 42,
  chains = 4, parallel_chains = 4,
  iter_warmup = 800,
  iter_sampling = 800,
  refresh = 100,
  adapt_delta = 0.9,
  max_treedepth = 12
)

# Save draws
fit_rds <- file.path(out_dir, sprintf("ibfa_fit_%s.rds", format(Sys.time(), "%Y%m%d_%H%M%S")))
fit$save_object(file = fit_rds)

# ---- Quick diagnostics ----
fit$cmdstan_diagnose()  # divergences, E-BFMI, treedepth
print(fit$summary(variables = c("kappa", "alphaV[1]", "bV[1]", "alphaH[1]", "alphaM[1]")), n = 20)

# ---- Basic sanity checks (rough PPCs) ----
# Observed zero rates by word item
obs_zero_rate <- colMeans(Y == 0)
png(file.path(ppc_dir, "ppc_zero_rates.png"), width = 900, height = 500)
par(mfrow = c(1, 2))
hist(obs_zero_rate, breaks = 15, main = "Observed zero rates (words)", xlab = "Pr(Y=0)")
plot(tokens, rowSums(Y > 0), pch = 19, cex = 0.6,
     xlab = "Tokens (exposure)", ylab = "Num. positive words per respondent",
     main = "Exposure vs. positives")
dev.off()

# Posterior summaries of a few loadings (as a smoke test)
sum_kappa <- fit$summary(variables = grep("^kappa\\[", fit$metadata()$model_params, value = TRUE))
head(sum_kappa, 10)
