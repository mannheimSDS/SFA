// IBFA: Probit-IRT (votes) + Hurdle-NB-IRT (words) with battery-wise Regularized Horseshoe
// - Shared latent traits theta[N, K_s]
// - Optional view-specific traits u_vote[N, K_v], u_word[N, K_w]
// - Votes use probit link
// - Words use hurdle (probit gate) + zero-truncated NB with offset log(tokens)
// - Regularized Horseshoe on loadings, with SEPARATE priors by battery/component
//   (votes shared/specific; words shared/specific for appearance and intensity)

data {
  int<lower=1> N;                 // respondents
  int<lower=1> J;                 // vote items
  int<lower=1> L;                 // word items

  int<lower=0,upper=1> X[N, J];   // votes (binary)
  int<lower=0> Y[N, L];           // word counts (incl. zeros)
  vector<lower=1e-8>[N] tokens;   // exposure (document length), strictly > 0!

  int<lower=1> K_s;               // shared factor dimensions
  int<lower=0> K_v;               // vote-specific dims (0 allowed)
  int<lower=0> K_w;               // word-specific dims (0 allowed)
}

transformed data {
  // ---------- Reasonable default hyperparameters ----------
  // Latent trait & intercept/difficulty scales
  real<lower=0> SD_THETA   = 1.0;
  real<lower=0> SD_UV      = 1.0;
  real<lower=0> SD_UW      = 1.0;
  real<lower=0> SD_ALPHA_V = 1.5;
  real<lower=0> SD_B_V     = 1.5;
  real<lower=0> SD_ALPHA_H = 1.5;
  real<lower=0> SD_ALPHA_M = 1.5;

  // Regularized Horseshoe slab (shared across blocks)
  real<lower=0> HS_SLAB_SCALE = 2.5;  // s
  real<lower=0> HS_SLAB_DF    = 1.0;  // nu

  // Global half-Cauchy scales for each block (can tune per dataset)
  real<lower=0> TAU0_VOTE_SHARED   = 1.0;
  real<lower=0> TAU0_VOTE_SPECIFIC = 1.0;
  real<lower=0> TAU0_WORD_APP_SHARED   = 1.0;
  real<lower=0> TAU0_WORD_APP_SPECIFIC = 1.0;
  real<lower=0> TAU0_WORD_INT_SHARED   = 1.0;
  real<lower=0> TAU0_WORD_INT_SPECIFIC = 1.0;
}

parameters {
  // ---------- Latent traits ----------
  matrix[N, K_s] theta;                 // shared
  matrix[N, K_v] u_vote;                // vote-specific (0 cols ok)
  matrix[N, K_w] u_word;                // word-specific (0 cols ok)

  // ---------- Votes: probit IRT ----------
  // Difficulties / intercepts
  vector[J] bV;                         // difficulties
  vector[J] alphaV;                     // intercepts (can be absorbed into bV)

  // RHS: shared loadings a_s_vote[K_s, J]
  matrix[K_s, J] z_a_s_vote;
  matrix<lower=0>[K_s, J] lambda_a_s_vote;
  real<lower=0> tau_a_s_vote;
  real<lower=0> c2_a_s_vote;

  // RHS: vote-specific loadings a_v_vote[K_v, J] (skips if K_v=0)
  matrix[K_v, J] z_a_v_vote;
  matrix<lower=0>[K_v, J] lambda_a_v_vote;
  real<lower=0> tau_a_v_vote;
  real<lower=0> c2_a_v_vote;

  // ---------- Words: hurdle (appearance) ----------
  vector[L] alphaH;                     // hurdle intercepts

  // RHS: shared appearance loadings c_s_word[K_s, L]
  matrix[K_s, L] z_c_s_word;
  matrix<lower=0>[K_s, L] lambda_c_s_word;
  real<lower=0> tau_c_s_word;
  real<lower=0> c2_c_s_word;

  // RHS: word-specific appearance c_w_word[K_w, L]
  matrix[K_w, L] z_c_w_word;
  matrix<lower=0>[K_w, L] lambda_c_w_word;
  real<lower=0> tau_c_w_word;
  real<lower=0> c2_c_w_word;

  // ---------- Words: intensity (NB for positives) ----------
  vector[L] alphaM;                     // intensity intercepts
  vector<lower=0>[L] kappa;             // NB-2 dispersion (phi/size): Var = mu + mu^2 / kappa

  // RHS: shared intensity loadings b_s_word[K_s, L]
  matrix[K_s, L] z_b_s_word;
  matrix<lower=0>[K_s, L] lambda_b_s_word;
  real<lower=0> tau_b_s_word;
  real<lower=0> c2_b_s_word;

  // RHS: word-specific intensity b_w_word[K_w, L]
  matrix[K_w, L] z_b_w_word;
  matrix<lower=0>[K_w, L] lambda_b_w_word;
  real<lower=0> tau_b_w_word;
  real<lower=0> c2_b_w_word;
}

transformed parameters {
  // Actual loadings built via the regularized horseshoe multipliers
  matrix[K_s, J] a_s_vote;
  matrix[K_v, J] a_v_vote;

  matrix[K_s, L] c_s_word;
  matrix[K_w, L] c_w_word;

  matrix[K_s, L] b_s_word;
  matrix[K_w, L] b_w_word;

  // Votes: shared
  for (k in 1:K_s)
    for (j in 1:J) {
      real lt = lambda_a_s_vote[k, j] * tau_a_s_vote;
      real mult = (sqrt(c2_a_s_vote) * lt) / sqrt(c2_a_s_vote + square(lt));
      a_s_vote[k, j] = z_a_s_vote[k, j] * mult;
    }

  // Votes: specific
  for (k in 1:K_v)
    for (j in 1:J) {
      real lt = lambda_a_v_vote[k, j] * tau_a_v_vote;
      real mult = (sqrt(c2_a_v_vote) * lt) / sqrt(c2_a_v_vote + square(lt));
      a_v_vote[k, j] = z_a_v_vote[k, j] * mult;
    }

  // Words: appearance (shared)
  for (k in 1:K_s)
    for (l in 1:L) {
      real lt = lambda_c_s_word[k, l] * tau_c_s_word;
      real mult = (sqrt(c2_c_s_word) * lt) / sqrt(c2_c_s_word + square(lt));
      c_s_word[k, l] = z_c_s_word[k, l] * mult;
    }

  // Words: appearance (specific)
  for (k in 1:K_w)
    for (l in 1:L) {
      real lt = lambda_c_w_word[k, l] * tau_c_w_word;
      real mult = (sqrt(c2_c_w_word) * lt) / sqrt(c2_c_w_word + square(lt));
      c_w_word[k, l] = z_c_w_word[k, l] * mult;
    }

  // Words: intensity (shared)
  for (k in 1:K_s)
    for (l in 1:L) {
      real lt = lambda_b_s_word[k, l] * tau_b_s_word;
      real mult = (sqrt(c2_b_s_word) * lt) / sqrt(c2_b_s_word + square(lt));
      b_s_word[k, l] = z_b_s_word[k, l] * mult;
    }

  // Words: intensity (specific)
  for (k in 1:K_w)
    for (l in 1:L) {
      real lt = lambda_b_w_word[k, l] * tau_b_w_word;
      real mult = (sqrt(c2_b_w_word) * lt) / sqrt(c2_b_w_word + square(lt));
      b_w_word[k, l] = z_b_w_word[k, l] * mult;
    }
}

model {
  // ---------- Priors ----------
  // Latent traits
  to_vector(theta)  ~ normal(0, SD_THETA);
  if (K_v > 0) to_vector(u_vote) ~ normal(0, SD_UV);
  if (K_w > 0) to_vector(u_word) ~ normal(0, SD_UW);

  // Intercepts/difficulties
  alphaV ~ normal(0, SD_ALPHA_V);
  bV     ~ normal(0, SD_B_V);

  alphaH ~ normal(0, SD_ALPHA_H);
  alphaM ~ normal(0, SD_ALPHA_M);

  // Dispersion for NB (lognormal is a good default)
  kappa ~ lognormal(0, 1);

  // ----- Regularized Horseshoe priors (battery-wise blocks) -----
  // Locals ~ half-Cauchy(0,1)   (positive Cauchy)
  to_vector(lambda_a_s_vote) ~ cauchy(0, 1);
  to_vector(lambda_a_v_vote) ~ cauchy(0, 1);
  to_vector(lambda_c_s_word) ~ cauchy(0, 1);
  to_vector(lambda_c_w_word) ~ cauchy(0, 1);
  to_vector(lambda_b_s_word) ~ cauchy(0, 1);
  to_vector(lambda_b_w_word) ~ cauchy(0, 1);

  // Globals ~ half-Cauchy(0, tau0)
  tau_a_s_vote ~ cauchy(0, TAU0_VOTE_SHARED);
  tau_a_v_vote ~ cauchy(0, TAU0_VOTE_SPECIFIC);

  tau_c_s_word ~ cauchy(0, TAU0_WORD_APP_SHARED);
  tau_c_w_word ~ cauchy(0, TAU0_WORD_APP_SPECIFIC);

  tau_b_s_word ~ cauchy(0, TAU0_WORD_INT_SHARED);
  tau_b_w_word ~ cauchy(0, TAU0_WORD_INT_SPECIFIC);

  // Slab variances c^2 ~ Inv-Gamma(nu/2, nu*s^2/2)
  c2_a_s_vote ~ inv_gamma(0.5 * HS_SLAB_DF, 0.5 * HS_SLAB_DF * square(HS_SLAB_SCALE));
  c2_a_v_vote ~ inv_gamma(0.5 * HS_SLAB_DF, 0.5 * HS_SLAB_DF * square(HS_SLAB_SCALE));

  c2_c_s_word ~ inv_gamma(0.5 * HS_SLAB_DF, 0.5 * HS_SLAB_DF * square(HS_SLAB_SCALE));
  c2_c_w_word ~ inv_gamma(0.5 * HS_SLAB_DF, 0.5 * HS_SLAB_DF * square(HS_SLAB_SCALE));

  c2_b_s_word ~ inv_gamma(0.5 * HS_SLAB_DF, 0.5 * HS_SLAB_DF * square(HS_SLAB_SCALE));
  c2_b_w_word ~ inv_gamma(0.5 * HS_SLAB_DF, 0.5 * HS_SLAB_DF * square(HS_SLAB_SCALE));

  // Standard normals for z (good geometry)
  to_vector(z_a_s_vote) ~ normal(0, 1);
  to_vector(z_a_v_vote) ~ normal(0, 1);
  to_vector(z_c_s_word) ~ normal(0, 1);
  to_vector(z_c_w_word) ~ normal(0, 1);
  to_vector(z_b_s_word) ~ normal(0, 1);
  to_vector(z_b_w_word) ~ normal(0, 1);

  // ---------- Likelihood ----------
  vector[N] log_tokens = log(tokens);

  // Votes: probit IRT
  for (j in 1:J) {
    vector[N] eta_V = alphaV[j]
                    + (theta * a_s_vote[, j])
                    - bV[j];
    if (K_v > 0)
      eta_V += u_vote * a_v_vote[, j];

    target += bernoulli_lpmf(X[, j] | Phi(eta_V));
  }

  // Words: hurdle NB
  for (l in 1:L) {
    vector[N] zeta = alphaH[l] + (theta * c_s_word[, l]);
    vector[N] nu   = alphaM[l] + log_tokens + (theta * b_s_word[, l]);
    if (K_w > 0) {
      zeta += u_word * c_w_word[, l];
      nu   += u_word * b_w_word[, l];
    }
    vector[N] rho = Phi(zeta);   // appearance probability
    vector[N] mu  = exp(nu);     // NB mean for positives

    for (i in 1:N) {
      if (Y[i, l] == 0) {
        target += bernoulli_lpmf(0 | rho[i]); // log(1 - rho)
      } else {
        // truncated NB: log rho + log NB(y) - log(1 - NB(0))
        target += bernoulli_lpmf(1 | rho[i]);
        target += neg_binomial_2_lpmf(Y[i, l] | mu[i], kappa[l])
                - neg_binomial_2_lpmf(0 | mu[i], kappa[l]);
      }
    }
  }
}

generated quantities {
  // Quick posterior predictive summaries (optional)
  vector[L] mean_zero_prob;
  for (l in 1:L) mean_zero_prob[l] = 0;
  // (Fill if you want PPCs; omitted to keep compile time light.)
}
