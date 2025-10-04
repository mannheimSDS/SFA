// ============================================================================
// IBFA on a common latent probit scale
//   • Votes: Type-II/CJR probit (no item intercepts in params); item thresholds
//            bV_j ~ N(0, sigma_bV^2) are MARGINALIZED per item via GH quadrature.
//   • Words: Ordinal probit on GLOBAL ranks {0..C}; word intercepts
//            alphaW_l ~ N(0, sigma_alphaW^2) are MARGINALIZED per item via GH.
//   • Shared theta + battery-specific u_vote, u_word; learned zeta offset.
//   • Separate Regularized Horseshoe (RHS) by battery & block.
//   • Minimal changes from prior file: removed alphaW_raw/bV_raw (and their
//     centering); added sigma_bV, sigma_alphaW, and GH nodes/weights in data.
// ============================================================================

data {
  // Sizes
  int<lower=1> N;        // legislators
  int<lower=1> J;        // vote items
  int<lower=1> L;        // word items
  int<lower=1> C;        // top rank (e.g., 100 -> categories 0..100)

  // Observations
  array[N, J] int<lower=0, upper=1> X;   // votes
  array[N, L] int<lower=0, upper=C> Y;   // word global ranks

 // Old code
 // int<lower=0,upper=1> X[N, J];     // votes
 // int<lower=0,upper=C> Y[N, L];     // word global ranks
  vector<lower=1e-8>[N] tokens;     // doc lengths (>0)

  // Factor dimensions
  int<lower=1> K_s;      // shared dims
  int<lower=0> K_v;      // vote-specific dims (0 ok)
  int<lower=0> K_w;      // word-specific dims (0 ok)

  // Gauss–Hermite nodes/weights for ∫ exp(-x^2) g(x) dx ≈ Σ w_m g(x_m)
  int<lower=1> Mgh;
  vector[Mgh] gh_nodes;
  vector[Mgh] gh_weights;
}

transformed data {
  // z-score(log-length) for offset
  vector[N] log_tokens;
  real mu_log = mean(log(tokens));
  real sd_log = sd(log(tokens));
  vector[N] z_log_tokens;
  for (i in 1:N) log_tokens[i] = log(tokens[i]);
  if (sd_log <= 1e-12) sd_log = 1.0;
  for (i in 1:N) z_log_tokens[i] = (log_tokens[i] - mu_log) / sd_log;

  // GH normalization: ∫ φ(γ;0,σ²) f(γ) dγ = (1/√π) Σ w_m f(√2 σ x_m)
  real log_inv_sqrt_pi = -0.5 * log(pi());
}

parameters {
  // -------- Latent traits --------
  matrix[N, K_s] theta;           // shared
  matrix[N, K_v] u_vote;          // vote-specific (0 cols ok)
  matrix[N, K_w] u_word;          // word-specific (0 cols ok)

  // -------- Votes (Type-II/CJR probit) --------
  vector[N] gammaV_raw;           // person RE (will be centered)
  real<lower=0> sigma_gammaV;     // sd(person RE)
  real<lower=0> sigma_bV;         // sd(item thresholds) — marginalized

  // RHS: vote loadings
  matrix[K_s, J] z_a_s_vote;
  matrix<lower=0>[K_s, J] lambda_a_s_vote;
  real<lower=0> tau_a_s_vote;
  real<lower=0> c2_a_s_vote;

  matrix[K_v, J] z_a_v_vote;
  matrix<lower=0>[K_v, J] lambda_a_v_vote;
  real<lower=0> tau_a_v_vote;
  real<lower=0> c2_a_v_vote;

  // -------- Words (ordinal probit with global cutpoints) --------
  real tau0;                      // free hurdle cutpoint
  real log_delta;                 // spacing; delta = exp(log_delta)
  real zeta;                      // slope on z_log_tokens (offset)
  real<lower=0> sigma_alphaW;     // sd(word intercepts) — marginalized

  // RHS: word loadings
  matrix[K_s, L] z_beta_s_word;
  matrix<lower=0>[K_s, L] lambda_beta_s_word;
  real<lower=0> tau_beta_s_word;
  real<lower=0> c2_beta_s_word;

  matrix[K_w, L] z_beta_w_word;
  matrix<lower=0>[K_w, L] lambda_beta_w_word;
  real<lower=0> tau_beta_w_word;
  real<lower=0> c2_beta_w_word;
}

transformed parameters {
  // Center person RE (sum_i gammaV_i = 0)
  vector[N] gammaV;
  {
    real m = mean(gammaV_raw);
    for (i in 1:N) gammaV[i] = gammaV_raw[i] - m;
  }

  // Positive spacing
  real delta = exp(log_delta);

  // Build RHS loadings (actual coefficients)
  matrix[K_s, J] a_s_vote;
  matrix[K_v, J] a_v_vote;
  matrix[K_s, L] beta_s_word;
  matrix[K_w, L] beta_w_word;

  for (k in 1:K_s) for (j in 1:J) {
    real lt = lambda_a_s_vote[k, j] * tau_a_s_vote;
    real mult = (sqrt(c2_a_s_vote) * lt) / sqrt(c2_a_s_vote + square(lt));
    a_s_vote[k, j] = z_a_s_vote[k, j] * mult;
  }
  for (k in 1:K_v) for (j in 1:J) {
    real lt = lambda_a_v_vote[k, j] * tau_a_v_vote;
    real mult = (sqrt(c2_a_v_vote) * lt) / sqrt(c2_a_v_vote + square(lt));
    a_v_vote[k, j] = z_a_v_vote[k, j] * mult;
  }
  for (k in 1:K_s) for (l in 1:L) {
    real lt = lambda_beta_s_word[k, l] * tau_beta_s_word;
    real mult = (sqrt(c2_beta_s_word) * lt) / sqrt(c2_beta_s_word + square(lt));
    beta_s_word[k, l] = z_beta_s_word[k, l] * mult;
  }
  for (k in 1:K_w) for (l in 1:L) {
    real lt = lambda_beta_w_word[k, l] * tau_beta_w_word;
    real mult = (sqrt(c2_beta_w_word) * lt) / sqrt(c2_beta_w_word + square(lt));
    beta_w_word[k, l] = z_beta_w_word[k, l] * mult;
  }
}

model {
  // -------- Priors (unchanged except new sigmas) --------
  to_vector(theta) ~ normal(0, 1);
  if (K_v > 0) to_vector(u_vote) ~ normal(0, 1);
  if (K_w > 0) to_vector(u_word) ~ normal(0, 1);

  gammaV_raw   ~ normal(0, sigma_gammaV);
  sigma_gammaV ~ normal(0, 1) T[0,];

  sigma_bV     ~ normal(0, 1) T[0,];      // votes: item-threshold sd
  sigma_alphaW ~ normal(0, 1) T[0,];      // words: item-intercept sd

  tau0      ~ normal(0, 1.5);
  log_delta ~ normal(0, 0.5);
  zeta      ~ normal(0, 1);

  // RHS (same as before)
  to_vector(lambda_a_s_vote)    ~ cauchy(0, 1);
  to_vector(lambda_a_v_vote)    ~ cauchy(0, 1);
  to_vector(lambda_beta_s_word) ~ cauchy(0, 1);
  to_vector(lambda_beta_w_word) ~ cauchy(0, 1);

  tau_a_s_vote    ~ cauchy(0, 1);
  tau_a_v_vote    ~ cauchy(0, 1);
  tau_beta_s_word ~ cauchy(0, 1);
  tau_beta_w_word ~ cauchy(0, 1);

  c2_a_s_vote     ~ inv_gamma(0.5, 0.5 * square(2.5));
  c2_a_v_vote     ~ inv_gamma(0.5, 0.5 * square(2.5));
  c2_beta_s_word  ~ inv_gamma(0.5, 0.5 * square(2.5));
  c2_beta_w_word  ~ inv_gamma(0.5, 0.5 * square(2.5));

  to_vector(z_a_s_vote)    ~ normal(0, 1);
  to_vector(z_a_v_vote)    ~ normal(0, 1);
  to_vector(z_beta_s_word) ~ normal(0, 1);
  to_vector(z_beta_w_word) ~ normal(0, 1);

  // -------- Likelihood (only the item effects part changed) --------

  // Votes: marginalize item thresholds bV_j ~ N(0, sigma_bV^2)
  for (j in 1:J) {
    vector[N] eta_no_b = gammaV + (theta * a_s_vote[, j]);
    if (K_v > 0) eta_no_b += u_vote * a_v_vote[, j];

    vector[Mgh] lp_m;
    for (m in 1:Mgh) {
      real b_m = sqrt(2) * sigma_bV * gh_nodes[m];
      vector[N] p = Phi(eta_no_b - b_m);
      real ll = 0;
      for (i in 1:N) ll += bernoulli_lpmf(X[i, j] | p[i]);
      lp_m[m] = log(gh_weights[m]) + ll;
    }
    target += log_sum_exp(lp_m) - 0.5 * log(pi()); // add log(1/sqrt(pi))
  }

  // Words: marginalize word intercepts alphaW_l ~ N(0, sigma_alphaW^2)
  for (l in 1:L) {
    vector[N] v0 = zeta * z_log_tokens + (theta * beta_s_word[, l]);
    if (K_w > 0) v0 += u_word * beta_w_word[, l];

    vector[Mgh] lp_m;
    for (m in 1:Mgh) {
      real a_m = sqrt(2) * sigma_alphaW * gh_nodes[m];
      real ll = 0;
      for (i in 1:N) {
        int y = Y[i, l];
        real v = v0[i] + a_m;
        if (y == 0) {
          ll += normal_lcdf(tau0 - v | 0, 1);
        } else if (y < C) {
          real ub = tau0 + exp(log_delta) * y;
          real lb = tau0 + exp(log_delta) * (y - 1);
          ll += log_diff_exp(normal_lcdf(ub - v | 0, 1),
                             normal_lcdf(lb - v | 0, 1));
        } else { // y == C
          real t_last = tau0 + exp(log_delta) * (C - 1);
          ll += normal_lccdf(t_last - v | 0, 1);
        }
      }
      lp_m[m] = log(gh_weights[m]) + ll;
    }
    target += log_sum_exp(lp_m) - 0.5 * log(pi());
  }
}

generated quantities {
  // Convenience
  // real delta = exp(log_delta);

  // ---------- Ideal-point summaries (per legislator) ----------
  vector[N] ip_vote_shared_raw             = rep_vector(0, N);
  vector[N] ip_vote_specific_raw           = rep_vector(0, N);
  vector[N] ip_vote_total_raw              = rep_vector(0, N);

  vector[N] ip_word_shared_raw             = rep_vector(0, N);
  vector[N] ip_word_specific_raw           = rep_vector(0, N);
  vector[N] ip_word_total_raw              = rep_vector(0, N);
  vector[N] ip_word_total_w_offset_raw     = rep_vector(0, N);

  vector[N] ip_shared_joint_raw            = rep_vector(0, N);

  // Votes: average contributions over items (include person RE in total)
  {
    matrix[N, J] contrib_shared = theta * a_s_vote;    // N x J
    matrix[N, J] contrib_spec;
    if (K_v > 0) contrib_spec = u_vote * a_v_vote;     // N x J

    for (i in 1:N) {
      real acc_sh = 0;
      real acc_sp = 0;
      for (j in 1:J) {
        acc_sh += contrib_shared[i, j];
        if (K_v > 0) acc_sp += contrib_spec[i, j];
      }
      ip_vote_shared_raw[i]   = acc_sh / J;
      ip_vote_specific_raw[i] = (K_v > 0) ? (acc_sp / J) : 0;
      ip_vote_total_raw[i]    = gammaV[i] + ip_vote_shared_raw[i] + ip_vote_specific_raw[i];
    }
  }

  // Words: average contributions over items (offset reported separately)
  {
    matrix[N, L] contrib_shared = theta * beta_s_word; // N x L
    matrix[N, L] contrib_spec;
    if (K_w > 0) contrib_spec = u_word * beta_w_word;  // N x L

    for (i in 1:N) {
      real acc_sh = 0;
      real acc_sp = 0;
      for (l in 1:L) {
        acc_sh += contrib_shared[i, l];
        if (K_w > 0) acc_sp += contrib_spec[i, l];
      }
      ip_word_shared_raw[i]           = acc_sh / L;
      ip_word_specific_raw[i]         = (K_w > 0) ? (acc_sp / L) : 0;
      ip_word_total_raw[i]            = ip_word_shared_raw[i] + ip_word_specific_raw[i];
      ip_word_total_w_offset_raw[i]   = ip_word_total_raw[i] + zeta * z_log_tokens[i];
    }
  }

  // Joint shared = average of shared contributions from both batteries
  for (i in 1:N)
    ip_shared_joint_raw[i] = 0.5 * (ip_vote_shared_raw[i] + ip_word_shared_raw[i]);

  // Z-score across legislators (mean 0, sd 1) for comparability
  vector[N] ip_vote_shared          = ip_vote_shared_raw;
  vector[N] ip_vote_specific        = ip_vote_specific_raw;
  vector[N] ip_vote_total           = ip_vote_total_raw;

  vector[N] ip_word_shared          = ip_word_shared_raw;
  vector[N] ip_word_specific        = ip_word_specific_raw;
  vector[N] ip_word_total           = ip_word_total_raw;
  vector[N] ip_word_total_w_offset  = ip_word_total_w_offset_raw;

  vector[N] ip_shared_joint         = ip_shared_joint_raw;

  {
    real m; real s;
    m = mean(ip_vote_shared); s = sd(ip_vote_shared); if (s <= 1e-12) s = 1;
    ip_vote_shared = (ip_vote_shared - rep_vector(m, N)) / s;

    m = mean(ip_vote_specific); s = sd(ip_vote_specific); if (s <= 1e-12) s = 1;
    ip_vote_specific = (ip_vote_specific - rep_vector(m, N)) / s;

    m = mean(ip_vote_total); s = sd(ip_vote_total); if (s <= 1e-12) s = 1;
    ip_vote_total = (ip_vote_total - rep_vector(m, N)) / s;

    m = mean(ip_word_shared); s = sd(ip_word_shared); if (s <= 1e-12) s = 1;
    ip_word_shared = (ip_word_shared - rep_vector(m, N)) / s;

    m = mean(ip_word_specific); s = sd(ip_word_specific); if (s <= 1e-12) s = 1;
    ip_word_specific = (ip_word_specific - rep_vector(m, N)) / s;

    m = mean(ip_word_total); s = sd(ip_word_total); if (s <= 1e-12) s = 1;
    ip_word_total = (ip_word_total - rep_vector(m, N)) / s;

    m = mean(ip_word_total_w_offset); s = sd(ip_word_total_w_offset); if (s <= 1e-12) s = 1;
    ip_word_total_w_offset = (ip_word_total_w_offset - rep_vector(m, N)) / s;

    m = mean(ip_shared_joint); s = sd(ip_shared_joint); if (s <= 1e-12) s = 1;
    ip_shared_joint = (ip_shared_joint - rep_vector(m, N)) / s;
  }

  // ---------- Dimension weights (rowwise energy, normalized) ----------
  // Votes: shared & specific
  vector[K_s] dim_energy_vote_shared = rep_vector(0, K_s);
  vector[K_s] dim_weight_vote_shared;
  for (k in 1:K_s)
    for (j in 1:J)
      dim_energy_vote_shared[k] += square(a_s_vote[k, j]);
  {
    real tot = sum(dim_energy_vote_shared);
    if (tot <= 1e-16) tot = 1;
    dim_weight_vote_shared = dim_energy_vote_shared / tot;
  }

  vector[K_v] dim_energy_vote_specific = rep_vector(0, K_v);
  vector[K_v] dim_weight_vote_specific;
  if (K_v > 0) {
    for (k in 1:K_v)
      for (j in 1:J)
        dim_energy_vote_specific[k] += square(a_v_vote[k, j]);
    {
      real tot = sum(dim_energy_vote_specific);
      if (tot <= 1e-16) tot = 1;
      dim_weight_vote_specific = dim_energy_vote_specific / tot;
    }
  }

  // Words: shared & specific
  vector[K_s] dim_energy_word_shared = rep_vector(0, K_s);
  vector[K_s] dim_weight_word_shared;
  for (k in 1:K_s)
    for (l in 1:L)
      dim_energy_word_shared[k] += square(beta_s_word[k, l]);
  {
    real tot = sum(dim_energy_word_shared);
    if (tot <= 1e-16) tot = 1;
    dim_weight_word_shared = dim_energy_word_shared / tot;
  }

  vector[K_w] dim_energy_word_specific = rep_vector(0, K_w);
  vector[K_w] dim_weight_word_specific;
  if (K_w > 0) {
    for (k in 1:K_w)
      for (l in 1:L)
        dim_energy_word_specific[k] += square(beta_w_word[k, l]);
    {
      real tot = sum(dim_energy_word_specific);
      if (tot <= 1e-16) tot = 1;
      dim_weight_word_specific = dim_energy_word_specific / tot;
    }
  }

  // Joint shared across batteries
  vector[K_s] dim_energy_shared_joint;
  vector[K_s] dim_weight_shared_joint;
  for (k in 1:K_s)
    dim_energy_shared_joint[k] = dim_energy_vote_shared[k] + dim_energy_word_shared[k];
  {
    real tot = sum(dim_energy_shared_joint);
    if (tot <= 1e-16) tot = 1;
    dim_weight_shared_joint = dim_energy_shared_joint / tot;
  }
}

