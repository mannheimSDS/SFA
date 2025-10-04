# SFA: Sparse Factor Analysis with Stan and Docker

This repository provides a ready-to-use **Docker environment** for running **RStudio Server** with **Stan** pre-installed.
It supports running the **Sparse Factor Analysis (SFA)** model described in:

> **Kim, In Song, John Londregan, and Marc Ratkovic.**
> *Estimating Spatial Preferences from Votes and Text*
> Political Analysis, 2018. [https://doi.org/10.1017/pan.2018.7](https://doi.org/10.1017/pan.2018.7)

The Docker image includes everything needed to develop, run, and test SFA models in a reproducible, cross-platform setup.

---

## 1. Clone this repository and navigate into the directory

```bash
git clone https://github.com/mannheimsds/sfa.git
cd sfa
```


---

## 2. Pull the prebuilt Docker image

Ensure that you have Docker Desktop installed and running. [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

```bash
docker pull mannheimsds/rstudio-stan-cache:latest
```

> The image is **multi-architecture**, supporting both `amd64` (Intel/AMD) and `arm64` (Apple Silicon).

---

## 3. Run RStudio with Docker

```bash
  docker run --rm --user root \
  -p 8787:8787 \
  -e PASSWORD=yourpw \
  -e CMDSTAN=/home/rstudio/.cmdstan/cmdstan-2.37.0 \
  -v "$(pwd)/SFA_Stan:/home/rstudio/SFA_Stan" \
  mannheimsds/rstudio-stan-cache:latest
```

* Open [http://localhost:8787](http://localhost:8787) in your browser.
* Log in with:

  * **Username:** `rstudio`
  * **Password:** `yourpw`

> **macOS users:** If you don’t see files in RStudio, ensure the repo directory is allowed under
> *Docker Desktop → Settings → Resources → File Sharing*.

---

## 4. Verify Stan installation

Paste this into the **RStudio console** to confirm that Stan and `cmdstanr` are ready:

```r
library(cmdstanr)

stan_code <- "
data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  y ~ bernoulli(theta);
}
"

# Write the Stan file
stan_file_path <- write_stan_file(stan_code)

# Compile the model
mod <- cmdstan_model(stan_file_path)

# Example data
data_list <- list(
  N = 10,
  y = c(0, 1, 0, 1, 1, 0, 0, 1, 1, 0)
)

# Sample
fit <- mod$sample(
  data = data_list,
  chains = 2,
  iter_sampling = 500,
  iter_warmup = 250
)

# Print summary
fit$summary()
```

You should see a table with a parameter `theta` estimated around **0.5**, confirming that Stan is working correctly.

---

## 5. Stopping the container

Press **Ctrl + C** in the terminal where `docker run` is active.
Because we used `--rm`, the container cleans itself up automatically.

---

## 7. Troubleshooting

| Problem                         | Solution                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| **Port already in use (8787)**  | Change the left side of `-p`, e.g., `-p 8888:8787`, then open `http://localhost:8888`. |
| **No files visible in RStudio** | Ensure the repo folder is shared with Docker Desktop (macOS/Windows).                  |
| **CmdStan not initialized**     | Run `cmdstanr::install_cmdstan()` once inside RStudio to build CmdStan locally.        |
| **Slow startup on first run**   | This is normal—CmdStan compiles the model the first time. Subsequent runs are cached.  |

---

## Model (hierarchical)

**Data and notation.**  
- \(N\): legislators (respondents)  
- \(J\): vote items; \(X_{ij}\in\{0,1\}\)  
- \(L\): word items; \(Y_{i\ell}\in\{0,1,\dots,C\}\) are **global ordinal categories** (built from counts)  
- \(\text{tokens}_i>0\): document length for legislator \(i\) (used as an offset)  
- \(F_y[c]\in(0,1)\), \(c=1,\dots,C\): empirical CDF (or survival) of the global categories used to **place cutpoints**  
- \(\mathrm{F\_dir}\in\{-1,+1\}\): \(+1\) if \(F_y\) is a CDF (increasing), \(-1\) if a survival function (decreasing)

**Latent structure.**  
- Shared factors per legislator: \(\theta_i\in\mathbb{R}^{K_s}\)  
- Battery–specific factors: \(u^{V}_i\in\mathbb{R}^{K_v}\) (votes), \(u^{W}_i\in\mathbb{R}^{K_w}\) (words)  
- Loadings:  
  - votes: \(a^{S}_j\in\mathbb{R}^{K_s}\), \(a^{V}_j\in\mathbb{R}^{K_v}\) for item \(j\)  
  - words: \(\beta^{S}_\ell\in\mathbb{R}^{K_s}\), \(\beta^{W}_\ell\in\mathbb{R}^{K_w}\) for item \(\ell\)

**Person/item random effects.**  
- Vote person RE: \(\gamma^{V}_i \sim \mathcal{N}(0,\sigma_{\gamma V}^2)\), then centered so \(\sum_i \gamma^{V}_i=0\)  
- Vote item thresholds (marginalized): \(b^{V}_j \sim \mathcal{N}(0,\sigma_{bV}^2)\)  
- Word item intercepts (marginalized): \(\alpha^{W}_\ell \sim \mathcal{N}(0,\sigma_{\alpha W}^2)\)

**Offsets.**  Let \(z_i = \mathrm{zscore}\big(\log(\text{tokens}_i)\big)\) and \(\zeta\) be its slope in the word battery.

### Votes: Type-II/CJR probit with marginalized thresholds
Define the linear predictor (before the item threshold)
\[
\eta_{ij}^{(\text{no }b)} \;=\; \gamma^{V}_i \;+\; \theta_i^\top a^{S}_j \;+\; (u^V_i)^\top a^{V}_j.
\]
With probit link and a per-item threshold \(b^{V}_j\),
\[
X_{ij}\mid \cdot,\,b^{V}_j \;\sim\; \mathrm{Bernoulli}\!\left(\Phi\big(\eta_{ij}^{(\text{no }b)} - b^{V}_j\big)\right),\quad
b^{V}_j\sim\mathcal{N}(0,\sigma_{bV}^2).
\]
In the implementation, \(b^{V}_j\) is **integrated out** via Gauss–Hermite quadrature:
\[
\Pr(X_{ij}=x) \;=\; \int \Pr(X_{ij}=x\mid b)\,\phi(b;0,\sigma_{bV}^2)\,db.
\]

### Words: ordinal probit with global (shared) cutpoints
For legislator \(i\) and word item \(\ell\),
\[
v_{i\ell} \;=\; \zeta\, z_i \;+\; \theta_i^\top \beta^{S}_\ell \;+\; (u^W_i)^\top \beta^{W}_\ell \;+\; \alpha^{W}_\ell,\qquad
\alpha^{W}_\ell\sim \mathcal{N}(0,\sigma_{\alpha W}^2).
\]
Global cutpoints (common across all word items) are **data-driven via an empirical CDF warp**:
\[
\tau_c \;=\; \beta_0 \;+\; \mathrm{F\_dir}\cdot \beta_1\, \big(F_y[c]\big)^{\beta_2},
\qquad c=1,\dots,C,
\]
which guarantees monotonicity when \(\beta_1>0\) and \(\beta_2>0\).

Conditional category probabilities (with \(\alpha^{W}_\ell\) marginalized by GH in the code):
\[
\Pr(Y_{i\ell}=0)=\Phi(\tau_1 - v_{i\ell}),\quad
\Pr(Y_{i\ell}=y)=\Phi(\tau_{y+1}-v_{i\ell})-\Phi(\tau_{y}-v_{i\ell})\ (1\le y\le C-1),\quad
\Pr(Y_{i\ell}=C)=1-\Phi(\tau_{C}-v_{i\ell}).
\]

> **Computation.** Both \(b^{V}_j\) and \(\alpha^{W}_\ell\
