# =========================================================
# RStudio + Stan (CmdStanR) + rstan/brms/bayesplot/posterior
# Multi-arch friendly (amd64 + arm64)
# Base image: rocker/rstudio:4
# =========================================================
FROM rocker/rstudio:4

# ---- System dependencies ----
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential g++ make git curl cmake ccache \
    pandoc \
    libcurl4-gnutls-dev libssl-dev libxml2-dev libxt-dev libgit2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev \
    libpng-dev libtiff5-dev libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- ccache + parallel compiles ----
ENV PATH="/usr/lib/ccache:${PATH}"
RUN mkdir -p /root/.R /home/rstudio/.R && \
    printf "CXX14=g++ -fPIC\nCXX17=g++ -fPIC\nMAKEFLAGS=-j4\n" | tee /root/.R/Makevars >/home/rstudio/.R/Makevars && \
    chown -R rstudio:rstudio /home/rstudio/.R

# ---- cmdstanr package ----
RUN R -q -e "install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', 'https://cloud.r-project.org'))"

# ---- Build CmdStan at image build time (as rstudio) ----
USER rstudio
RUN mkdir -p /home/rstudio/.cmdstan && \
    R -q -e "cmdstanr::install_cmdstan(dir='~/.cmdstan', cores = parallel::detectCores(logical = FALSE), overwrite = TRUE)"

# ---- Minimal additions: make stable path + env visible ----
USER root
# 1) create a stable symlink /home/rstudio/.cmdstan/cmdstan -> cmdstan-<ver>
RUN ln -s /home/rstudio/.cmdstan/cmdstan-* /home/rstudio/.cmdstan/cmdstan || true
# 2) export CMDSTAN for all sessions (R + shell)
ENV CMDSTAN=/home/rstudio/.cmdstan/cmdstan
RUN printf '\n# CmdStan path\nCMDSTAN=/home/rstudio/.cmdstan/cmdstan\n' >> /usr/local/lib/R/etc/Renviron.site && \
    echo 'CMDSTAN=/home/rstudio/.cmdstan/cmdstan' >> /home/rstudio/.Renviron && \
    echo 'CMDSTAN=/home/rstudio/.cmdstan/cmdstan' >> /root/.Renviron

# ---- R packages commonly used with Stan ----
RUN R -q -e "install.packages(c('rstan','brms','bayesplot','posterior','loo', 'statmod'), repos='https://cloud.r-project.org')"

# ---- Expose RStudio Server ----
EXPOSE 8787

# NOTE: Do NOT set USER rstudio here; rocker/s6 init must start as root.

# Code for build/push
# docker buildx build \
#  --platform linux/amd64,linux/arm64 \
#  -t mannheimsds/rstudio-stan-cache:latest \
#  --push \
#  .

