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

## 8. Optional: Use Docker Compose

Create a `docker-compose.yml` file with this content:

```yaml
version: "3.9"
services:
  rstudio:
    image: mannheimsds/rstudio-stan-cache:latest
    ports:
      - "8787:8787"
    environment:
      PASSWORD: yourpw
    volumes:
      - ./SFA_Stan:/home/rstudio/SFA_Stan
```

Then launch with:

```bash
docker compose up
```

Stop with:

```bash
docker compose down
```

---

## Project Structure

```
sfa/
├── README.md            <- This file
├── Dockerfile           <- Build instructions for the image
└── SFA_Stan/            <- You
```
