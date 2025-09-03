
# SFA: Software for Sparse Factor Analysis

This software is used to estimate the average effect of a treatment variable on an outcome, after controlling for background covariates.  Unlike the standard regression model, the proposed method combines a machine learning method to control for the background covariates while using a regression on the treatment variable of interest.  

The method, the Partially Linear Causal Effect (PLCE) model, models both the treatment and the outcome variable, returning a causal effect of the treatment on the outcome under the assumption that there are no omitted confounders and the treatment is random. The method handles random effects, incorporates a set of sensitivity analyses and, as shown in the accompanying manuscript, outperforms several existing method that use machine learning for causal inference.

For more details, see  [Kim, Londregan, Ratkovic (2018)](cite).


## Installation 

The latest version can be installed by:
```R
devtools::install_github('ratkovic/PLCE')
```


## Troubleshooting installation

The software relies on `C++` code integrated into the `R` code through the `Rcpp` package.  If the software does not run on your machine, it may be fixed by ensuring that your compilers are set up properly.

This advice below comes from `KRLS` by Hazlet and Sonnet [(link)](https://github.com/lukesonnet/KRLS).

#### Windows
If you are on Windows, you will need to install [RTools](https://cran.r-project.org/bin/windows/Rtools/) if you haven't already. If you still are having difficulty with installing and it says that the compilation failed, try installing it without support for multiple architectures:
```R
devtools::install_github('ratkovic/PLCE', args=c('--no-multiarch'))
```

#### Mac OSX

In order to compile the `C++` in this package, `RcppArmadillo` will require you to have compilers installed on your machine. You may already have these, but you can install them by running:

```bash
xcode-select --install
```

If you are having problems with this install on Mac OSX, specifically if you are getting errors with either `lgfortran` or `lquadmath`, then try open your Terminal and try the following:

```bash
curl -O http://r.research.att.com/libs/gfortran-4.8.2-darwin13.tar.bz2
sudo tar fvxz gfortran-4.8.2-darwin13.tar.bz2 -C /
```

Also see section 2.16 [here](http://dirk.eddelbuettel.com/code/rcpp/Rcpp-FAQ.pdf)
