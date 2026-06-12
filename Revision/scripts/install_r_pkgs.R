.libPaths(c("/home/sunj107/scratch/Rlibs", .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"),
        Ncpus = max(1, parallel::detectCores()-1))
cran <- c("knockoff","vegan","uwot","ggplot2","ggpubr","UpSetR",
          "dplyr","tidyr","readxl","matrixStats","reshape2","RColorBrewer")
have <- rownames(installed.packages())
need <- setdiff(cran, have)
cat("Installing CRAN:", paste(need, collapse=", "), "\n")
if (length(need)) install.packages(need, lib="/home/sunj107/scratch/Rlibs")
# sva from Bioconductor
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager", lib="/home/sunj107/scratch/Rlibs")
library(BiocManager)
if (!requireNamespace("sva", quietly=TRUE))
  BiocManager::install("sva", lib="/home/sunj107/scratch/Rlibs", update=FALSE, ask=FALSE)
cat("DONE. Final check:\n")
for (p in c(cran,"sva")) cat(sprintf("  %-12s %s\n", p, requireNamespace(p, quietly=TRUE)))
