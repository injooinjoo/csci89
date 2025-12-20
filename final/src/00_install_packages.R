# Install required packages for STM topic modeling
# Run this script first if packages are not installed

# Set user library path
user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
}
.libPaths(c(user_lib, .libPaths()))

packages <- c(
  "tidyverse",
  "lubridate",
  "stringr",
  "quanteda",
  "stm",
  "wordcloud",
  "ggplot2"
)

# Install packages that are not already installed
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste0("Installing: ", pkg, "\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/", lib = user_lib, dependencies = TRUE)
  } else {
    cat(paste0("Already installed: ", pkg, "\n"))
  }
}

for (pkg in packages) {
  install_if_missing(pkg)
}

cat("\n=== Package installation complete ===\n")
