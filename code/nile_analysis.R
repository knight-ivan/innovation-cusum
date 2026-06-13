#!/usr/bin/env Rscript
# =============================================================================
#  nile_analysis.R — Nile River analysis using base R + changepoint
#  Dependencies: changepoint (only external package needed)
# =============================================================================

suppressPackageStartupMessages(library(changepoint))

script_dir  <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
data_dir    <- file.path(script_dir, "data")
figures_dir <- file.path(script_dir, "figures")
results_dir <- file.path(script_dir, "results")
for (d in c(data_dir, figures_dir, results_dir))
  dir.create(d, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load and export data ───────────────────────────────────────────────────
data("Nile")
nile  <- as.numeric(Nile)
years <- 1871:1970
T     <- length(nile)   # 100

nile_df <- data.frame(year = years, flow = nile)
write.csv(nile_df, file.path(data_dir, "nile.csv"), row.names = FALSE)
cat("Exported: nile.csv  (T =", T, ")\n")

# ── 2. AR(1) fit via OLS ──────────────────────────────────────────────────────
phi_hat   <- sum(nile[-1] * nile[-T]) / sum(nile[-T]^2)
resid_ar1 <- nile[-1] - phi_hat * nile[-T]    # length 99
sigma_hat <- sd(resid_ar1)
It_const  <- abs(phi_hat) * sigma_hat / sqrt(1 - phi_hat^2)

cat(sprintf("\nAR(1) fit:  phi = %.4f,  sigma = %.4f\n", phi_hat, sigma_hat))
cat(sprintf("Stationary I_t (constant): %.4f\n", It_const))

# ── 3. Ljung-Box MDS tests on AR(1) residuals ─────────────────────────────────
lb_test <- function(x, lag = 10) {
  n   <- length(x)
  rho <- acf(x, lag.max = lag, plot = FALSE)$acf[-1]
  Q   <- n * (n + 2) * sum(rho^2 / (n - seq_along(rho)))
  pv  <- 1 - pchisq(Q, df = lag)
  list(stat = Q, pval = pv, lag = lag)
}

lb5  <- lb_test(resid_ar1, lag = 5)
lb10 <- lb_test(resid_ar1, lag = 10)

cat(sprintf("\nLjung-Box (lag=5):   Q = %.3f,  p = %.4f  %s\n",
            lb5$stat, lb5$pval, ifelse(lb5$pval > 0.05, "[PASS]", "[FAIL]")))
cat(sprintf("Ljung-Box (lag=10):  Q = %.3f,  p = %.4f  %s\n",
            lb10$stat, lb10$pval, ifelse(lb10$pval > 0.05, "[PASS]", "[FAIL]")))

mds_df <- data.frame(
  model    = c("AR(1) residuals", "AR(1) residuals"),
  lag      = c(5, 10),
  LB_stat  = round(c(lb5$stat,  lb10$stat),  3),
  LB_pval  = round(c(lb5$pval,  lb10$pval),  4),
  pass_MDS = c(lb5$pval > 0.05, lb10$pval > 0.05)
)
write.csv(mds_df, file.path(results_dir, "nile_mds_tests.csv"), row.names = FALSE)
cat("Saved: nile_mds_tests.csv\n")

# ── 4. Change-point detection ─────────────────────────────────────────────────
# (a) PELT with BIC penalty (changepoint package)
pelt_fit  <- cpt.mean(nile, method = "PELT", penalty = "MBIC")
pelt_cpts <- cpts(pelt_fit)
cat(sprintf("\nPELT (BIC) change-point(s): t = %s  (year %s)\n",
            paste(pelt_cpts, collapse = ", "),
            paste(years[pelt_cpts], collapse = ", ")))

# (b) At-most-one change (AMOC) via PELT
amoc_fit  <- cpt.mean(nile, method = "AMOC")
amoc_cpt  <- cpts(amoc_fit)
cat(sprintf("AMOC change-point:          t = %d  (year %d)\n",
            amoc_cpt, years[amoc_cpt]))

# (c) Manual CUSUM to match Python study
mu0    <- mean(nile[1:20])
sig0   <- sd(nile[1:20])
S_p    <- numeric(T);  S_m <- numeric(T)
k_ref  <- 0.5;  h_thresh <- 5.0
alarm  <- NA
for (t in 2:T) {
  z      <- (nile[t] - mu0) / sig0
  S_p[t] <- max(0, S_p[t-1] + z - k_ref)
  S_m[t] <- max(0, S_m[t-1] - z - k_ref)
  if (is.na(alarm) && (S_p[t] > h_thresh || S_m[t] > h_thresh)) alarm <- t
}
cat(sprintf("Manual CUSUM (h=5):         alarm at t = %s  (year %s)\n",
            ifelse(is.na(alarm), "none", alarm),
            ifelse(is.na(alarm), "none", years[alarm])))

cp_df <- data.frame(
  method   = c("PELT-BIC", "AMOC", "CUSUM (h=5)"),
  tau_hat  = c(paste(pelt_cpts, collapse=";"), amoc_cpt,
               ifelse(is.na(alarm), NA, alarm)),
  year_hat = c(paste(years[pelt_cpts], collapse=";"), years[amoc_cpt],
               ifelse(is.na(alarm), NA, years[alarm])),
  true_tau = 28,  true_year = 1898
)
write.csv(cp_df, file.path(results_dir, "nile_changepoint.csv"), row.names = FALSE)
cat("Saved: nile_changepoint.csv\n")

# ── 5. Figures ────────────────────────────────────────────────────────────────
# Fig 1: Nile series with change-point markers
png(file.path(figures_dir, "nile_series.png"), width = 900, height = 430, res = 120)
par(mar = c(4, 4.5, 3, 1))
plot(years, nile, type = "l", col = "steelblue", lwd = 1.6,
     main = "Nile River Annual Discharge  (1871–1970)",
     xlab = "Year", ylab = expression("Flow  (" * 10^8 ~ m^3/yr * ")"),
     las = 1)
abline(v = 1898,          col = "firebrick",  lty = 2, lwd = 1.8)
abline(v = years[amoc_cpt], col = "darkorange", lty = 3, lwd = 1.8)
legend("bottomleft", bty = "n", cex = 0.82,
       legend = c("Observed flow",
                  "Known change-point (1898)",
                  sprintf("AMOC estimate (year %d)", years[amoc_cpt])),
       col = c("steelblue","firebrick","darkorange"), lty = c(1,2,3), lwd = 1.8)
dev.off()

# Fig 2: AR(1) residuals + ACF
png(file.path(figures_dir, "nile_ar1_residuals.png"), width = 900, height = 560, res = 120)
par(mfrow = c(2,2), mar = c(4,4,3,1))

plot(years[-1], resid_ar1, type = "l", col = "steelblue", lwd = 1.2,
     main = sprintf("AR(1) Residuals  (phi=%.3f)", phi_hat),
     xlab = "Year", ylab = "Residual")
abline(h = 0, col = "gray50", lty = 2)

acf(resid_ar1, main = "ACF — AR(1) Residuals",  col = "steelblue", lag.max = 20)
pacf(resid_ar1, main = "PACF — AR(1) Residuals", col = "coral",     lag.max = 20)

# Cumulative CUSUM of standardised residuals
cs <- cumsum(resid_ar1 / sigma_hat)
plot(cs, type = "l", col = "darkorange", lwd = 1.5,
     main = "Cumul. CUSUM of Residuals",
     xlab = "Index (t-1)", ylab = "Cumulative sum")
abline(h = 0, col = "gray", lty = 2)
abline(v = which.max(abs(cs)), col = "firebrick", lty = 2, lwd = 1.5)
dev.off()

# Fig 3: CUSUM statistic trajectory
png(file.path(figures_dir, "nile_cusum_test.png"), width = 800, height = 420, res = 120)
par(mar = c(4, 4, 3, 1))
plot(years, S_p, type = "l", col = "steelblue", lwd = 1.5, ylim = range(c(S_p, S_m)),
     main = "Page CUSUM — Nile River", xlab = "Year", ylab = "CUSUM statistic")
lines(years, S_m, col = "coral", lwd = 1.5)
abline(h = h_thresh, col = "gray40", lty = 2)
abline(v = 1898, col = "firebrick", lty = 2, lwd = 1.5)
if (!is.na(alarm)) abline(v = years[alarm], col = "darkorange", lty = 3, lwd = 1.5)
legend("topright", bty = "n", cex = 0.82,
       legend = c(expression(S[t]^"+"), expression(S[t]^"-"),
                  "Threshold h=5", "True CP (1898)",
                  sprintf("Alarm year %d", ifelse(is.na(alarm), NA, years[alarm]))),
       col = c("steelblue","coral","gray40","firebrick","darkorange"),
       lty = c(1,1,2,2,3), lwd = 1.5)
dev.off()

cat("\nNile analysis complete.\n")
cat(sprintf("  AR(1) phi = %.4f  |  sigma = %.4f  |  I_t(const) = %.4f\n",
            phi_hat, sigma_hat, It_const))
cat(sprintf("  LB(5) p = %.4f  |  LB(10) p = %.4f\n", lb5$pval, lb10$pval))
