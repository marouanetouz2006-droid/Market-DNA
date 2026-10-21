packages <- c("readr", "dplyr", "sandwich", "lmtest")
new <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new) > 0) install.packages(new)

library(readr)
library(dplyr)
library(sandwich)
library(lmtest)

df <- read_csv("../data/processed/research_dataset.csv", show_col_types = FALSE)
df$Date <- as.Date(df$Date)

# H1 — future volatility
h1 <- df %>% filter(!is.na(future_RV_20), !is.na(MIDI), !is.na(VIX_LEVEL), !is.na(past_RV_20))
m1 <- lm(future_RV_20 ~ MIDI + VIX_LEVEL + past_RV_20, data=h1)
print(coeftest(m1, vcov.=NeweyWest(m1, lag=19, prewhite=FALSE, adjust=TRUE)))

# H2 — future average correlation
h2 <- df %>% filter(!is.na(future_avg_corr_20), !is.na(MIDI), !is.na(VIX_LEVEL), !is.na(past_RV_20))
m2 <- lm(future_avg_corr_20 ~ MIDI + VIX_LEVEL + past_RV_20, data=h2)
print(coeftest(m2, vcov.=NeweyWest(m2, lag=19, prewhite=FALSE, adjust=TRUE)))

# H3 — drawdown risk
h3 <- df %>% filter(!is.na(crash_20d_5pct), !is.na(MIDI), !is.na(VIX_LEVEL), !is.na(past_RV_20))
m3 <- glm(crash_20d_5pct ~ MIDI + VIX_LEVEL + past_RV_20, family=binomial(link="logit"), data=h3)
print(summary(m3))

# H4 — holdout comparison (2022+)
h <- h1 %>% filter(Date >= as.Date("2022-01-01"))
a <- lm(future_RV_20 ~ VIX_LEVEL, data=h)
b <- lm(future_RV_20 ~ VIX_LEVEL + past_RV_20, data=h)
c <- lm(future_RV_20 ~ VIX_LEVEL + past_RV_20 + MIDI, data=h)
print(data.frame(
  Model=c("VIX", "VIX+PastRV", "VIX+PastRV+MIDI"),
  AdjR2=c(summary(a)$adj.r.squared, summary(b)$adj.r.squared, summary(c)$adj.r.squared),
  AIC=c(AIC(a), AIC(b), AIC(c))
))
# ============================================================
# STEP 5 — Change in MIDI
# ============================================================

df <- df %>%
  arrange(Date) %>%
  mutate(
    dMIDI_20 = MIDI - lag(MIDI, 20)
  )

delta_h1 <- df %>%
  filter(
    !is.na(future_RV_20),
    !is.na(MIDI),
    !is.na(dMIDI_20),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )

delta_model <- lm(
  future_RV_20 ~ MIDI + dMIDI_20 + VIX_LEVEL + past_RV_20,
  data = delta_h1
)

print("STEP 5 — MIDI level + 20-day change in MIDI")

print(
  coeftest(
    delta_model,
    vcov. = NeweyWest(
      delta_model,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)
# ============================================================
# STEP 6 — TRUE OUT-OF-SAMPLE TEST
# Train: 2015-2021
# Test : 2022-2026
# ============================================================

oos_df <- df %>%
  filter(
    !is.na(future_RV_20),
    !is.na(MIDI),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )

train <- oos_df %>%
  filter(
    Date >= as.Date("2015-01-01"),
    Date <= as.Date("2021-12-31")
  )

test <- oos_df %>%
  filter(
    Date >= as.Date("2022-01-01")
  )

# Model A: VIX only
model_A <- lm(
  future_RV_20 ~ VIX_LEVEL,
  data = train
)

# Model B: VIX + past volatility
model_B <- lm(
  future_RV_20 ~ VIX_LEVEL + past_RV_20,
  data = train
)

# Model C: VIX + past volatility + MIDI
model_C <- lm(
  future_RV_20 ~ VIX_LEVEL + past_RV_20 + MIDI,
  data = train
)

# Out-of-sample predictions
test$pred_A <- predict(model_A, newdata = test)
test$pred_B <- predict(model_B, newdata = test)
test$pred_C <- predict(model_C, newdata = test)

# Evaluation functions
RMSE <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

MAE <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

# Results
results_oos <- data.frame(
  Model = c(
    "VIX",
    "VIX + PastRV",
    "VIX + PastRV + MIDI"
  ),
  
  RMSE = c(
    RMSE(test$future_RV_20, test$pred_A),
    RMSE(test$future_RV_20, test$pred_B),
    RMSE(test$future_RV_20, test$pred_C)
  ),
  
  MAE = c(
    MAE(test$future_RV_20, test$pred_A),
    MAE(test$future_RV_20, test$pred_B),
    MAE(test$future_RV_20, test$pred_C)
  )
)

print("STEP 6 — TRUE OUT-OF-SAMPLE RESULTS")
print(results_oos)

# Incremental out-of-sample R² relative to VIX-only model
SSE_A <- sum(
  (test$future_RV_20 - test$pred_A)^2,
  na.rm = TRUE
)

SSE_B <- sum(
  (test$future_RV_20 - test$pred_B)^2,
  na.rm = TRUE
)

SSE_C <- sum(
  (test$future_RV_20 - test$pred_C)^2,
  na.rm = TRUE
)

OOS_R2_B_vs_A <- 1 - SSE_B / SSE_A
OOS_R2_C_vs_A <- 1 - SSE_C / SSE_A

print(
  paste(
    "OOS R2 — Model B vs VIX:",
    round(OOS_R2_B_vs_A, 4)
  )
)

print(
  paste(
    "OOS R2 — Model C vs VIX:",
    round(OOS_R2_C_vs_A, 4)
  )
)

# ============================================================
# STEP 8 — MIDI v1 vs Dynamic MIDI
# TRUE OUT-OF-SAMPLE COMPARISON
# Train: 2015-2021
# Test : 2022-2026
# ============================================================

df2 <- read_csv(
  "../data/processed/research_dataset_v2.csv",
  show_col_types = FALSE
)

df2$Date <- as.Date(df2$Date)

oos2 <- df2 %>%
  filter(
    !is.na(future_RV_20),
    !is.na(MIDI),
    !is.na(DMIDI),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )

train2 <- oos2 %>%
  filter(
    Date >= as.Date("2015-01-01"),
    Date <= as.Date("2021-12-31")
  )

test2 <- oos2 %>%
  filter(
    Date >= as.Date("2022-01-01")
  )


# ------------------------------------------------------------
# Models estimated only on 2015-2021
# ------------------------------------------------------------

model_VIX <- lm(
  future_RV_20 ~ VIX_LEVEL,
  data = train2
)

model_BASE <- lm(
  future_RV_20 ~ VIX_LEVEL + past_RV_20,
  data = train2
)

model_MIDI <- lm(
  future_RV_20 ~ VIX_LEVEL + past_RV_20 + MIDI,
  data = train2
)

model_DMIDI <- lm(
  future_RV_20 ~ VIX_LEVEL + past_RV_20 + DMIDI,
  data = train2
)


# ------------------------------------------------------------
# Predictions on unseen 2022-2026 data
# ------------------------------------------------------------

test2$pred_VIX <- predict(
  model_VIX,
  newdata = test2
)

test2$pred_BASE <- predict(
  model_BASE,
  newdata = test2
)

test2$pred_MIDI <- predict(
  model_MIDI,
  newdata = test2
)

test2$pred_DMIDI <- predict(
  model_DMIDI,
  newdata = test2
)


# ------------------------------------------------------------
# Evaluation functions
# ------------------------------------------------------------

RMSE <- function(actual, predicted) {
  sqrt(
    mean(
      (actual - predicted)^2,
      na.rm = TRUE
    )
  )
}

MAE <- function(actual, predicted) {
  mean(
    abs(actual - predicted),
    na.rm = TRUE
  )
}


# ------------------------------------------------------------
# RMSE + MAE
# ------------------------------------------------------------

results_dynamic <- data.frame(
  
  Model = c(
    "VIX",
    "VIX + PastRV",
    "VIX + PastRV + MIDI",
    "VIX + PastRV + DMIDI"
  ),
  
  RMSE = c(
    RMSE(test2$future_RV_20, test2$pred_VIX),
    RMSE(test2$future_RV_20, test2$pred_BASE),
    RMSE(test2$future_RV_20, test2$pred_MIDI),
    RMSE(test2$future_RV_20, test2$pred_DMIDI)
  ),
  
  MAE = c(
    MAE(test2$future_RV_20, test2$pred_VIX),
    MAE(test2$future_RV_20, test2$pred_BASE),
    MAE(test2$future_RV_20, test2$pred_MIDI),
    MAE(test2$future_RV_20, test2$pred_DMIDI)
  )
)

print(
  "STEP 8 — MIDI vs DMIDI OUT-OF-SAMPLE"
)

print(
  results_dynamic
)


# ------------------------------------------------------------
# OOS R2 relative to VIX-only benchmark
# ------------------------------------------------------------

SSE_VIX <- sum(
  (test2$future_RV_20 - test2$pred_VIX)^2,
  na.rm = TRUE
)

SSE_MIDI <- sum(
  (test2$future_RV_20 - test2$pred_MIDI)^2,
  na.rm = TRUE
)

SSE_DMIDI <- sum(
  (test2$future_RV_20 - test2$pred_DMIDI)^2,
  na.rm = TRUE
)


OOS_R2_MIDI <- 1 - SSE_MIDI / SSE_VIX

OOS_R2_DMIDI <- 1 - SSE_DMIDI / SSE_VIX


print(
  paste(
    "OOS R2 — MIDI vs VIX:",
    round(OOS_R2_MIDI, 4)
  )
)

print(
  paste(
    "OOS R2 — DMIDI vs VIX:",
    round(OOS_R2_DMIDI, 4)
  )
)
# ============================================================
# STEP 10 — NONLINEAR QUANT ANALYSIS
# ============================================================

dfq <- read_csv(
  "../data/processed/quant_research_dataset.csv",
  show_col_types = FALSE
)

dfq$Date <- as.Date(dfq$Date)


# ------------------------------------------------------------
# Training thresholds: 2015-2021
# Avoid defining regimes using future 2022-2026 data
# ------------------------------------------------------------

training_dmidi <- dfq %>%
  filter(
    Date >= as.Date("2015-01-01"),
    Date <= as.Date("2021-12-31"),
    !is.na(DMIDI)
  )


p20 <- quantile(
  training_dmidi$DMIDI,
  0.20,
  na.rm = TRUE
)

p80 <- quantile(
  training_dmidi$DMIDI,
  0.80,
  na.rm = TRUE
)

p90 <- quantile(
  training_dmidi$DMIDI,
  0.90,
  na.rm = TRUE
)


print(
  paste(
    "DMIDI 20th percentile:",
    round(p20, 3)
  )
)

print(
  paste(
    "DMIDI 80th percentile:",
    round(p80, 3)
  )
)

print(
  paste(
    "DMIDI 90th percentile:",
    round(p90, 3)
  )
)


# ------------------------------------------------------------
# Create nonlinear variables
# ------------------------------------------------------------

dfq <- dfq %>%
  arrange(Date) %>%
  mutate(
    
    DMIDI2 = DMIDI^2,
    
    HIGH_DMIDI = ifelse(
      DMIDI >= p90,
      1,
      0
    ),
    
    DMIDI_REGIME = case_when(
      
      DMIDI <= p20 ~ "Low",
      
      DMIDI >= p80 ~ "High",
      
      TRUE ~ "Middle"
    )
  )


dfq$DMIDI_REGIME <- factor(
  dfq$DMIDI_REGIME,
  levels = c(
    "Middle",
    "Low",
    "High"
  )
)


# ============================================================
# TEST 1 — QUADRATIC DMIDI AND FUTURE VOLATILITY
# ============================================================

quad_vol_data <- dfq %>%
  filter(
    !is.na(future_RV_20),
    !is.na(DMIDI),
    !is.na(DMIDI2),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )


quad_vol <- lm(
  future_RV_20 ~
    DMIDI +
    DMIDI2 +
    VIX_LEVEL +
    past_RV_20,
  data = quad_vol_data
)


print(
  "STEP 10A — QUADRATIC DMIDI → FUTURE VOLATILITY"
)

print(
  coeftest(
    quad_vol,
    vcov. = NeweyWest(
      quad_vol,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)


# Turning point of quadratic curve

b1 <- coef(quad_vol)["DMIDI"]
b2 <- coef(quad_vol)["DMIDI2"]

turning_point <- -b1 / (2 * b2)

print(
  paste(
    "Estimated DMIDI turning point:",
    round(turning_point, 3)
  )
)


# ============================================================
# TEST 2 — TOP 10% DMIDI REGIME
# ============================================================

high_vol_data <- dfq %>%
  filter(
    !is.na(future_RV_20),
    !is.na(HIGH_DMIDI),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )


high_vol <- lm(
  future_RV_20 ~
    HIGH_DMIDI +
    VIX_LEVEL +
    past_RV_20,
  data = high_vol_data
)


print(
  "STEP 10B — TOP 10% DMIDI → FUTURE VOLATILITY"
)

print(
  coeftest(
    high_vol,
    vcov. = NeweyWest(
      high_vol,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)


# ============================================================
# TEST 3 — LOW / MIDDLE / HIGH DMIDI REGIMES
# ============================================================

regime_vol_data <- dfq %>%
  filter(
    !is.na(future_RV_20),
    !is.na(DMIDI_REGIME),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )


regime_vol <- lm(
  future_RV_20 ~
    DMIDI_REGIME +
    VIX_LEVEL +
    past_RV_20,
  data = regime_vol_data
)


print(
  "STEP 10C — DMIDI REGIMES → FUTURE VOLATILITY"
)

print(
  coeftest(
    regime_vol,
    vcov. = NeweyWest(
      regime_vol,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)


# ============================================================
# SAME NONLINEAR TEST FOR FUTURE CORRELATION
# ============================================================

quad_corr_data <- dfq %>%
  filter(
    !is.na(future_avg_corr_20),
    !is.na(DMIDI),
    !is.na(DMIDI2),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )


quad_corr <- lm(
  future_avg_corr_20 ~
    DMIDI +
    DMIDI2 +
    VIX_LEVEL +
    past_RV_20,
  data = quad_corr_data
)


print(
  "STEP 10D — QUADRATIC DMIDI → FUTURE CORRELATION"
)

print(
  coeftest(
    quad_corr,
    vcov. = NeweyWest(
      quad_corr,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)


# ============================================================
# DESCRIPTIVE REGIME TABLE
# ============================================================

regime_summary <- dfq %>%
  filter(
    !is.na(DMIDI_REGIME)
  ) %>%
  group_by(
    DMIDI_REGIME
  ) %>%
  summarise(
    
    Observations = n(),
    
    Mean_DMIDI = mean(
      DMIDI,
      na.rm = TRUE
    ),
    
    Future_Vol_20 = mean(
      future_RV_20,
      na.rm = TRUE
    ),
    
    Future_Corr_20 = mean(
      future_avg_corr_20,
      na.rm = TRUE
    ),
    
    Future_Return_20 = mean(
      future_return_20,
      na.rm = TRUE
    ),
    
    Crash_Probability = mean(
      crash_20d_5pct,
      na.rm = TRUE
    )
  )


print(
  "STEP 10E — REGIME SUMMARY"
)

print(
  regime_summary
)

# ============================================================
# STEP 11 — TRUE OUT-OF-SAMPLE TEST:
# DMIDI AND FUTURE CROSS-ASSET CORRELATION
#
# Train: 2015-2021
# Test : 2022-2026
# ============================================================

df11 <- read_csv(
  "../data/processed/quant_research_dataset.csv",
  show_col_types = FALSE
)

df11$Date <- as.Date(df11$Date)

assets <- c(
  "QQQ",
  "XLF",
  "XLE",
  "FEZ",
  "TLT",
  "GLD"
)


# ============================================================
# 11A — BUILD PAST 20-DAY AVERAGE CORRELATION
# ============================================================

df11$past_avg_corr_20 <- NA_real_

for (i in 20:nrow(df11)) {
  
  window <- as.matrix(
    df11[
      (i - 19):i,
      assets
    ]
  )
  
  if (
    nrow(window) == 20 &&
    all(complete.cases(window))
  ) {
    
    corr_matrix <- cor(window)
    
    pairwise_corr <- corr_matrix[
      upper.tri(corr_matrix)
    ]
    
    df11$past_avg_corr_20[i] <- mean(
      pairwise_corr,
      na.rm = TRUE
    )
  }
}


# ============================================================
# CLEAN SAMPLE
# ============================================================

oos_corr <- df11 %>%
  filter(
    !is.na(future_avg_corr_20),
    !is.na(past_avg_corr_20),
    !is.na(DMIDI),
    !is.na(VIX_LEVEL),
    !is.na(past_RV_20)
  )


train_corr <- oos_corr %>%
  filter(
    Date >= as.Date("2015-01-01"),
    Date <= as.Date("2021-12-31")
  )


test_corr <- oos_corr %>%
  filter(
    Date >= as.Date("2022-01-01")
  )


# ============================================================
# CENTER DMIDI USING TRAINING DATA ONLY
# ============================================================

dmidi_mean_train <- mean(
  train_corr$DMIDI,
  na.rm = TRUE
)

train_corr <- train_corr %>%
  mutate(
    DMIDI_C = DMIDI - dmidi_mean_train,
    DMIDI_C2 = DMIDI_C^2
  )

test_corr <- test_corr %>%
  mutate(
    DMIDI_C = DMIDI - dmidi_mean_train,
    DMIDI_C2 = DMIDI_C^2
  )


# ============================================================
# MODELS — ESTIMATED ONLY ON 2015-2021
# ============================================================

# Model A:
# correlation persistence only

corr_A <- lm(
  future_avg_corr_20 ~
    past_avg_corr_20,
  data = train_corr
)


# Model B:
# stronger benchmark

corr_B <- lm(
  future_avg_corr_20 ~
    past_avg_corr_20 +
    VIX_LEVEL +
    past_RV_20,
  data = train_corr
)


# Model C:
# benchmark + linear DMIDI

corr_C <- lm(
  future_avg_corr_20 ~
    past_avg_corr_20 +
    VIX_LEVEL +
    past_RV_20 +
    DMIDI_C,
  data = train_corr
)


# Model D:
# benchmark + nonlinear DMIDI

corr_D <- lm(
  future_avg_corr_20 ~
    past_avg_corr_20 +
    VIX_LEVEL +
    past_RV_20 +
    DMIDI_C +
    DMIDI_C2,
  data = train_corr
)


# ============================================================
# TRAINING-SAMPLE NEWey-WEST RESULTS
# ============================================================

print(
  "STEP 11A — TRAINING QUADRATIC DMIDI MODEL"
)

print(
  coeftest(
    corr_D,
    vcov. = NeweyWest(
      corr_D,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)


# ============================================================
# OUT-OF-SAMPLE PREDICTIONS — 2022-2026
# ============================================================

test_corr$pred_A <- predict(
  corr_A,
  newdata = test_corr
)

test_corr$pred_B <- predict(
  corr_B,
  newdata = test_corr
)

test_corr$pred_C <- predict(
  corr_C,
  newdata = test_corr
)

test_corr$pred_D <- predict(
  corr_D,
  newdata = test_corr
)


# ============================================================
# METRICS
# ============================================================

RMSE <- function(actual, predicted) {
  
  sqrt(
    mean(
      (actual - predicted)^2,
      na.rm = TRUE
    )
  )
}


MAE <- function(actual, predicted) {
  
  mean(
    abs(
      actual - predicted
    ),
    na.rm = TRUE
  )
}


corr_results <- data.frame(
  
  Model = c(
    "Past correlation",
    "Baseline",
    "Baseline + DMIDI",
    "Baseline + DMIDI + DMIDI2"
  ),
  
  RMSE = c(
    RMSE(
      test_corr$future_avg_corr_20,
      test_corr$pred_A
    ),
    
    RMSE(
      test_corr$future_avg_corr_20,
      test_corr$pred_B
    ),
    
    RMSE(
      test_corr$future_avg_corr_20,
      test_corr$pred_C
    ),
    
    RMSE(
      test_corr$future_avg_corr_20,
      test_corr$pred_D
    )
  ),
  
  MAE = c(
    MAE(
      test_corr$future_avg_corr_20,
      test_corr$pred_A
    ),
    
    MAE(
      test_corr$future_avg_corr_20,
      test_corr$pred_B
    ),
    
    MAE(
      test_corr$future_avg_corr_20,
      test_corr$pred_C
    ),
    
    MAE(
      test_corr$future_avg_corr_20,
      test_corr$pred_D
    )
  )
)


print(
  "STEP 11B — OUT-OF-SAMPLE CORRELATION FORECAST"
)

print(
  corr_results
)


# ============================================================
# OOS R² RELATIVE TO STRONG BASELINE (MODEL B)
# ============================================================

actual_corr <- test_corr$future_avg_corr_20


SSE_B <- sum(
  (
    actual_corr -
      test_corr$pred_B
  )^2,
  na.rm = TRUE
)


SSE_C <- sum(
  (
    actual_corr -
      test_corr$pred_C
  )^2,
  na.rm = TRUE
)


SSE_D <- sum(
  (
    actual_corr -
      test_corr$pred_D
  )^2,
  na.rm = TRUE
)


OOS_R2_LINEAR <- (
  1 - SSE_C / SSE_B
)


OOS_R2_QUADRATIC <- (
  1 - SSE_D / SSE_B
)


print(
  paste(
    "OOS R2 — Linear DMIDI vs baseline:",
    round(
      OOS_R2_LINEAR,
      4
    )
  )
)


print(
  paste(
    "OOS R2 — Quadratic DMIDI vs baseline:",
    round(
      OOS_R2_QUADRATIC,
      4
    )
  )
)


# ============================================================
# 11C — HAC TEST OF FORECAST ERROR IMPROVEMENT
#
# Positive loss_difference means:
# quadratic DMIDI has lower squared forecast error
# ============================================================

test_corr$loss_difference <- (
  (
    actual_corr -
      test_corr$pred_B
  )^2
  -
    (
      actual_corr -
        test_corr$pred_D
    )^2
)


loss_test <- lm(
  loss_difference ~ 1,
  data = test_corr
)


print(
  "STEP 11C — FORECAST ERROR IMPROVEMENT TEST"
)


print(
  coeftest(
    loss_test,
    vcov. = NeweyWest(
      loss_test,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)

# ============================================================
# STEP 12 — EXPANDING-WINDOW OUT-OF-SAMPLE VALIDATION
# Target: future 20-day cross-asset correlation
#
# Purged design:
# last 20 trading days before each test year are excluded
# from training to avoid forward-window leakage.
# ============================================================

df12 <- read_csv(
  "../data/processed/quant_research_dataset.csv",
  show_col_types = FALSE
)

df12$Date <- as.Date(df12$Date)

assets <- c(
  "QQQ",
  "XLF",
  "XLE",
  "FEZ",
  "TLT",
  "GLD"
)

df12 <- df12 %>%
  arrange(Date) %>%
  mutate(
    ROW_ID = row_number()
  )


# ============================================================
# BUILD PAST 20-DAY AVERAGE CORRELATION
# ============================================================

df12$past_avg_corr_20 <- NA_real_

for (i in 20:nrow(df12)) {
  
  window <- as.matrix(
    df12[
      (i - 19):i,
      assets
    ]
  )
  
  if (
    nrow(window) == 20 &&
    all(complete.cases(window))
  ) {
    
    corr_matrix <- cor(window)
    
    pairwise_corr <- corr_matrix[
      upper.tri(corr_matrix)
    ]
    
    df12$past_avg_corr_20[i] <- mean(
      pairwise_corr,
      na.rm = TRUE
    )
  }
}


# ============================================================
# ELIGIBLE OBSERVATIONS
# ============================================================

df12$eligible <- complete.cases(
  df12[
    c(
      "future_avg_corr_20",
      "past_avg_corr_20",
      "DMIDI",
      "VIX_LEVEL",
      "past_RV_20"
    )
  ]
)


# ============================================================
# TEST YEARS
# ============================================================

test_years <- 2018:2026

yearly_results <- data.frame()

all_predictions <- data.frame()


# ============================================================
# EXPANDING-WINDOW LOOP
# ============================================================

for (yr in test_years) {
  
  test_rows_all <- df12 %>%
    filter(
      format(Date, "%Y") == as.character(yr),
      eligible
    )
  
  if (nrow(test_rows_all) == 0) {
    next
  }
  
  first_test_row <- min(
    test_rows_all$ROW_ID
  )
  
  # Purge 20 trading days:
  # training outcome windows must finish before test begins
  max_train_row <- first_test_row - 21
  
  
  train <- df12 %>%
    filter(
      Date >= as.Date("2015-01-01"),
      ROW_ID <= max_train_row,
      eligible
    )
  
  
  test <- test_rows_all
  
  
  if (
    nrow(train) < 250 ||
    nrow(test) < 30
  ) {
    next
  }
  
  
  # ----------------------------------------------------------
  # Center DMIDI using TRAINING DATA ONLY
  # ----------------------------------------------------------
  
  dmidi_mean_train <- mean(
    train$DMIDI,
    na.rm = TRUE
  )
  
  train <- train %>%
    mutate(
      DMIDI_C = DMIDI - dmidi_mean_train,
      DMIDI_C2 = DMIDI_C^2
    )
  
  test <- test %>%
    mutate(
      DMIDI_C = DMIDI - dmidi_mean_train,
      DMIDI_C2 = DMIDI_C^2
    )
  
  
  # ----------------------------------------------------------
  # BASELINE MODEL
  # ----------------------------------------------------------
  
  model_base <- lm(
    future_avg_corr_20 ~
      past_avg_corr_20 +
      VIX_LEVEL +
      past_RV_20,
    data = train
  )
  
  
  # ----------------------------------------------------------
  # QUADRATIC DMIDI MODEL
  # ----------------------------------------------------------
  
  model_dmidi <- lm(
    future_avg_corr_20 ~
      past_avg_corr_20 +
      VIX_LEVEL +
      past_RV_20 +
      DMIDI_C +
      DMIDI_C2,
    data = train
  )
  
  
  # ----------------------------------------------------------
  # OUT-OF-SAMPLE PREDICTIONS
  # ----------------------------------------------------------
  
  pred_base <- predict(
    model_base,
    newdata = test
  )
  
  pred_dmidi <- predict(
    model_dmidi,
    newdata = test
  )
  
  
  actual <- test$future_avg_corr_20
  
  
  # ----------------------------------------------------------
  # METRICS
  # ----------------------------------------------------------
  
  SSE_base <- sum(
    (actual - pred_base)^2,
    na.rm = TRUE
  )
  
  SSE_dmidi <- sum(
    (actual - pred_dmidi)^2,
    na.rm = TRUE
  )
  
  RMSE_base <- sqrt(
    mean(
      (actual - pred_base)^2,
      na.rm = TRUE
    )
  )
  
  RMSE_dmidi <- sqrt(
    mean(
      (actual - pred_dmidi)^2,
      na.rm = TRUE
    )
  )
  
  MAE_base <- mean(
    abs(
      actual - pred_base
    ),
    na.rm = TRUE
  )
  
  MAE_dmidi <- mean(
    abs(
      actual - pred_dmidi
    ),
    na.rm = TRUE
  )
  
  OOS_R2 <- 1 - SSE_dmidi / SSE_base
  
  
  yearly_results <- rbind(
    yearly_results,
    data.frame(
      Year = yr,
      Train_N = nrow(train),
      Test_N = nrow(test),
      RMSE_Baseline = RMSE_base,
      RMSE_DMIDI = RMSE_dmidi,
      MAE_Baseline = MAE_base,
      MAE_DMIDI = MAE_dmidi,
      OOS_R2 = OOS_R2
    )
  )
  
  
  fold_predictions <- data.frame(
    Date = test$Date,
    Year = yr,
    Actual = actual,
    Pred_Baseline = pred_base,
    Pred_DMIDI = pred_dmidi
  )
  
  all_predictions <- rbind(
    all_predictions,
    fold_predictions
  )
}


# ============================================================
# PRINT YEAR-BY-YEAR RESULTS
# ============================================================

print(
  "STEP 12A — YEAR-BY-YEAR EXPANDING-WINDOW RESULTS"
)

print(
  yearly_results
)


# ============================================================
# OVERALL POOLED OUT-OF-SAMPLE PERFORMANCE
# ============================================================

SSE_base_all <- sum(
  (
    all_predictions$Actual -
      all_predictions$Pred_Baseline
  )^2,
  na.rm = TRUE
)

SSE_dmidi_all <- sum(
  (
    all_predictions$Actual -
      all_predictions$Pred_DMIDI
  )^2,
  na.rm = TRUE
)

OOS_R2_all <- (
  1 -
    SSE_dmidi_all /
    SSE_base_all
)


RMSE_base_all <- sqrt(
  mean(
    (
      all_predictions$Actual -
        all_predictions$Pred_Baseline
    )^2,
    na.rm = TRUE
  )
)

RMSE_dmidi_all <- sqrt(
  mean(
    (
      all_predictions$Actual -
        all_predictions$Pred_DMIDI
    )^2,
    na.rm = TRUE
  )
)


print(
  "STEP 12B — POOLED OUT-OF-SAMPLE RESULTS"
)

print(
  paste(
    "Baseline RMSE:",
    round(
      RMSE_base_all,
      6
    )
  )
)

print(
  paste(
    "Quadratic DMIDI RMSE:",
    round(
      RMSE_dmidi_all,
      6
    )
  )
)

print(
  paste(
    "Pooled OOS R2:",
    round(
      OOS_R2_all,
      4
    )
  )
)


# ============================================================
# CONSISTENCY ACROSS YEARS
# ============================================================

positive_years <- sum(
  yearly_results$OOS_R2 > 0,
  na.rm = TRUE
)

total_years <- nrow(
  yearly_results
)


print(
  paste(
    "Years with positive OOS R2:",
    positive_years,
    "/",
    total_years
  )
)


print(
  paste(
    "Median yearly OOS R2:",
    round(
      median(
        yearly_results$OOS_R2,
        na.rm = TRUE
      ),
      4
    )
  )
)


# ============================================================
# FORECAST ERROR IMPROVEMENT TEST
# ============================================================

all_predictions$loss_difference <- (
  (
    all_predictions$Actual -
      all_predictions$Pred_Baseline
  )^2
  -
    (
      all_predictions$Actual -
        all_predictions$Pred_DMIDI
    )^2
)


loss_model_12 <- lm(
  loss_difference ~ 1,
  data = all_predictions
)


print(
  "STEP 12C — POOLED FORECAST ERROR IMPROVEMENT"
)


print(
  coeftest(
    loss_model_12,
    vcov. = NeweyWest(
      loss_model_12,
      lag = 19,
      prewhite = FALSE,
      adjust = TRUE
    )
  )
)


# ============================================================
# SAVE RESULTS FOR LINKEDIN / REPORT
# ============================================================

write.csv(
  yearly_results,
  "../output/tables/expanding_window_results.csv",
  row.names = FALSE
)

write.csv(
  all_predictions,
  "../output/tables/expanding_window_predictions.csv",
  row.names = FALSE
)