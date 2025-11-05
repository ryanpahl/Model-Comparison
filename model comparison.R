library(tidyverse)
library(AER)
library(pscl)
library(MASS)
set.seed(123)
data("PhDPublications")
data <- PhDPublications
tail(data, n=200)

# poisson distribution
plot(0:10, dpois(0:10, mean(PhDPublications$articles)), 
     type = "b", col = 2,
     xlab = "Number of articles", ylab = "Probability")
lines(0:10, prop.table(table(PhDPublications$articles))[1:11], 
      type = "b")
legend("topright", c("observed", "predicted"), 
       col = 1:2, lty = rep(1, 2), bty = "n")

# log-adjusted linear model
fm_lrm <- lm(log(articles + 0.5) ~ ., 
             data = PhDPublications)
summary(fm_lrm)
-2 * logLik(fm_lrm)
fm_prm <- glm(articles ~ ., 
              data = PhDPublications, family = poisson)

fm_nbrm <- glm.nb(articles ~ ., data = PhDPublications)

# zero inflated model

fm_zip <- zeroinfl(articles ~ . | ., 
                   data = PhDPublications)
fm_zinb <- zeroinfl(articles ~ . | ., 
                    data = PhDPublications, dist = "negbin")

AIC(fm_lrm, fm_prm, fm_zip, fm_zinb)
# The log adjusted linear model has the lowest AIC on the training data, so at first blush, that one appears to be the best fit.
# Let's split the data into train, test, and validation sets and train new models

# Total observations
n <- nrow(PhDPublications)

# Create random indices to avoid biased list order
indices <- sample(1:n)

# Split sizes (60% train, 20% validation, 20% test)
train_idx <- indices[1:round(0.6 * n)]
valid_idx <- indices[(round(0.6 * n) + 1):round(0.8 * n)]
test_idx  <- indices[(round(0.8 * n) + 1):n]

# Split datasets
train_data <- PhDPublications[train_idx, ]
valid_data <- PhDPublications[valid_idx, ]
test_data  <- PhDPublications[test_idx, ]

# 1. Linear model (log-transformed)
lm_model <- lm(log(articles + 0.5) ~ ., data = train_data)

# 2. Poisson regression
poisson_model <- glm(articles ~ ., family = poisson, data = train_data)

# 3. Negative binomial regression
nb_model <- glm.nb(articles ~ ., data = train_data)

# Predict on validation set
pred_lm_valid <- exp(predict(lm_model, newdata = valid_data)) - 0.5
pred_pois_valid <- predict(poisson_model, newdata = valid_data, type = "response")
pred_nb_valid <- predict(nb_model, newdata = valid_data, type = "response")

# Predict on test set
pred_lm_test <- exp(predict(lm_model, newdata = test_data)) - 0.5
pred_pois_test <- predict(poisson_model, newdata = test_data, type = "response")
pred_nb_test <- predict(nb_model, newdata = test_data, type = "response")

# Define functions
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
mae  <- function(actual, predicted) mean(abs(actual - predicted))

# Validation metrics
cat("=== VALIDATION SET ===\n")
cat("Linear Model (log): RMSE =", rmse(valid_data$articles, pred_lm_valid), 
    " MAE =", mae(valid_data$articles, pred_lm_valid), "\n")
cat("Poisson Model:      RMSE =", rmse(valid_data$articles, pred_pois_valid), 
    " MAE =", mae(valid_data$articles, pred_pois_valid), "\n")
cat("NegBin Model:       RMSE =", rmse(valid_data$articles, pred_nb_valid), 
    " MAE =", mae(valid_data$articles, pred_nb_valid), "\n")

# Test metrics
cat("\n=== TEST SET ===\n")
cat("Linear Model (log): RMSE =", rmse(test_data$articles, pred_lm_test), 
    " MAE =", mae(test_data$articles, pred_lm_test), "\n")
cat("Poisson Model:      RMSE =", rmse(test_data$articles, pred_pois_test), 
    " MAE =", mae(test_data$articles, pred_pois_test), "\n")
cat("NegBin Model:       RMSE =", rmse(test_data$articles, pred_nb_test), 
    " MAE =", mae(test_data$articles, pred_nb_test), "\n")

# LM Plot
plot(test_data$articles, pred_lm_test, 
     main = "Linear Model Predictions (Test Set)", 
     xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = "red")

# Poisson Plot
plot(test_data$articles, pred_pois_test, 
     main = "Linear Model Predictions (Test Set)", 
     xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = "red")

# NB Plot
plot(test_data$articles, pred_nb_test, 
     main = "Linear Model Predictions (Test Set)", 
     xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = "red")

hist(test_data$articles - pred_lm_test, main = "Linear Model Residuals", xlab = "Residuals")
hist(test_data$articles - pred_pois_test, main = "Poisson Residuals", xlab = "Residuals")
hist(test_data$articles - pred_nb_test, main = "Negative Binomial Residuals", xlab = "Residuals")

mu_hat <- exp(coef(nb_model))
theta_hat <- nb_model$theta

# Range of counts to plot
x_vals <- 0:10

# Predicted probabilities from NB model
nb_probs <- dnbinom(x_vals, size = theta_hat, mu = mu_hat)

# Observed proportions (empirical)
obs_probs <- prop.table(table(PhDPublications$articles))[1:11]

# Plot
plot(x_vals, nb_probs, type = "b", col = 2,
     xlab = "Number of articles", ylab = "Probability",
     main = "Observed vs Negative Binomial Predicted Distribution")
lines(x_vals, obs_probs, type = "b", col = 1)
legend("topright", c("Observed", "NB Predicted"), col = 1:2, lty = 1, bty = "n")
