# SVMRD tuning ----

# Load package(s) ----
library(tidyverse)
library(tictoc)
library(rsample)
library(tidymodels)
library(e1071)

# set seed
set.seed(3013)

# load required objects ----
load("bl_setup.rda")

# Define Model
svmrd_model <-
  svm_rbf(cost = tune(),
          rbf_sigma = tune(),
          margin = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

svmrd_params <- parameters(svmrd_model)

# grid
svmrd_grid <- grid_regular(svmrd_params, levels = 5)

# workflow
svmrd_wf <- workflow() %>% 
  add_model(svmrd_model) %>% 
  add_recipe(bl_recipe)


# tuning and fitting
tic("SVM Radial Model")
svmrd_tuned <- svmrd_wf %>% 
  tune_grid(bl_folds, grid = svmrd_grid)

toc(log = TRUE)

# save runtime info
svmrd_runtime <- tic.log(format = TRUE)

# fit to train data
svmrd_wf_tune <- en_wf %>% 
  finalize_workflow(select_best(svmrd_tuned, metric = "accuracy"))

svmrd_results <- fit(svmrd_wf_tune, bl_train)

final_svmrd_results <- svmrd_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id))

# Write out results & workflow
save(svmrd_tuned, svmrd_wf, svmrd_runtime, final_svmrd_results, file = "svmrd_tuned.rda")


