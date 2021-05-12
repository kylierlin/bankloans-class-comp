# NN tuning ----

# Load package(s) ----
library(tidyverse)
library(tictoc)
library(rsample)
library(tidymodels)
library(xgboost)
library(e1071)
library(earth)

# set seed
set.seed(3013)

# load required objects ----
load("bl_setup.rda")

# Define Model
nn_model <- mlp(hidden_units = tune(),
                penalty = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("nnet")

nn_params <- parameters(nn_model)

# Grid
nn_grid <- grid_regular(nn_params, levels = 5)

# Workflow
nn_wf <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(bl_nn_recipe)

# tuning and fitting
tic("Single Layer Neural Network Model")
nn_tuned <- nn_wf %>% 
  tune_grid(bl_folds, grid = nn_grid)

toc(log = TRUE)

# save runtime info
nn_runtime <- tic.log(format = TRUE)

# fit to train data
nn_wf_tune <- nn_wf %>% 
  finalize_workflow(select_best(nn_tuned, metric = "rmse"))

nn_results <- fit(nn_wf_tune, bl_train)

final_results <- nn_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>% 
  rename("Id" = id,
         "Predicted" = .pred)

final_nn_results <- final_results[c(2,1)]

write_csv(final_results, path="Lin_Kylie_NN_RegComp.csv")


# Write out results & workflow
save(nn_tuned, nn_wf, nn_runtime, final_nn_results, file = "rf_tuned.rda")


