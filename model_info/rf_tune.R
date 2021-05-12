# RF tuning ----

# Load package(s) ----
library(tidyverse)
library(tictoc)
library(rsample)
library(tidymodels)
library(ranger)

# set seed
set.seed(3013)

# load required objects ----
load("bl_setup.rda")

# Define model
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 10)))

# Grid
rf_grid <- grid_regular(rf_params, levels = 5)

# Workflow
rf_wf <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(bl_recipe)

# tuning and fitting - DON"T RUN HERE
tic("Random Forest Model")
rf_tuned <- rf_wf %>% 
  tune_grid(bl_folds, grid = rf_grid)

toc(log = TRUE)

# save runtime info
rf_runtime <- tic.log(format = TRUE)

# fit to train data
rf_wf_tune <- rf_wf %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

rf_results <- fit(rf_wf_tune, bl_train)

final_results <- rf_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>% 
  rename("Id" = id,
         "Predicted" = .pred)

final_results <- final_results[c(2,1)]

write_csv(final_results, path="Lin_Kylie_RF_RegComp.csv")


# Write out results & workflow
save(rf_tuned, rf_wf, rf_runtime, final_results, file = "rf_tuned.rda")

