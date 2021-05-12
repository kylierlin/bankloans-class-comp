# Elastic Net tuning ----

# Load package(s) ----
library(tidyverse)
library(tictoc)
library(rsample)
library(tidymodels)
library(glmnet)

# set seed
set.seed(3013)

# load required objects ----
load("bl_setup.rda")

# Define model ----
en_model <- logistic_reg(mode = "classification",
                         penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet")


# set-up tuning grid ----
en_params <- parameters(en_model)


# define tuning grid
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_wf <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(bl_en_recipe)

# Tuning/fitting ----
tic("Elastic Net Model")
en_tuned <- en_wf %>% 
  tune_grid(bl_folds, grid = en_grid)

toc(log = TRUE)

# save runtime info
en_runtime <- tic.log(format = TRUE)

# fit to train data
en_wf_tune <- en_wf %>% 
  finalize_workflow(select_best(en_tuned, metric = "rmse"))

en_results <- fit(en_wf_tune, bl_train)

final_results <- en_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) 

# %>%
#   rename("Id" = id,
#          "Predicted" = .pred)
# 
# final_results <- final_results[c(2,1)]
# 
# write_csv(final_results, path="Lin_Kylie_EN_RegComp.csv")

# Write out results & workflow
save(en_tuned, en_wf, en_runtime, final_results, file = "en_tuned.rda")



