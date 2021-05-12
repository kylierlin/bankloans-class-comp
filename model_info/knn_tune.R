# Knn tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(kknn)

# Seed
set.seed(3013)

# load required objects ----
load("bl_setup.rda")

# Define model ----
knn_model <- nearest_neighbor(mode = "regression",
                              neighbors = tune()
  ) %>%
  set_engine("kknn")

# # check tuning parameters
# parameters(knn_model)

# set-up tuning grid ----
knn_params <- parameters(knn_model) %>%
  update(neighbors = neighbors(range = c(1,15)))

# define grid
knn_grid <- grid_regular(knn_params, levels = 15)

# workflow ----
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(bl_recipe)

# Tuning/fitting ----
knn_tune <- knn_workflow %>%
  tune_grid(
    resamples = bl_folds,
    grid = knn_grid
  )

# fit to train data
knn_wf_tune <- knn_workflow %>% 
  finalize_workflow(select_best(knn_tune, metric = "rmse"))

knn_results <- fit(knn_wf_tune, bl_train)

final_results <- knn_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>% 
  rename("Id" = id,
         "Predicted" = .pred)

write_csv(final_results, path = "Lin_Kylie_RegComp.csv")

# final_results %>% 
#   rename("Id" = id,
#          "Predicted" = .pred)

# write_csv(finalresults, "output.csv")

# Write out results & workflow
save(knn_tune, knn_workflow, knn_results, final_results, file = "knn_tune.rda")

