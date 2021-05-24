# Elastic Net tuning ----

# Load package(s) ----
library(tidyverse)
library(janitor)
library(tictoc)
library(rsample)
library(tidymodels)
library(glmnet)

# set seed
set.seed(3013)

# setup ----
bl_test <- read_csv("data/test.csv") %>% 
  clean_names()

bl_train <- read_csv("data/train.csv") %>% 
  clean_names()

bl_train <- bl_train %>% 
  mutate(
    hi_int_prncp_pd = ifelse(hi_int_prncp_pd == 0, "0", "1"),
    hi_int_prncp_pd = as.factor(hi_int_prncp_pd)
  )

bl_folds <- bl_train %>%
  vfold_cv(v = 5, repeats = 1, strata = hi_int_prncp_pd)
bl_folds

bl_recipe <- recipe(hi_int_prncp_pd ~ out_prncp_inv + int_rate + loan_amnt + term + grade,
                    data = bl_train) %>% 
  step_dummy(term, grade) %>% 
  step_normalize(all_predictors())

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
  add_recipe(bl_recipe)

# Tuning/fitting ----
tic("Elastic Net Model")
en_tuned <- en_wf %>% 
  tune_grid(bl_folds, grid = en_grid)

toc(log = TRUE)

# save runtime info
en_runtime <- tic.log(format = TRUE)

# fit to train data
en_wf_tune <- en_wf %>% 
  finalize_workflow(select_best(en_tuned, metric = "accuracy"))

en_results <- fit(en_wf_tune, bl_train)

final_results <- en_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>%
  rename("Id" = id,
         "Category" = .pred_class)

final_results <- final_results[c(2,1)]

write_csv(final_results, path="Lin_Kylie_EN_ClassComp.csv")

# Write out results & workflow
save(en_tuned, en_wf, en_runtime, final_results, file = "en_tuned.rda")



