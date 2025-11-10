library(tidymodels)
library(readr)
library(dplyr)


train <- read_csv("train.csv")
test  <- read_csv("test.csv")


train$type <- as.factor(train$type)


rf_recipe <- recipe(type ~ ., data = train) |>
  update_role(id, new_role = "ID") |>   
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())


rf_model <- rand_forest(
  trees = 1000,
  mtry = tune(),   
  min_n = tune()   
) |>
  set_engine("randomForest") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_recipe(rf_recipe) |>
  add_model(rf_model)



cv_folds <- vfold_cv(train, v = 10, repeats = 5, strata = type)


rf_grid <- grid_regular(
  mtry(range = c(2, 9)),
  min_n(),
  levels = 5
)


rf_tuned <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(accuracy)
)


best_params <- select_best(rf_tuned, metric = "accuracy")


final_rf <- finalize_workflow(rf_wf, best_params)


final_fit <- fit(final_rf, data = train)


test_preds <- predict(final_fit, test) |>
  bind_cols(test |> select(id))


submission <- test_preds |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

write_csv(submission, "submission.csv")
