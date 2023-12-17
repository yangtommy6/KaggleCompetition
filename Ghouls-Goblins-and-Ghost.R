library(tidyverse) # for data manipulation
library(caret) # for the training model
library(xgboost) # for gradient boosting
library(readr)
trainSet <- read_csv("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/train.csv") # this file has no missing values
missSet <- read_csv("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/trainWithMissingValues.csv") # this file has missing values
library(recipes)
library(dplyr)

# Assuming 'missSet' is the dataset with missing values
# First, identify numeric and nominal columns
numeric_columns <- missSet %>% select_if(is.numeric) %>% names()
nominal_columns <- missSet %>% select_if(is.factor) %>% names()

# Create a recipe that imputes all the missing values in the missing dataset
recipe <- recipe(~ ., data = missSet) %>%
  step_impute_mean(all_of(numeric_columns)) %>%
  step_impute_mode(all_of(nominal_columns))

# Prep the recipe and impute the data
prepped_recipe <- prep(recipe, training = missSet)
imputedSet <- bake(prepped_recipe, new_data = NULL)

library(yardstick)
# Assuming 'trainSet' is your original data without missing values.
# Replace 'imputedSet' with the actual imputed dataset variable.
# You may need to adjust the column names depending on the structure of your datasets.
rmse_vec(trainSet[is.na(missSet)], imputedSet[is.na(missSet)])

##################################

train_data <- read_csv("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/train.csv")
library(tidymodels)
library(tidyverse)
library(nnet)


# Pre-processing recipe
nn_recipe <- recipe(type ~ ., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  step_normalize(all_numeric_predictors())

# Define the model with a conservative number of hidden units
nn_model <- mlp(hidden_units = tune(), penalty = 0) %>%
  set_engine("nnet", linout = FALSE, trace = FALSE) %>%
  set_mode("classification")

# Define the cross-validation folds
cv_folds <- vfold_cv(train_data, v = 5)

# Set a more conservative range for hidden units, let's say up to 10
maxHiddenUnits <- 10
nn_tuneGrid <- grid_regular(hidden_units(range = c(1, maxHiddenUnits)), levels = 10)

# Combine the recipe and model into a workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

# Tune the model
tuned_nn <- nn_wf %>%
  tune_grid(resamples = cv_folds, grid = nn_tuneGrid)

# Extract the results
accuracy_results <- tuned_nn %>% 
  collect_metrics() %>%
  filter(.metric == "accuracy")

# Plot the results
accuracy_plot <- ggplot(accuracy_results, aes(x = hidden_units, y = mean)) +
  geom_line() +
  labs(x = "Number of Hidden Units", y = "Mean Accuracy")

# Print the plot
print(accuracy_plot)


###
# Load required libraries
library(tidymodels)
library(vroom)
library(caret)
library(dplyr)

# Define file paths for training and test data
train_file <- "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/train.csv"
test_file <- "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/test.csv"

# Read the data
train_complete <- vroom(train_file)
test_complete <- vroom(test_file)

# Feature Engineering: Create interaction terms
interaction_terms <- c("hair_soul", "bone_flesh", "bone_hair", "bone_soul", "flesh_hair", "flesh_soul")

train_complete <- train_complete %>%
  mutate(
    hair_soul = hair_length * has_soul,
    bone_flesh = bone_length * rotting_flesh,
    bone_hair = bone_length * hair_length,
    bone_soul = bone_length * has_soul,
    flesh_hair = rotting_flesh * hair_length,
    flesh_soul = rotting_flesh * has_soul
  )

test_complete <- test_complete %>%
  mutate(
    hair_soul = hair_length * has_soul,
    bone_flesh = bone_length * rotting_flesh,
    bone_hair = bone_length * hair_length,
    bone_soul = bone_length * has_soul,
    flesh_hair = rotting_flesh * hair_length,
    flesh_soul = rotting_flesh * has_soul
  )

# Train Control for cross-validation
myControl <- trainControl(method = "cv", number = 10)

# Random Forest Model
set.seed(10)
rf_model <- train(
  type ~ .,  # Using all features
  data = train_complete, 
  method = "ranger", 
  trControl = myControl,
  importance = 'impurity',
  tuneLength = 3
)

# GLMnet Model
set.seed(10)
glm_model <- train(
  type ~ .,  # Using all features
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1, length = 20)),
  data = train_complete,
  trControl = myControl
)

# Model comparison
models <- list(rf = rf_model, glmnet = glm_model)
resampled <- resamples(models)
summary(resampled)

# Choose the best model and predict on the test set
# Assuming glmnet model performs better based on the summary
predicted_class <- predict(glm_model, test_complete)

# Prepare the submission file
my_solution <- data.frame(id = test_complete$id, Type = predicted_class)

# Write the submission data frame to a CSV file
vroom_write(x = my_solution, file = "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/ggg.csv", delim = ",")



##############


# Load required libraries
library(tidymodels)
library(vroom)
library(caret)
library(dplyr)
library(randomForest)

# Define file paths for training and test data
train_file <- "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/train.csv"
test_file <- "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/test.csv"


# Feature Engineering: Create interaction terms
train_complete <- train_complete %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

test_complete <- test_complete %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)


############
# Prepare data for PCA
train_data_pca <- train_complete %>% select(-id, -type, -color)
test_data_pca <- test_complete %>% select(-id, -color)

# Apply PCA
pca_model <- prcomp(train_data_pca, center = TRUE, scale. = TRUE)
train_pca <- predict(pca_model, train_data_pca)
test_pca <- predict(pca_model, test_data_pca)

# Combine PCA data with 'color' for Random Forest
train_for_rf <- data.frame(train_pca, color = train_complete$color)
test_for_rf <- data.frame(test_pca, color = test_complete$color)

# Convert 'color' to factor for Random Forest
train_for_rf$color <- as.factor(train_for_rf$color)
test_for_rf$color <- as.factor(test_for_rf$color)

# Train Random Forest Model
set.seed(10)
rf_model <- randomForest(x = train_for_rf, y = train_labels, importance = TRUE, ntree = 500)

# Predict on Test Set
predictions <- predict(rf_model, test_for_rf)

# Prepare the submission file
my_solution <- data.frame(id = test_complete$id, Type = predictions)



# Write the submission data frame to a CSV file
vroom_write(x = my_solution, file = "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/submission.csv", delim = ",")



####
library(tidymodels)
library(vroom)
library(caret)
library(dplyr)

# Load the data
train_complete <- vroom("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/train.csv") 
test_complete <- vroom("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/test.csv")
library(tidymodels)
library(vroom)
library(caret)
library(dplyr)
library(randomForest)
library(gbm)


train_complete <- train_complete %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

test_complete <- test_complete %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

# PCA for dimensionality reduction
pca_model <- preProcess(train_complete[, -which(names(train_complete) %in% c("id", "type"))], method = "pca", pcaComp = 2)
train_pca <- predict(pca_model, train_complete)
test_pca <- predict(pca_model, test_complete)

# Combine PCA components with the original data
train_pca$type <- train_complete$type
test_pca$id <- test_complete$id

# Train Control for cross-validation
myControl <- trainControl(method = "cv", number = 10)

# Random Forest Model
set.seed(10)
rf_model <- train(
  type ~ ., 
  data = train_pca, 
  method = "rf", 
  trControl = myControl,
  tuneLength = 3
)

# GBM Model

set.seed(10)
gbm_model <- train(
  type ~ ., 
  data = train_pca, 
  method = "gbm", 
  trControl = myControl,
  verbose = FALSE,
  tuneGrid = expand.grid(
    interaction.depth = 1:3,
    n.trees = (1:3) * 50,
    shrinkage = 0.1,
    n.minobsinnode = c(5, 10, 15)  # Example values
  )
)


# Model comparison
models <- list(rf = rf_model, gbm = gbm_model)
resampled <- resamples(models)
summary(resampled)

# Choose the best model based on the summary and predict on test set
# Assuming gbm_model performs better (for example)
predicted_class <- predict(gbm_model, test_pca)

# Prepare the submission file
my_solution <- data.frame(id = test_pca$id, type = predicted_class)

# Write the submission data frame to a CSV file
write.csv(my_solution, "submission.csv", row.names = FALSE)


# Write the submission data frame to a CSV file
write.csv(my_solution, file = "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/6.csv", row.names = FALSE)



########


library(tidymodels)
library(vroom)
library(caret)
library(dplyr)
library(randomForest)
library(gbm)

# Load data
train_complete <- vroom("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/train.csv") 
test_complete <- vroom("/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/test.csv")

# Advanced Feature Engineering
# [Add any additional feature engineering steps here if needed]

# PCA with more components
pca_model <- preProcess(train_complete[, -which(names(train_complete) %in% c("id", "type"))], method = "pca", pcaComp = 5)
train_pca <- predict(pca_model, train_complete)
test_pca <- predict(pca_model, test_complete)

# Combine PCA components with the original data
train_pca$type <- train_complete$type
test_pca$id <- test_complete$id

# Train Control for cross-validation
myControl <- trainControl(method = "cv", number = 10)

# Hyperparameter Tuning for Random Forest
set.seed(10)
rf_model <- train(
  type ~ ., 
  data = train_pca, 
  method = "rf", 
  trControl = myControl,
  tuneLength = 5  # Increase tune length for more options
)

# Updated GBM Model with expanded tuneGrid
set.seed(10)
gbm_model <- train(
  type ~ ., 
  data = train_pca, 
  method = "gbm", 
  trControl = myControl,
  verbose = FALSE,
  tuneGrid = expand.grid(
    interaction.depth = 1:5,             # Increased depth
    n.trees = seq(50, 200, by = 50),     # More trees
    shrinkage = c(0.05, 0.1, 0.15),      # Different shrinkage rates
    n.minobsinnode = c(5, 10, 20)        # Different values for min. observations
  )
)


# Model comparison
models <- list(rf = rf_model, gbm = gbm_model)
resampled <- resamples(models)
summary(resampled)

# Choose the best model and predict on test set
# [Select the best model based on the summary]
# For example, assuming gbm_model performs better
predicted_class <- predict(gbm_model, test_pca)

# Prepare the submission file
my_solution <- data.frame(id = test_pca$id, type = predicted_class)

# Write the submission data frame to a CSV file
write.csv(my_solution, "/Users/christian/Desktop/STAT348/GGG/Ghouls-Goblins-and-Ghosts...-Boo-/8.csv", row.names = FALSE)

