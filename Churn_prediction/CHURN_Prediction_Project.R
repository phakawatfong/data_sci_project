# Churn Prediction

# Build model
# Logistic (glm)
## 1. accuracy
## 2. Recall
## 3. Precision
## 4. F1

# Bonus
# 1. K-mean clustering
#########################################################

# import library
library(dplyr)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)


## load data
df <- read_csv('churn.csv')

## explore data
glimpse(df)
mean(is.na(df))
mean(complete.cases(df))
# mean =1 => therfore, no missing value

## check for imbalanced class
df %>%
  count(churn) %>%
  mutate(pct = n/nrow(df))


## our label is Churn column -> convert it into factor for analysis
df <- df %>%
  select(everything()) %>%
  mutate(churn = factor(df$churn),
         internationalplan = factor(df$internationalplan),
         voicemailplan = factor(df$voicemailplan))

## Prepare Train Test Split DATA
# 1. split data

n <- nrow(df)

set.seed(101)
id <- sample(1:n, size=0.8*n)
train_df <- df[id, ]
test_df <- df[-id, ]

# 2. Train Model
## randomForest

## setup trainControl for optimization
set.seed(101)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs =  TRUE,
  summaryFunction = prSummary,
  verboseIter = TRUE,
)

## setup tuneGrid for optimization
myGrid <- data.frame(mtry = 2:16)

rf_model <- train(
  churn ~ .,
  data = train_df,
  method = "rf", # random Forest
  metric = "AUC",
  preProcess = c("center", "scale"),
  tuneGrid = myGrid,
  trControl = ctrl)

## check model
rf_model

# 3. Test Model (predict)
churn_rf <- predict(rf_model, newdata = test_df)


## Evaluate model
conM_rf <- confusionMatrix(churn_rf, test_df$churn,
                           dnn = c("Predicted", "Actual"),
                           mode = "prec_recall",
                           positive = "Yes")

### Pre 93.6%, Recall 74.6%

########## Let's Train other models ############

# Train Model
## logistic regression
set.seed(101)
ctrl_glm <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = prSummary,
  verboseIter = TRUE
)

glm_model <- train(
  churn ~ .,
  data = train_df,
  method = "glm",
  metric = "AUC",
  family = "binomial",
  preProcess = c("center", "scale"),
  trControl = ctrl_glm
)

# 3. Test model
churn_glm <- predict(glm_model, newdata = test_df)

mean(churn_glm == test_df$churn)
conM_glm <-confusionMatrix(churn_glm, test_df$churn, dnn = c("predcited", "actual"),
                mode = "prec_recall",
                positive = "Yes")

### Prec 63%, Recall 23% -> Bad model, needs to train again
## logistic regression
set.seed(101)

al_grid <- expand.grid(alpha = 0:1, lambda = seq(0.001,1, length=20))

glm_model2 <- train(
  churn ~ .,
  data = train_df,
  method = "glmnet",
  metric = "precision",
  family = "binomial",
  tuneGrid = al_grid,
  preProcess = c("center", "scale"),
  trControl = trainControl(method="cv",
                           number = 5,
                           summaryFunction = prSummary,
                           classProbs = TRUE)
)

# test model
churn_glm2 <- predict(glm_model2, newdata = test_df,
                      mode = "prec_recall",
                      positive = "Yes")

mean(churn_glm2 == test_df$churn)
conM_glm2 <- confusionMatrix(churn_glm2, test_df$churn, dnn = c("predcited", "actual"),
                mode = "prec_recall",
                positive = "Yes")

########## Let's Train other models ############
# Train Model
## decision Tree
set.seed(101)
ctrl_tree <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

tree_model <- train(
  churn ~ .,
  data = train_df,
  method = "rpart", # decision tree
  preProcess = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl_tree
)

# Test model
churn_tree <- predict(tree_model, newdata = test_df)

# Evaluate model
mean(churn_tree == test_df$churn)
conM_tree <- confusionMatrix(churn_tree, test_df$churn, dnn = c("predcited", "actual"),
                mode = "prec_recall",
                positive = "Yes")
# Plot rpart.plot
rpart.plot(tree_model$finalModel)
