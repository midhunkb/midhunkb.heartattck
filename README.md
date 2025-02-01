# midhunkb.heartattack
# Heart Attack Prediction Analysis

## 1. Introduction
We aims to use machine learning to identify key variables that influence  heart attack risk  and to build a predictive model that could make predictions based on available data that would help healthcare professional create strategies for heart disease management and prevention
This analysis explores a dataset related to heart attack risks using various machine learning models such as Decision Trees, Random Forest, Logistic Regression, and K-Nearest Neighbors (KNN). Data preprocessing, feature selection, and model evaluation are key components of this study.

## 2. Loading Dataset
```r
summary(heart_attack_dataset)
```

## 3. Loading Necessary Libraries
```r
library(dplyr)
library(tidyr)
library(ranger)
library(ggplot2)
library(caret)
library(vip)
library(rattle)
library(rpart)
```

## 4. Data Cleaning and Preprocessing
### Removing Duplicates
```r
dataset <- distinct(heart_attack_dataset)  # Removing duplicate rows
```

### Splitting Blood Pressure into Two Columns
```r
heart_dataset <- separate(heart_attack_dataset, Blood.Pressure, into = c("BP_systolic", "BP_diastolic"), sep = "/")
heart_dataset$BP_systolic <- as.numeric(heart_dataset$BP_systolic)
heart_dataset$BP_diastolic <- as.numeric(heart_dataset$BP_diastolic)
```

### Converting Dataset to Data Frame
```r
heart_dataset <- as.data.frame(heart_dataset)
```

### Removing Unnecessary Columns
```r
heart_set <- heart_dataset[-c(1, 24, 25, 26)]
```

### Encoding Categorical Variables
```r
heart_set$Sex <- ifelse(heart_set$Sex == "Male", 1, 0)
heart_set$Diet <- recode(heart_set$Diet, "Unhealthy" = 1, "Average" = 2, "Healthy" = 3)
```

## 5. Decision Tree Model
```r
heart_tree <- rpart(Heart.Attack.Risk ~ ., data = heart_set)
summary(heart_tree)
fancyRpartPlot(heart_tree)
rpart.plot(heart_tree)
```

## 6. Adjusting Decision Tree Parameters
```r
heart_tree1 <- rpart(Heart.Attack.Risk ~ ., data = heart_set, method = "class", control = rpart.control(cp = 0.01))
rpart.plot(heart_tree1)
summary(heart_tree1)
```

## 7. Feature Selection
```r
imp_features <- heart_set[-c(2,7,8,9,10,11,13,14,15,16,21,22)]
```

## 8. Logistic Regression Model
```r
set.seed(310)
data_split <- initial_split(heart_set, prop = .7, strata = "Heart.Attack.Risk")
data_train <- training(data_split)
data_test  <- testing(data_split)

model1 <- glm(Heart.Attack.Risk ~ ., family = "binomial", data = data_train)
summary(model1)
```

## 9. ROC Curve Analysis
```r
library(pROC)
predicted_probs <- predict(model1, type = "response")
roc_curve <- roc(data_train$Heart.Attack.Risk, predicted_probs)
plot(roc_curve, col = "blue", main = "ROC Curve", lwd = 2)
```

## 10. Cross-Validation for Logistic Regression
```r
cv_model3 <- train(Heart.Attack.Risk ~ ., data = data_train, method = "glm", family = "binomial", trControl = trainControl(method = "cv", number = 10))
```

## 11. Confusion Matrix
```r
pred_class <- predict(cv_model3, newdata = data_test, type = "raw")
conf_matrix <- confusionMatrix(data = pred_class, reference = data_test$Heart.Attack.Risk)
print(conf_matrix)
```

## 12. K-Nearest Neighbors (KNN) Model
```r
knndata <- heart_set[c(3,4,7,22,23)]
data_task <- makeClassifTask(data = knndata, target = "Heart.Attack.Risk")
knn <- makeLearner("classif.knn", par.vals = list("k" = 13))
```

## 13. K-Fold Cross-Validation for KNN
```r
kfold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 20, stratify = TRUE)
kfoldcv <- resample(learner = knn, task = data_task, resampling = kfold, measures = acc)
kfoldcv$aggr
```

## 14. Hyperparameter Tuning for KNN
```r
knnparamspace <- makeParamSet(makeDiscreteParam("k", values = 1:10))
gridsearch <- makeTuneControlGrid()
cvfortunning <- makeResampleDesc("RepCV", folds = 10, reps = 20)
tunedk <- tuneParams("classif.knn", task = data_task, resampling = cvfortunning, par.set = knnparamspace, control = gridsearch)
```

## 15. Data Exploration
### Distribution of Heart Attack Risk
```r
ggplot(heart_set, aes(x = Heart.Attack.Risk, fill = Heart.Attack.Risk)) +
  geom_bar() +
  labs(x = 'Values', y = 'Count', title = 'Count of Yes and No') +
  theme_minimal()
```

### Heart Attack Risk by Continent
```r
ggplot(heart_dataset, aes(x = Continent, y = Heart.Attack.Risk, fill = Continent)) +
  geom_bar(stat = "identity") +
  labs(x = 'Continents', y = 'Count')
```

## 16. Conclusion
This study examined heart attack risk using various machine learning models. Decision trees and logistic regression were used for classification, while KNN was optimized using cross-validation. Further research could explore deep learning models and additional feature engineering to improve predictive performance.
