##########################################################
# Create testing and validation set
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")

library(rattle)
library(scales)
library(corrplot)
library(recosystem)
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(ggcorrplot)

## Loading dataset
df <- read_csv("./Breast_Cancer.csv")

# Divide into validation (10%) and main (90%, here named df) 
colnames(df) <- make.names(colnames(df))
trainIndex <- createDataPartition(df$Status, p = 0.9, list = FALSE, times = 1)
validation  <- df[-trainIndex,]
df <- df[ trainIndex,]

rm(trainIndex)

#################################
# Clean and Tansform Dataset
#################################

# Check null values
apply(df, 2, function(x) sum(is.na(x)))

# Check types of df columns
sapply(df, class)

# Nb of unique values per columns
apply(df, 2, function(x) length(unique(x)))

# Transform everything into numerical
df_original <- df
df <- as.data.frame(apply(df, 2, function(x) ifelse(x == "numeric", x, as.numeric(as.factor(x)))))

#################################
# Exploratory Data Analysis
#################################

# Number of patients that died
sum(df_original$Status=="Alive" )/nrow(df_original)

df_original %>% ggplot(aes(Status, fill=Status)) + geom_bar() + theme_minimal() +
  scale_fill_manual(values=c("deepskyblue3", "red3"))

# Correlation plot
cor <- cor(df)
colnames(cor) <- c("Age", "Race", "Marital", "T.Stage", "N.Stage", "6th.Stage", "Differentiate", "Grade", "A.Stage",
                   "T Size", "Estrogen", "Progesterone", "Node Exam", "Node +", "Survaval M", "Status")
corrplot(cor, addCoef.col = 1, number.cex = 0.5 )     

# T.Stage
df_original %>% group_by(T.Stage) %>% 
  summarize(Alive = sum(Status=="Alive"), Dead = sum(Status=="Dead"), Total = Alive+Dead) %>% 
  arrange(T.Stage) %>% pivot_longer(cols = c(Dead, Alive), values_to = "N", names_to = "Status") %>%
  ggplot(aes(x=0, y=N, fill=Status, group=Status)) + 
  geom_bar(stat = "identity", position="fill") +
  geom_text(aes(label = percent(N/Total)), position = position_fill(vjust = 0.5)) +
  coord_polar(theta = "y") +
  facet_wrap(.~ T.Stage) +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.grid  = element_blank()) +
  scale_fill_manual(values=c("deepskyblue3", "red3"))


# Survival.Months
df_original %>% group_by(Survival.Months) %>% select(Status) %>%
  summarize(Alive = sum(Status=="Alive"), Dead = sum(Status=="Dead")) %>% 
  arrange(Survival.Months) %>% pivot_longer(cols = c(Dead, Alive), values_to = "N", names_to = "Status") %>%
  ggplot(aes(x=Survival.Months,y=N, fill=Status)) +
  geom_bar(stat = "identity", position="fill") + 
  scale_fill_manual(values=c("deepskyblue3", "red3"))

#################################
# Predictions
#################################

# Split data into train and test set
set.seed(1, sample.kind="Rounding") 
colnames(df) <- make.names(colnames(df))
trainIndex <- createDataPartition(df$Status, p = 0.85, list = FALSE, times = 1)
train_set <- df[ trainIndex,]
test_set  <- df[-trainIndex,]

# KNN
fit_knn <- train(as.factor(Status) ~ .,  method = "knn", 
                 tuneGrid = data.frame(k = 5), 
                 data = train_set)
predict_knn_all <- predict(fit_knn, newdata = test_set)
acc_all <- sum(predict_knn_all==test_set$Status)/nrow(test_set)

fit_knn <- train(as.factor(Status) ~ Survival.Months + Reginol.Node.Positive + T.Stage + N.Stage + Estrogen.Status + X6th.Stage + Progesterone.Status,  method = "knn", 
                 tuneGrid = data.frame(k = 3), 
                 data = train_set)
predict_knn <- predict(fit_knn, newdata = test_set)
acc_few <- sum(predict_knn==test_set$Status)/nrow(test_set)

accuracy_results <- tibble(method = "KNN Accuracy", Accuracy_All = acc_all, Accuracy_Part = acc_few)

# RBorist
#With all the predictors
fit_rborist <- train(as.factor(Status) ~  .,
                     method = "Rborist",
                     tuneGrid = data.frame(predFixed = 2, minNode = 3),
                     data = train_set)
predict_rborist_all <- predict(fit_rborist, newdata = test_set)
acc_all_rborist <- sum(predict_rborist_all==test_set$Status)/nrow(test_set)

# With only a few predictors
fit_rborist <- train(as.factor(Status) ~  Survival.Months + Reginol.Node.Positive + T.Stage + N.Stage + Estrogen.Status + X6th.Stage + Progesterone.Status,
                     method = "Rborist",
                     tuneGrid = data.frame(predFixed = 2, minNode = 3),
                     data = train_set)
predict_rborist <- predict(fit_rborist, newdata = test_set)
acc_few_rborist <- sum(predict_rborist==test_set$Status)/nrow(test_set)

accuracy_results <- bind_rows(accuracy_results,
                              tibble(method="RBorist",
                                     Accuracy_All = acc_all_rborist, Accuracy_Part= acc_few_rborist)) 

# Random forest
#With all the predictors
fit_rf <- train(as.factor(Status) ~  .,
                data=train_set,
                method = "rf", trControl=trainControl(method="cv", number=5))
predict_rf_all <- predict(fit_rf, newdata = test_set)
acc_all_rf <- sum(predict_rf_all==test_set$Status)/nrow(test_set)

# With only a few predictors
fit_rf <- train(as.factor(Status) ~  Survival.Months + Reginol.Node.Positive + T.Stage + N.Stage + Estrogen.Status + X6th.Stage + Progesterone.Status,
                     data=train_set,
                     method = "rf", trControl=trainControl(method="cv", number=5))
predict_rf <- predict(fit_rf, newdata = test_set)
acc_few_rf <- sum(predict_rf==test_set$Status)/nrow(test_set)

accuracy_results <- bind_rows(accuracy_results,
                              tibble(method="Random Forest",
                                     Accuracy_All = acc_all_rf, Accuracy_Part = acc_few_rf)) 

# GLM
#With all the predictors
fit_glm <- train(as.factor(Status) ~ ., data=train_set,
                    method='glm')
predict_glm_all <- predict(fit_glm, newdata = test_set)
acc_all_glm <- sum(predict_glm_all==test_set$Status)/nrow(test_set)

# With only a few predictors
fit_glm <- train(as.factor(Status) ~ Survival.Months + Reginol.Node.Positive + T.Stage + N.Stage + Estrogen.Status + X6th.Stage + Progesterone.Status
                 , data=train_set,
                 method='glm')
predict_glm <- predict(fit_glm, newdata = test_set)
acc_few_glm <- sum(predict_glm==test_set$Status)/nrow(test_set)

accuracy_results <- bind_rows(accuracy_results,
                              tibble(method="GLM",
                                     Accuracy_All = acc_all_glm, Accuracy_Part = acc_few_glm)) 

# RPart
#With all the predictors
fit_rpart <- train(as.factor(Status) ~  ., data = train_set,
                     method = "rpart")
predict_rpart_all <- predict(fit_rpart, newdata = test_set)
acc_all_rpart <- sum(predict_rpart_all==test_set$Status)/nrow(test_set)

fancyRpartPlot(fit_rpart$finalModel)

# With only a few predictors
fit_rpart <- train(as.factor(Status) ~ Survival.Months + Reginol.Node.Positive + T.Stage + N.Stage + Estrogen.Status + X6th.Stage + Progesterone.Status,
                   data = train_set,
                   method = "rpart")
predict_rpart <- predict(fit_rpart, newdata = test_set)
acc_few_rpart <- sum(predict_rpart==test_set$Status)/nrow(test_set)

accuracy_results <- bind_rows(accuracy_results,
                              tibble(method="RPart",
                                     Accuracy_All = acc_all_rpart, Accuracy_Part = acc_few_rpart)) 

# Combining methods
# Creating a dataframe with all the previous predictions
pfinal <- tibble(predict_knn = as.numeric(predict_knn_all),
                 predict_rborist = as.numeric(predict_rborist_all),
                 predict_rf = as.numeric(predict_rf_all),
                 predict_glm = as.numeric(predict_glm_all),
                 predict_rpart = as.numeric(predict_rpart_all))

# Choose the predicted outcome depending on the majority
prediction_combined <- ifelse(rowSums(pfinal)>7, 2, 1)

# Compute accuracy
acc_all_combined <- sum(prediction_combined==test_set$Status)/nrow(test_set)

# Same with few predictors
pfinal <- tibble(predict_knn = as.numeric(predict_knn),
                 predict_rborist = as.numeric(predict_rborist),
                 predict_rf = as.numeric(predict_rf),
                 predict_glm = as.numeric(predict_glm),
                 predict_rpart = as.numeric(predict_rpart))

prediction_combined <- ifelse(rowSums(pfinal)>7, 2, 1)
acc_few_combined <- sum(prediction_combined==test_set$Status)/nrow(test_set)


accuracy_results <- bind_rows(accuracy_results,
                              tibble(method="Combined",
                                     Accuracy_All = acc_all_combined, Accuracy_Part = acc_few_combined)) 

accuracy_results

#################################
# Final Predictions
#################################

# Validation set transformation
validation <- as.data.frame(apply(validation, 2, function(x) ifelse(x == "numeric", x, as.numeric(as.factor(x)))))

# Random Forest model
fit_final <- train(as.factor(Status) ~  .,
                data=df,
                method = "rf", trControl=trainControl(method="cv", number=5))
predict_final <- predict(fit_final, newdata = validation)
acc_final <- sum(predict_final==validation$Status)/nrow(validation)

