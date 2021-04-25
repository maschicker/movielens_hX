######### FINAL PROJECT - MOVIELENS #######
######### HarvardX - DATA SCIENCE ########

#######INSTALL PACKAGES#######
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(stringr)
library(ggplot2)
library(lubridate)
library(Hmisc)
library(corrplot)
library(rpart)
library(matrixStats)


######## Create edx set, validation set (final hold-out test set) #######

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#######PRE-PROCESSING DATA########

#small data_excerpt to try code fast
try_index <- createDataPartition(y = edx$rating, times = 1, p = 0.001, list = FALSE)
edx_try <- edx[try_index,]
dim(edx_try)
### data exploration
summary(edx)
str(edx)
edx_desc <- describe(edx)

# mutate data (REMOVE NA´S, rating as.factor, convert timestamp 
#edx_clean <- edx_try %>% 
#mutate(rating = ifelse(is.na(rating), median(rating, na.rm = TRUE), rating),
#         timestamp = as_datetime(timestamp)# convert timestamp to a date_time object
#  )


#look for duplicate entries
problems <- which(duplicated(edx_try))
problems


#Correlation of predictors
d <- data.frame(userId = edx_try$userId,
                movieId = edx_try$movieId,
                rating = edx_try$rating,
                timestamp = edx_try$timestamp)
corrplot(cor(d), method = "number")



### data visualization
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

edx_visual <- edx_try %>% mutate(
  year = year(as_datetime(timestamp)),
  month = month(as_datetime(timestamp), label=TRUE),
  day = wday(as_datetime(timestamp), label = TRUE))


##histogram rating
edx_visual %>% ggplot(aes(rating))+
  geom_histogram(binwidth = 0.5)

r<- mean(edx_visual$rating)

##avg rating by year
edx_visual %>%
  group_by(year)%>%
  summarize(avg_rating = mean(rating))%>%
  ggplot(aes(x=year, y=avg_rating))+
  geom_point()+
  geom_line()

##boxplot ratings by year
edx_visual %>%
  ggplot(aes(x=year, y=rating, group=year))+
  geom_boxplot()+
 # geom_jitter(shape=21, alpha=0.5)+
  stat_summary(fun=mean, geom="point", shape=15, size=4, col=2)+
  coord_flip()


##boxplot ratings by month
edx_visual %>%
  ggplot(aes(x=month, y=rating, group=month))+
  geom_boxplot()+
  # geom_jitter(shape=21, alpha=0.5)+
  stat_summary(fun=mean, geom="point", shape=15, size=4, col=2)+
  coord_flip()

    
##boxplot ratings by weekday
edx_visual %>%
  ggplot(aes(x=day, y=rating, group=day))+
  geom_boxplot()+
  #geom_jitter(shape=21, alpha=0.5)+
  stat_summary(fun=mean, geom="point", shape=15, size=4, col=2)+
  coord_flip()





  ##by genre


##### standardizing predictors


##### log-transform???


##### removing predictors with non-unique values or zero-variation
nzv <- nearZeroVar(edx_try)
#image(matrix(1:784 %in% nzv, 28, 28))
nzv
col_index <- setdiff(1:ncol(edx_try), nzv)
length(col_index)


#########MODELING#########
####DEFINE RMSE-FUNCTION####
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


####CREATING TRAINING AND TEST SET#####

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx_try$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx_try[-test_index,]
edx_test <- edx_try[test_index,]

edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

####MODEL 1 - AVERAGE####
mu_hat <- mean(edx_train$rating)
mu_hat

naive_rmse <- RMSE(edx_test$rating, mu_hat)
naive_rmse


# dataframe to compare performance of models#
rmse_results <- data_frame(method = "Average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()





####MODEL 2.1 - LM with movieId####
mu <- mean(edx_train$rating) 
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu)
predicted_ratings

model_2.1_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="linear movieId",
                                     RMSE = model_2.1_rmse ))
rmse_results %>% knitr::kable()

####MODEL 2.2 - LM with userId + movieId####
edx_train %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=10) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 10, color = "black")


user_avgs <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i)) 

predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu+b_i)
#predicted_ratings <- round(predicted_ratings*2,1)/2 #round the result to the closest 0.5-step


model_2.2_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="linear movieId + userId",
                                     RMSE = model_2.2_rmse ))
rmse_results %>% knitr::kable()

####MODEL 2.3 - LM with userId + movieId + timestamp

#define category for genre combination and keep only cat w/>n=1000. plot error bar. which genre has lowest average rating
edx_train %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100) %>% #increase to 1000 when working with complete data set
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))






####MODEL 3.1 - knn####

train_knn <- train(rating ~ userId+movieId+as_datetime(timestamp), method = "knn", 
                   data = edx_train,
                   tuneGrid = data.frame(k = seq(5, 41, 2)))
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
train_knn$finalModel
min(train_knn$results$RMSE)
model3.1_predict <- predict(train_knn, edx_test, type = "raw")
model_3.1_rmse <- RMSE(edx_test$rating, model3.1_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="knn",
                                     RMSE = model_3_rmse))

rmse_results %>% knitr::kable()
####MODEL 3.2 - knn-cross-validation####
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(rating ~ userId+movieId+as_datetime(timestamp), method = "knn", 
                      data = edx_train,
                      tuneGrid = data.frame(k = seq(5, 41, 2)),
                      trControl = control)
ggplot(train_knn_cv, highlight = TRUE)

train_knn$results %>% 
  ggplot(aes(x = k, y = RMSE)) +
  geom_line() +
  geom_point() +
  
model3.2_predict <- predict(train_knn_cv, edx_test, type = "raw")
model_3.2_rmse <- RMSE(edx_test$rating, model3.2_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="knn_CV",
                                     RMSE = model_3.2_rmse))


####MODEL 4 - DT ####
set.seed(1, sample.kind = "Rounding")
train_dt <- train(rating ~userId+movieId+, 
                  method = "rpart", 
                  tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                  metric="RMSE",
                  data = edx_train)
train_dt$bestTune
model4_predict <- predict(train_dt, edx_test)



model_4_rmse <- RMSE(edx_test$rating, model4_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Decision tree",
                                     RMSE = model_4_rmse ))
####MODEL 5 - LOESS####
train_rf <- train(rating ~., 
                  method = "loess", 
                  tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 100,
                  data = edx_train)
train_rf$bestTune
model5_predict <- predict(train_rf, test_set)

model_5_rmse <- RMSE(edx_test$rating, model5_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="LOESS",
                                     RMSE = model_5_rmse ))
####MODEL 6 - Random Forest####
train_rf <- train(rating ~., 
                  method = "rf", 
                  tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 100,
                  data = edx_train)
train_rf$bestTune
model6_predict <- predict(train_rf, test_set)


model_6_rmse <- RMSE(edx_test$rating, model6_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="random forest",
                                     RMSE = model_6_rmse ))
#####REGULARIZATION######

###watch out: no regularization if values are not centered around 0 ==> first subtract the overall mean

#########VALIDATE BEST MODEL#########

predict_final <- predict(_INSERT MODEL HERE_, validation, type = "raw")

validation_rmse <- RMSE(validation$rating, predict_final)
validation_rmse
