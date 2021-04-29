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
summary(edx_try)
edx_desc <- describe(edx_try)
edx_desc

         

## extract release year from title column to create a new predictor for edx and validation
## extract year, month and day from timestamp
edx_try <- edx_try %>% 
  mutate(rel_year = str_extract(title, "(\\d{4})"), # create rel_year base on the title-entry
         rel_year = as.numeric(str_replace_all(rel_year,"(|)", "")), # remove parantheses and convert to number
         rel_year = ifelse(rel_year>2020, 1994, ifelse(rel_year < 1900, 1994,rel_year) ),
         ts_year = year(as_datetime(timestamp)),
         ts_month = month(as_datetime(timestamp), label = FALSE),
         ts_day = wday(as_datetime(timestamp), label = FALSE)
        )

validation <- validation %>%
  mutate(rel_year = str_extract(title, "(\\d{4})"),
         rel_year = as.numeric(str_replace_all(rel_year,"(|)", "")),
         ts_year = year(as_datetime(timestamp)),
         ts_month = month(as_datetime(timestamp), label = FALSE),
         ts_day = wday(as_datetime(timestamp), label = FALSE)
         )


#look for duplicate entries
problems <- which(duplicated(edx_try))
problems


#Correlation of predictors
d <- data.frame(userId = edx_try$userId,
                movieId = edx_try$movieId,
                rating = edx_try$rating,
                timestamp = edx_try$timestamp,
                rel_year = edx_try$rel_year,
                ts_year = edx_try$ts_year,
                ts_month = edx_try$ts_month,
                ts_day = edx_try$ts_day
                )
corrplot(cor(d), method = "number")

genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

### data visualization


##histogram rating overall
edx_try %>% ggplot(aes(rating))+
  geom_histogram(binwidth = 0.5)

r<- mean(edx_try$rating)


##histograms by filter ()



##avg rating by year
edx_try %>%
  group_by(ts_year)%>%
  summarize(avg_rating = mean(rating))%>%
  ggplot(aes(x=ts_year, y=avg_rating))+
  geom_point()+
  geom_line()

##boxplot ratings by year
edx_try %>%
  ggplot(aes(x=ts_year, y=rating, group=ts_year))+
  geom_boxplot()+
 # geom_jitter(shape=21, alpha=0.5)+
  stat_summary(fun=mean, geom="point", shape=15, size=4, col=2)+
  coord_flip()


##boxplot ratings by month
edx_try %>%
  ggplot(aes(x=ts_month, y=rating, group=ts_month))+
  geom_boxplot()+
  # geom_jitter(shape=21, alpha=0.5)+
  stat_summary(fun=mean, geom="point", shape=15, size=4, col=2)+
  coord_flip()

    
##boxplot ratings by weekday
edx_try %>%
  ggplot(aes(x=ts_day, y=rating, group=ts_day))+
  geom_boxplot()+
  #geom_jitter(shape=21, alpha=0.5)+
  stat_summary(fun=mean, geom="point", shape=15, size=4, col=2)+
  coord_flip()

##boxplot ratings by release year of the movie
edx_try %>%
  ggplot(aes(x=rel_year, y=rating, group=rel_year))+
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


genre_avgs <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
 # summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  #filter(n >= 100) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u)) 

pred_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres')%>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

pred_ratings <- ifelse(!is.na(pred_ratings),pred_ratings, mu+b_i+b_u)

model_2.3_rmse <- RMSE(edx_test$rating, pred_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="linear movieId + userId + genre",
                                     RMSE = model_2.3_rmse ))
rmse_results %>% knitr::kable()



####MODEL2.4 - REGULARIZED LINEAR MODEL####
lambda <- 3
mu <- mean(edx_train$rating)
movie_reg_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

edx_train %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

edx_train %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

predicted_ratings <- edx_test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

####use cross validation to find the best lambda
lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

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
train_loess <- train(rating ~userId+movieId+as_datetime(timestamp), 
                  method = "loess", 
                  data = edx_train)
train_rf$bestTune
model5_predict <- predict(train_loess, test_set)

model_5_rmse <- RMSE(edx_test$rating, model5_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="LOESS",
                                     RMSE = model_5_rmse ))
####MODEL 6 - Random Forest####
train_rf <- train(rating ~userId+movieId+as_datetime(timestamp), 
                  method = "rf", 
                  tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 5,
                  data = edx_train)
train_rf$bestTune
model6_predict <- predict(train_rf, edx_test)


model_6_rmse <- RMSE(edx_test$rating, model6_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="random forest",
                                     RMSE = model_6_rmse ))
rmse_results %>% knitr::kable()
#####MODEL 7 - GLM####

train_glm <- train(rating ~userId+movieId+as_datetime(timestamp), 
                  method = "glm", 
                  data = edx_train)
train_glm$bestTune
model7_predict <- predict(train_glm, edx_test)


model_7_rmse <- RMSE(edx_test$rating, model7_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="GLM",
                                     RMSE = model_7_rmse ))
rmse_results %>% knitr::kable()

#####REGULARIZATION######

###watch out: no regularization if values are not centered around 0 ==> first subtract the overall mean

#########VALIDATE BEST MODEL#########

predict_final <- predict(_INSERT MODEL HERE_, validation, type = "raw")

validation_rmse <- RMSE(validation$rating, predict_final)
validation_rmse
