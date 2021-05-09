######### FINAL PROJECT - MOVIELENS #######
######### HarvardX - DATA SCIENCE ########

#######INSTALL PACKAGES#######
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(stringr)
library(ggplot2)
library(lubridate)
library(corrplot)
library(rpart)
library(matrixStats)
library(gam)
library(splines)


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

#small data_excerpt to try code fast (not needed for final run)
#try_index <- createDataPartition(y = edx$rating, times = 1, p = 0.001, list = FALSE)
#edx_try <- edx[try_index,]
#dim(edx_try)


### data exploration
summary(edx)
dim(edx)
head(edx)
sapply(edx, class)

# extract release year from title column to create a new predictor for edx and validation
# extract year, month and day from timestamp
# convert numeric to integer if possible to half data size (8b-->4b) ==> higher modeling speed
edx <- edx %>% 
  mutate(rel_year = str_extract(title, "\\(\\d{4}\\)"), 
         rel_year = as.integer(str_replace_all(rel_year,"\\(|\\)", "")), # remove parantheses and convert to integer
         ts_year = as.integer(year(as_datetime(timestamp))),
         ts_month = as.integer(month(as_datetime(timestamp), label = FALSE)),
         ts_day = as.integer(wday(as_datetime(timestamp), label = FALSE)),
         movieId = as.integer(movieId)
        )

validation <- validation %>%
  mutate(rel_year = str_extract(title, "\\(\\d{4}\\)"), 
         rel_year = as.integer(str_replace_all(rel_year,"\\(|\\)", "")), # remove parantheses and convert to integer
         ts_year = as.integer(year(as_datetime(timestamp))),
         ts_month = as.integer(month(as_datetime(timestamp), label = FALSE)),
         ts_day = as.integer(wday(as_datetime(timestamp), label = FALSE)),
         movieId = as.integer(movieId)
         )


#look for duplicate entries
problems <- which(duplicated(edx))
length(problems)


#Correlation of predictors
d <- data.frame(userId = edx$userId,
                movieId = edx$movieId,
                rating = edx$rating,
                timestamp = edx$timestamp,
                rel_year = edx$rel_year,
                ts_year = edx$ts_year,
                ts_month = edx$ts_month,
                ts_day = edx$ts_day
                )
corrplot(cor(d), method = "number")

#dropping the timestamp column in edx and validation to reduce size
edx <- edx[,-c("timestamp")]
validation <- validation[,-c("timestamp")]


# removing predictors with non-unique values or zero-variation
nzv <- nearZeroVar(edx, saveMetrics = TRUE)
nzv[,3:4]


### data visualization


##histogram rating overall
edx %>% ggplot(aes(rating))+
  geom_histogram(binwidth = 0.5)+
  labs(title="Ratings overall")

##histogram rating_year
edx %>% ggplot(aes(ts_year))+
  geom_histogram(binwidth = 1)+
  labs(title="Timestamp Years overall")

##histogram rating_month
edx %>% ggplot(aes(ts_month))+
  geom_histogram(binwidth = 1)+
  labs(title="Timestamp Months overall")

##histogram rating_day
edx %>% ggplot(aes(ts_day))+
  geom_histogram(binwidth = 1)+
  labs(title="Timestamp Weekdays overall")

##histogram release year movies
edx %>% ggplot(aes(rel_year))+
  geom_histogram(binwidth = 1)+
  labs(title="Release Year overall")

##histogram movieId
edx %>% ggplot(aes(movieId))+
  geom_histogram(binwidth = 1)+
  labs(title="MovieId")+
  ylim(0,15000)

##histogram userId
edx %>% ggplot(aes(userId))+
  geom_histogram(binwidth = 1)+
  labs(title="UserId")+
  ylim(0,2500)


##plot ratings over timestamp_year (median, avg, se & smoothline)

edx %>% group_by(ts_year)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating),
    se= sd(rating)
  )%>%
  ggplot(aes(x=ts_year, y=avg, ymin=avg-se, ymax=avg+se, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  geom_errorbar(aes(alpha=0.3))+
  geom_smooth()+
  labs(title="Timestamp Year", y="Rating")+
  ylim(0,5)

##deep dive ts_year because of non-conclusive data in 1995
length(which(edx$ts_year==1995))

##plot ratings over timestamp_month (median, avg, se & smoothline)
edx %>% group_by(ts_month)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating),
    se= sd(rating)
  )%>%
  ggplot(aes(x=ts_month, y=avg, ymin=avg-se, ymax=avg+se, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  geom_errorbar(aes(alpha=0.3))+
  geom_smooth()+
  labs(title="Timestamp Month", y="Rating")+
  ylim(0,5)

##plot ratings over timestamp_weekday (median, avg, se & smoothline)
edx %>% group_by(ts_day)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating),
    se= sd(rating)
  )%>%
  ggplot(aes(x=ts_day, y=avg, ymin=avg-se, ymax=avg+se, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  geom_errorbar(aes(alpha=0.3))+
  geom_smooth()+
  labs(title="Timestamp Day", y="Rating")+
  ylim(0,5)

##Dropping non-differentiating variables ts_month and ts_day
edx <- edx[,1:7]
validation <- validation[,1:7]


##plot ratings over release year of the movie (median, avg, se & smoothline)
edx %>% group_by(rel_year)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating),
    se= sd(rating)
  )%>%
  ggplot(aes(x=rel_year, y=avg, ymin=avg-se, ymax=avg+se, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  geom_errorbar(aes(alpha=0.3))+
  geom_smooth()+
  labs(title="Release Year", y="Rating")+
  ylim(0,5)

##plot ratings over genre w/n>1000 ratings (median, avg, se)
edx %>% group_by(genres)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating),
    se= sd(rating)
  )%>%
  arrange(avg)%>%
  filter(n>1000)%>%
  ggplot(aes(x=genres, y=avg, ymin=avg-se, ymax=avg+se, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  geom_errorbar(aes(alpha=0.3))+
  labs(title="Genres > 1000 occurances", y="Rating", x="")+
  ylim(0,5)

##plot ratings over movieId (median, avg)
edx %>% group_by(movieId)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating)
  )%>%
  ggplot(aes(x=movieId, y=avg, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  labs(title="MovieId", y="Rating", x="Movie Id")+
  ylim(0,5)

##plot ratings over userId (median, avg)
edx %>% group_by(userId)%>%
  summarize(
    n=n(),
    avg=mean(rating),
    med=median(rating)
  )%>%
  ggplot(aes(x=userId, y=avg, col="avg"))+
  geom_point()+
  geom_point(aes(y=med, col="median"))+
  labs(title="UserId", y="Rating", x="User Id")+
  ylim(0,5)


#########MODELING#########
####INITIALIZATION####

#CREATING TRAINING AND TEST SET

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_test <- edx[test_index,]

edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")


#DEFINE RMSE-FUNCTION
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Calculate Mu (average)
mu <- mean(edx_train$rating) 
mu

#set up results dataframe
rmse_results <- data.frame(method = character(),
                           RMSE = numeric())
str(rmse_results)

####MODEL 1 - AVERAGE####
avg_rmse <- RMSE(edx_test$rating, mu)


# add results to dataframe to compare performance of models#
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="1 - Average",
                                     RMSE = avg_rmse ))
rmse_results %>% knitr::kable()

####MODEL 2.1.1 - LM with movieId####

set.seed(1, sample.kind = "Rounding")

movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu)

model_2.1.1_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.1.1 - linear movieId",
                                     RMSE = model_2.1.1_rmse ))
rmse_results %>% knitr::kable()


####MODEL 2.1.2 - LM with userId####

set.seed(1, sample.kind = "Rounding")

user_avgs <- edx_train %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating-mu))

predicted_ratings <- mu + edx_test %>% 
  left_join(user_avgs, by='userId') %>%
  .$b_u

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu)

model_2.1.2_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.1.2 - linear userId",
                                     RMSE = model_2.1.2_rmse ))
rmse_results %>% knitr::kable()



####MODEL 2.2.1 - LM with movieId + userId####

set.seed(1, sample.kind = "Rounding")

user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId')%>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating-mu-b_i))


predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu)

model_2.2.1_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.2.1 - linear movieId + userId",
                                     RMSE = model_2.2.1_rmse ))
rmse_results %>% knitr::kable()


####MODEL 2.2.2 - LM with movieId + ts_year####

set.seed(1, sample.kind = "Rounding")

movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

ts_year_avgs <- edx_train %>%
  left_join(movie_avgs, by='movieId')%>%
  group_by(ts_year)%>%
  summarize(b_tsy = mean(rating-mu-b_i))

predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(ts_year_avgs, by='ts_year') %>%
  mutate(pred = mu + b_i + b_tsy) %>%
  .$pred

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu+b_i)

model_2.2.2_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.2.2 - linear movieId + ts_year",
                                     RMSE = model_2.2.2_rmse ))
rmse_results %>% knitr::kable()


####MODEL 2.2.3 - LM with userId + genres####

set.seed(1, sample.kind = "Rounding")

genres_avgs <- edx_train %>% 
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_gen = mean(rating - mu - b_u)) 

predicted_ratings <- edx_test %>% 
  left_join(genres_avgs, by='genres') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_gen + b_u) %>%
  .$pred

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu+b_i)

model_2.2.3_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.2.3 - linear userId + genres",
                                     RMSE = model_2.2.3_rmse ))
rmse_results %>% knitr::kable()


####MODEL 2.2.4 - LM with userId + rel_year####

set.seed(1, sample.kind = "Rounding")

rely_avgs <- edx_train %>% 
  left_join(user_avgs, by='userId') %>%
  group_by(rel_year) %>%
  summarize(b_rely = mean(rating - mu - b_u)) 

predicted_ratings <- edx_test %>% 
  left_join(rely_avgs, by='rel_year') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_rely + b_u) %>%
  .$pred

predicted_ratings <- ifelse(!is.na(predicted_ratings),predicted_ratings, mu+b_i)

model_2.2.4_rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.2.4 - linear userId + rel_year",
                                     RMSE = model_2.2.4_rmse ))
rmse_results %>% knitr::kable()



####MODEL 2.3.1 - LM with userId + movieId + genre

gc() #garbage collection to free memory

set.seed(1, sample.kind = "Rounding")

genre_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u)) 

pred_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres')%>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

pred_ratings <- ifelse(!is.na(pred_ratings),pred_ratings, mu+b_i+b_u)

model_2.3.1_rmse <- RMSE(edx_test$rating, pred_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.3.1 - linear movieId + userId + genre",
                                     RMSE = model_2.3.1_rmse ))
rmse_results %>% knitr::kable()



####MODEL 2.3.2 - LM with userId + movieId + rel_year

gc() #garbage collection to free memory

set.seed(1, sample.kind = "Rounding")

rely_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(rel_year) %>%
  summarize(b_rely = mean(rating - mu - b_i - b_u)) 

pred_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(rely_avgs, by='rel_year')%>%
  mutate(pred = mu + b_i + b_u + b_rely) %>%
  .$pred

pred_ratings <- ifelse(!is.na(pred_ratings),pred_ratings, mu+b_i+b_u)

model_2.3.2_rmse <- RMSE(edx_test$rating, pred_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.3.2 - linear movieId + userId + rel_year",
                                     RMSE = model_2.3.2_rmse ))
rmse_results %>% knitr::kable()


####MODEL 2.3.3 - LM with userId + movieId + ts_year

gc() #garbage collection to free memory

set.seed(1, sample.kind = "Rounding")

ts_year_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(ts_year) %>%
  summarize(b_tsy = mean(rating - mu - b_i - b_u)) 

pred_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(ts_year_avgs, by='ts_year')%>%
  mutate(pred = mu + b_i + b_u + b_tsy) %>%
  .$pred

pred_ratings <- ifelse(!is.na(pred_ratings),pred_ratings, mu+b_i+b_u)

model_2.3.3_rmse <- RMSE(edx_test$rating, pred_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.3.3 - linear movieId + userId + rel_year",
                                     RMSE = model_2.3.3_rmse ))
rmse_results %>% knitr::kable()

####MODEL 2.4.1 - LM with movieId + userId + genre + rel_year

gc() #garbage collection to free memory

set.seed(1, sample.kind = "Rounding")

rel_year_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres')%>%
group_by(rel_year) %>%
  summarize(b_ry = mean(rating - mu - b_i - b_u - b_g)) 

pred_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres')%>%
  left_join(rel_year_avgs, by='rel_year')%>%
mutate(pred = mu + b_i + b_u + b_g + b_ry) %>%
  .$pred

pred_ratings <- ifelse(!is.na(pred_ratings),pred_ratings, mu+b_i+b_u+b_g)

model_2.4.1_rmse <- RMSE(edx_test$rating, pred_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.4.1 - linear movieId + userId + genre + rel_year",
                                     RMSE = model_2.4.1_rmse ))
rmse_results %>% knitr::kable()




####Model 2.4.2 - movieId + userId + genres + ts_year####

gc() #garbage collection to free memory

set.seed(1, sample.kind = "Rounding")

ts_year_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres')%>%
  group_by(ts_year) %>%
  summarize(b_tsy = mean(rating - mu - b_i - b_u - b_g)) 

pred_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres')%>%
  left_join(ts_year_avgs, by='ts_year')%>%
  mutate(pred = mu + b_i + b_u + b_g + b_tsy) %>%
  .$pred

pred_ratings <- ifelse(!is.na(pred_ratings),pred_ratings, mu+b_i+b_u+b_g)

model_2.4.2_rmse <- RMSE(edx_test$rating, pred_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2.4.2 - linear movieId + userId + genre + ts_year",
                                     RMSE = model_2.4.2_rmse ))
rmse_results %>% knitr::kable()




####MODEL2.6 - REGULARIZED LINEAR MODEL####

####use cross validation to find the best lambda for model 2.4.1
gc()

set.seed(1, sample.kind = "Rounding")

lambdas <- seq(1, 5, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating)
  b_i <- edx_train %>% #penalize movieId with few n()
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx_train %>% #penalize userId with few n()
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- edx_train %>% ###penalize genres with few n()
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres)%>%
    summarize(b_g = sum(rating-b_i-b_u-mu)/(n()+l))
  b_ry <- edx_train %>% ###penalize rel_years with few n()
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres')%>%
    group_by(rel_year) %>%
    summarize(b_ry = mean(rating - mu - b_i - b_u - b_g)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_ry, by ="rel_year")%>%
    mutate(pred = mu + b_i + b_u + b_g + b_ry) %>%
    .$pred
  return(RMSE(edx_test$rating, predicted_ratings))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized 2.4.1 - movie+user+genre+rel_year",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

####MODEL 3.1 - knn#### 
#EXTREMELY LONG RUNTIME ==> 2 days...not working
set.seed(1, sample.kind = "Rounding")
train_knn <- train(rating ~ userId+movieId+ts_year+rel_year, method = "knn", 
                   data = edx_train,
                   #tuneGrid = data.frame(k = seq(9, 51, 3))
                   )
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
train_knn$finalModel
min(train_knn$results$RMSE)
model3.1_predict <- predict(train_knn, edx_test, type = "raw")
model_3.1_rmse <- RMSE(edx_test$rating, model3.1_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="knn",
                                     RMSE = model_3.1_rmse))

rmse_results %>% knitr::kable()


####MODEL 3.2 - knn-cross-validation#### 
# NOT WORKING

set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(rating ~ userId+movieId+rel_year, method = "knn", 
                      data = edx_train,
                     # tuneGrid = data.frame(k = seq(9, 51, 3)),
                      trControl = control)
ggplot(train_knn_cv, highlight = TRUE)

train_knn$results %>% 
  ggplot(aes(x = k, y = RMSE)) +
  geom_line() +
  geom_point()

model3.2_predict <- predict(train_knn_cv, edx_test, type = "raw")
model_3.2_rmse <- RMSE(edx_test$rating, model3.2_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="knn_CV",
                                     RMSE = model_3.2_rmse))


####MODEL 4 - DT ####
# NOT WORKING - lack of memory
set.seed(1, sample.kind = "Rounding")
train_dt <- train(rating ~userId+movieId+rel_year+ts_year+genres, 
                  method = "rpart", 
                  #tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                  metric="RMSE",
                  data = edx_train)
train_dt$bestTune
model4_predict <- predict(train_dt, edx_test)



model_4_rmse <- RMSE(edx_test$rating, model4_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Decision tree",
                                     RMSE = model_4_rmse ))



####MODEL 5 - LOESS#### 

train_loess <- train(rating ~userId+movieId, 
                     method = "gamLoess", 
                     data = edx_train)
model5_predict <- predict(train_loess, edx_test)

model_5_rmse <- RMSE(edx_test$rating, model5_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="5 - gamLOESS",
                                     RMSE = model_5_rmse ))



####MODEL 6 - Random Forest####
##NOT WORKING - Lack of memory
train_rf <- train(rating ~userId+movieId+rel_year+ts_year+genres, 
                  method = "rf", 
                 # tuneGrid = data.frame(mtry = seq(1:7)),
                  ntree = 5,
                  data = edx_train)
train_rf$bestTune
model6_predict <- predict(train_rf, edx_test)


model_6_rmse <- RMSE(edx_test$rating, model6_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="random forest",
                                     RMSE = model_6_rmse ))
rmse_results %>% knitr::kable()



####MODEL 7 - GLM####

train_glm <- train(rating ~userId+movieId, 
                   method = "glm", 
                   data = edx_train)
train_glm$bestTune
model7_predict <- predict(train_glm, edx_test)


model_7_rmse <- RMSE(edx_test$rating, model7_predict)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 7 - GLM",
                                     RMSE = model_7_rmse ))
rmse_results %>% knitr::kable()


#########VALIDATE BEST MODEL#########


l <- 4.75 #value based on the cross-validation of model 1.4.1
  b_i <- edx_train %>% #penalize movieId with few n()
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx_train %>% #penalize userId with few n()
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- edx_train %>% ###penalize genres with few n()
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres)%>%
    summarize(b_g = sum(rating-b_i-b_u-mu)/(n()+l))
  b_ry <- edx_train %>% ###penalize rel_years with few n()
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres')%>%
    group_by(rel_year) %>%
    summarize(b_ry = mean(rating - mu - b_i - b_u - b_g)/(n()+l))
 
  #  
  final_predict <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_ry, by ="rel_year")%>%
    mutate(pred = mu + b_i + b_u + b_g + b_ry) %>%
    .$pred
  
summary(final_predict) #look for NA´s in the data
validation %>% filter(is.na(final_predict))%>%.$title #check titles of movies w/NA´s
final_predict <- replace(final_predict, is.na(final_predict), mu) #exchange NA´s with avg rating
  
  final_model_rmse <- RMSE(validation$rating, final_predict)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="VALIDATION - Regularized 2.4.1 - movie+user+genre+rel_year",  
                                     RMSE = final_model_rmse))
rmse_results %>% knitr::kable()


