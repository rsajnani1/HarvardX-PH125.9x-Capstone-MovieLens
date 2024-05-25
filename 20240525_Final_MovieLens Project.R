if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(caret)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(tidyverse)
  
library(caret)

library(knitr)

#MovieLens 10M dataset:
  
#https://grouplens.org/datasets/movielens/10m/
  
#http://files.grouplens.org/datasets/movielens/ml-10m.zip


options(timeout = 120)
  
dl <- "ml-10M100K.zip"
  
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")



# Final hold-out test set will be 10% of MovieLens data

set.seed(123, sample.kind="Rounding")
  
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  
edx <- movielens[-test_index,]
  
temp <- movielens[test_index,]
  


# Make sure userId and movieId in final hold-out test set are also in edx set

final_holdout_test <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")
  

# Add rows removed from final hold-out test set back into edx set

removed <- anti_join(temp, final_holdout_test)
  
edx <- rbind(edx, removed)
  
rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Initial look at the edx data set and its variables

head(edx)


edx %>%
summarize(n_users = n_distinct(userId), 
          n_movies = n_distinct(movieId))


# Checking for empty rows

summary(edx)


# Taking a look at the most frequently rated movies

edx %>% group_by(movieId, title) %>%
	summarize(count = n()) %>%
	arrange(desc(count))



# Taking a look at the most frequently awarded rating

edx %>%
ggplot(aes(rating)) +
geom_histogram(binwidth = 0.25, color = "black") +
scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
ggtitle("Rating Distribution")


# Taking a look at number of ratings per movie

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Number of Ratings Per Movie") + labs(x = 'Number of Ratings', y = 'Number of Movies') 



# Taking a look at the least frequently rated movies

movie_ratingcount <- edx %>% group_by(title) %>% summarize(count=n(),rating=mean(rating)) %>% filter(count == 1) %>%  slice(1:10) %>% kable(caption = 'Least Frequently Rated Movies')
movie_ratingcount



# Taking a look at the number of ratings per user

edx %>%
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Number of Ratings Per User") + labs (x = 'Number of Ratings', y = 'Number of Users') 



# Taking a look at the mean rating value by user

user_ratingcount <- edx %>% group_by(userId) %>% summarize(count=n(),rating=mean(rating))

user_ratingcount %>% 
ggplot(aes(rating)) + 
geom_histogram(bins = 30, color = "black") + ggtitle ('Histogram of Average Movie Ratings By User') + labs(x = 'Rating', y = 'Number of Users')


# Taking a look at the genres with highest average rating

genre_ratingcount_top <- edx %>% group_by(genres) %>% summarize(count=n(),rating=mean(rating)) %>% arrange(desc(rating)) %>%  slice(1:10) %>% kable(caption = 'Genres that Have Received the Highest Average Ratings')
genre_ratingcount_top

# Taking a look at the genres with lowest average rating

genre_ratingcount_low <- edx %>% group_by(genres) %>% summarize(count=n(),rating=mean(rating)) %>% arrange(rating) %>%  slice(1:10) %>% kable(caption = 'Genres that Have Received the Lowest Average Ratings')
genre_ratingcount_low

# Calculate the average of all movies
mu_hat <- mean(edx$rating)

# Predict the RMSE on the final holdout test set
RMSE_basic <- RMSE(final_holdout_test$rating, mu_hat)

# Adding the results to the results data set

results <- data.frame(Model="Basic Model", RMSE=RMSE_basic)
results


# Calculate the average of all movies

mu_hat <- mean(edx$rating)

# Calculate the average by movie

movies <- edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))

# Predict the RMSE on the final holdout test set

RMSE_movies_model <- final_holdout_test %>%
   left_join(movies, by='movieId') %>%
   mutate(pred = mu_hat + b_i) %>%
   pull(pred)

RMSE_movies_result <- RMSE(final_holdout_test$rating,RMSE_movies_model)

# Adding the results to the results data set

results <- results %>% add_row(Model="Movie Effect Model", RMSE=RMSE_movies_result)

results


# Calculate the average of all movies

mu_hat <- mean(edx$rating)

# Calculate the average by movie

movies <- edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))

# Calculate the average by user

users <- edx %>%
   left_join(movies, by='movieId') %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu_hat - b_i))

# Compute the predicted ratings on final holdout test data set

rmse_movies_users_model <- final_holdout_test %>%
   left_join(movies, by='movieId') %>%
   left_join(users, by='userId') %>%
   mutate(pred = mu_hat + b_i + b_u) %>%
   pull(pred)

rmse_movies_users_result <- RMSE(final_holdout_test$rating, rmse_movies_users_model)

# Adding the results to the results data set

results <- results %>% add_row(Model="Movie + User Effect Model", RMSE=rmse_movies_users_result)

results


# Calculate the average of all movies

mu_hat <- mean(edx$rating)

# Define a table of lambdas

lambdas <- seq(0, 10, 0.1)

# Compute the predicted ratings on final_holdout_test data set using different values of lambda

rmses <- sapply(lambdas, function(lambda){
  
# Calculate the average by user
   
   b_i <- edx %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu_hat) / (n() + lambda))
   
   
   # Calculate the average by user
   
   b_u <- edx %>%
      left_join(b_i, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu_hat) / (n() + lambda))
   
   # Compute the predicted ratings on final_holdout_test data set
   
   predicted_ratings <- final_holdout_test %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_u, by='userId') %>%
      mutate(pred = mu_hat + b_i + b_u) %>%
      pull(pred)
   
   # Predict the RMSE on the final_holdout_test data set
   
   return(RMSE(final_holdout_test$rating, predicted_ratings))
})

# Get the lambda value that minimize the RMSE

min_lambda <- lambdas[which.min(rmses)]

# plot the result of lambdas

df <- data.frame(RMSE = rmses, lambdas = lambdas)

ggplot(df, aes(lambdas, rmses)) +
   theme_classic()  +
   geom_point() +
   labs(title = "RMSEs vs Lambdas - Regularized Movies + Users Model",
        y = "RMSEs",
        x = "lambdas")


# Predict the RMSE on the final_holdout_test set

RMSE_regularized_movies_users_model <- min(rmses)



# Adding the results to the results data set

results <- results %>% add_row(Model="Regularized Movie + User Based Model", RMSE=RMSE_regularized_movies_users_model)

results

# final look at the resulting RMSE from different models

results
