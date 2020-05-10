#NOTE: you will need the internet to download the required datasets
# ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,





##title: "MovieLens Recommendation System Project"
##author: "Mohammed Chihab"
##date: "5/5/2020"

#______ SECTION 01 __________________________________________________________________________
#Install or load the required libraries
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")


#______ SECTION 02 __________________________________________________________________________
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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


#______ SECTION 03 __________________________________________________________________________
###########################
# Exploring the edx dataset
###########################

# first three rows in the edx dataset
head(edx, 3)


# last three rows in the edx dataset
tail(edx, 3)


# edx dimentions:
dim(edx)

# The different movies in the edx dataset
n_distinct(edx$movieId)


# The different users in the edx dataset
n_distinct(edx$userId)


# The different genres combinations in the edx dataset
n_distinct(edx$genres)


# The top 2 ratings in the edx dataset
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(2) %>%
  arrange(desc(count))


# The lowest 2 ratings in the edx dataset
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(-2) %>%
  arrange(count)


edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()


#______ SECTION 04 __________________________________________________________________________
#############################################
# Rename and create train sets and test sets
#############################################

#As the data will be randomly created, it is important to set the seed to a certain value
set.seed(1, sample.kind="Rounding")

# Decide on the validation test set and main train set
train_set <- edx
test_set <- validation


# Split the train_set into train_train_set and train_test_set
train_test_index <- createDataPartition(y = train_set$rating , times = 1, p = 0.1, list = FALSE)
train_train_set <- train_set[-train_test_index,]
train_test_set <- train_set[train_test_index,]

# left join, avoid duplication in rows amongst data
train_test_set <- train_test_set %>% 
  semi_join(train_train_set, by = "movieId") %>%
  semi_join(train_train_set, by = "userId")

# set up the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# set up lambdas, the tuning parameters
lambdas <- seq(1, 8, 2)

# cross validate using the tuning parameters, lambdas
rmses <- sapply(lambdas, function(lambda){
  
  ##The mean of ratings in the train_train_set
  mu <- mean(train_train_set$rating)
  
  ## The effect of the average rating for each movie
  e_i <- train_train_set %>% 
    group_by(movieId) %>%
    summarize(e_i = sum(rating - mu)/(n()+lambda))
  
  ## The effect of each user on ratings
  e_u <- train_train_set %>% 
    left_join(e_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(e_u = sum(rating - e_i - mu)/(n()+lambda))
  
  ## The effect of genres compbinations on ratings
  e_g <- train_train_set %>% 
    left_join(e_i, by="movieId") %>%
    left_join(e_u, by="userId") %>%
    group_by(genres) %>%
    summarize(e_g = sum(rating - e_i - e_u - mu)/(n()+lambda))
  
  ## pred: the predicted ratings on the basis of all the previous effects
  predicted_ratings <- 
    train_test_set %>% 
    left_join(e_i, by = "movieId") %>%
    left_join(e_u, by = "userId") %>%
    left_join(e_g, by = "genres") %>%
    mutate(pred = mu + e_i + e_u + e_g) %>%
    .$pred
  
  ##Retunrs RMSE for each tuning parameter
  return(RMSE(predicted_ratings, train_test_set$rating))
})

# Plot lambdas against rmses
qplot(lambdas, rmses)  

# show the minimum rmse
min(rmses)

#Choose the tuning parameter based on the minimum rmse and print it
lambda <- lambdas[which.min(rmses)]
lambda



#______ SECTION 05 __________________________________________________________________________
#########################################################################
# Using our pre-tested model on a simulated alien validation test datset
#########################################################################


#Set up the tuning parameter to chosen one
lambda <- 5

#Use the train_set(=edx dataset) as whole to make predictions on the test_set(=validation dataset)

#The mean of ratings in the train_set
mu <- mean(train_set$rating)

# The effect of the average rating for each movie
e_i <- train_set %>%
  group_by(movieId) %>%
  summarize(e_i = sum(rating - mu)/(n()+lambda))

# The effect of each user on ratings
e_u <- train_set %>% 
  left_join(e_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(e_u = sum(rating - e_i - mu)/(n()+lambda))

# The effect of genres compbinations on ratings
e_g <- train_set %>% 
  left_join(e_i, by="movieId") %>%
  left_join(e_u, by="userId") %>%
  group_by(genres) %>%
  summarize(e_g = sum(rating - e_i - e_u - mu)/(n()+lambda))

# pred: the predicted ratings on the basis of all the previous effects
predicted_ratings <- 
  test_set %>% 
  left_join(e_i, by = "movieId") %>%
  left_join(e_u, by = "userId") %>%
  left_join(e_g, by = "genres") %>%
  mutate(pred = mu + e_i + e_u + e_g) %>%
  .$pred

# display the first 10 entries of the true ratings and the predicted ratings, print the data
df <- data.frame(true_ratings = head(test_set$rating, 10), prediced_ratings = head(predicted_ratings, 10))
df


#______ SECTION 06 __________________________________________________________________________
#################################################################################################
# Use of RMSE to present our result and model performence on the novel foreign simulated dataset
#################################################################################################

##Calculates the RMSE for our predected data against the true data, and print it
rmse_based_on_model <- RMSE(predicted_ratings, test_set$rating)
rmse_based_on_model


