# Project
Steven D. Rankine  
Sunday, November 16, 2014  
## Executive Summary
You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did.

## Reproducible Environment
The following chunk of code loads the libraries used in the analysis of the data.  Also, by setting a seed to a fixed value, will guarantee that the all the code in this report will produce the same results as another person running the code.


```r
# Load needed libraries
library(ggplot2);
library(lattice);
library(caret);
library(rpart);
library(rpart.plot);
library(randomForest);
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(rattle);
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.3.0 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
# Define seed to reproduce "randomness"
set.seed(1989);  
```

## Data
Test subjects were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [More Info](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).
The training data for this project are available here:  
[Training Data Link](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
The test data are available here:  
[Testing Data Link](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)  
Important: you are free to use this dataset for any purpose. This dataset is licensed under the Creative Commons license (CC BY-SA). The CC BY-SA license means you can remix, tweak, and build upon this work even for commercial purposes, as long as you credit the authors of the original work and you license your new creations under the identical terms we are licensing to you. This license is often compared to "copyleft" free and open source software licenses. All new works based on this dataset will carry the same license, so any derivatives will also allow commercial use.

<cite>Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.</cite>

Read in test and training data.

```r
# Read in raw training and testing data
testing <- read.csv("data/pml-testing.csv", header = TRUE)
training <- read.csv("data/pml-training.csv", header = TRUE)
dim(training)
```

```
## [1] 19622   160
```

## Data Preparation
Not all of the 160 variables in the training dataset will be used as predictors for the prediction model.  Variables where the majority of its values are labeled as NA (missing) will be excluded. Also excluded, will be variables that are associated with the test subject's identity and variables related to date/time.

```r
# Define array with names of useless variables
drops <- c("amplitude_pitch_arm", "amplitude_pitch_belt", "amplitude_pitch_dumbbell", 
    "amplitude_pitch_forearm", "amplitude_roll_arm", "amplitude_roll_belt", 
    "amplitude_roll_dumbbell", "amplitude_roll_forearm", "amplitude_yaw_arm", 
    "avg_pitch_arm", "avg_pitch_belt", "avg_pitch_dumbbell", "avg_pitch_dumbbell", 
    "avg_pitch_forearm", "avg_roll_arm", "avg_roll_belt", "avg_roll_dumbbell", 
    "avg_roll_forearm", "avg_yaw_arm", "avg_yaw_belt", "avg_yaw_dumbbell", "avg_yaw_forearm", 
    "cvtd_timestamp", "max_picth_arm", "max_picth_belt", "max_picth_dumbbell", 
    "max_picth_forearm", "max_roll_arm", "max_roll_belt", "max_roll_dumbbell", 
    "max_roll_forearm", "max_yaw_arm", "min_pitch_arm", "min_pitch_belt", "min_pitch_dumbbell", 
    "min_pitch_forearm", "min_roll_arm", "min_roll_belt", "min_roll_dumbbell", 
    "min_roll_forearm", "min_yaw_arm", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "stddev_pitch_arm", "stddev_pitch_belt", "stddev_pitch_dumbbell", "stddev_pitch_forearm", 
    "stddev_roll_arm", "stddev_roll_belt", "stddev_roll_dumbbell", "stddev_roll_forearm", 
    "stddev_yaw_arm", "stddev_yaw_belt", "stddev_yaw_dumbbell", "stddev_yaw_forearm", 
    "user_name", "var_accel_arm", "var_accel_dumbbell", "var_accel_forearm", 
    "var_pitch_arm", "var_pitch_belt", "var_pitch_dumbbell", "var_pitch_forearm", 
    "var_roll_arm", "var_roll_belt", "var_roll_dumbbell", "var_roll_forearm", 
    "var_total_accel_belt", "var_yaw_arm", "var_yaw_belt", "var_yaw_dumbbell", 
    "var_yaw_forearm", "X")
# Filter out useless variables
training.filt <- training[, !(names(training) %in% drops)]
testing.filt <- testing[, !(names(training) %in% drops)]
dim(training.filt)
```

```
## [1] 19622    88
```
By removing these variables, the number of possible predictors to sort through is 87.

### Zero or Near Zero Variance Predictors
Create subset of training data that removes variables whose variance are small.

```r
nzv <- nearZeroVar(training.filt)
training.filt <- training.filt[, -nzv]
testing.filt <- testing.filt[, -nzv]
dim(training.filt)
```

```
## [1] 19622    54
```

### Correated Predictors
Reduce the level of correlation between the predictors. The code chunk below shows the effect of removing descriptors with absolute correlations above 0.75. 

```r
corr <- findCorrelation(cor(training.filt[, 1:53]), cutoff = 0.75)
training.filt <- training.filt[, -corr]
testing.filt <- testing.filt[, -corr]
dim(training.filt)
```

```
## [1] 19622    34
```
The number of possible predictors are now at 33.  This will be the final set of predictors used to build the eventual prediction model.

## Model




```r
# fancyRpartPlot(model3$finalModel);

# confusionMatrix(model1); confusionMatrix(model2); confusionMatrix(model3);
# confusionMatrix(model4); confusionMatrix(model5);

# pred1 <- predict(model1,newdata=testing.filt); pred2 <-
# predict(model2,newdata=testing.filt); pred3 <-
# predict(model3,newdata=testing.filt); pred4 <-
# predict(model4,newdata=testing.filt); pred5 <-
# predict(model5,newdata=testing.filt);
```
### Cross-Validation
### Out of Sample Error
## Conclusion


 
