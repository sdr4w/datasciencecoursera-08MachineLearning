# Project
Steven D. Rankine  
Sunday, November 16, 2014  
## Executive Summary
The best prediction model was built when using the "kth nearest neighbor" method to train the model. Up to 93% accuracy was achieved on the training data.  

## Reproducible Environment
The following chunk of code loads the libraries used in the analysis of the data.  Also, by setting a seed to a fixed value, will guarantee that the all the code in this report will produce the same results as another person running the code.


```r
# Load needed libraries
library(ggplot2);
library(lattice);
library(caret);
library(nnet);
library(rpart);
library(rpart.plot);
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
The data contain features of test subjects who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [More Info](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

     [Training Data Link](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  

The test data are available here:  

     [Testing Data Link](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)  

### Credit

__Important__: you are free to use this dataset for any purpose. This dataset is licensed under the Creative Commons license (CC BY-SA). The CC BY-SA license means you can remix, tweak, and build upon this work even for commercial purposes, as long as you credit the authors of the original work and you license your new creations under the identical terms we are licensing to you. This license is often compared to "copyleft" free and open source software licenses. All new works based on this dataset will carry the same license, so any derivatives will also allow commercial use.

<cite>Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.</cite>

### Load data

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
Data set contains 159 predictor variables and one outcome variable. The outcome variable will have the label __classe__. 

### Discard Useless Predictors 
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
    "min_roll_forearm", "min_yaw_arm", "num_window", "raw_timestamp_part_1", 
    "raw_timestamp_part_2", "stddev_pitch_arm", "stddev_pitch_belt", "stddev_pitch_dumbbell", 
    "stddev_pitch_forearm", "stddev_roll_arm", "stddev_roll_belt", "stddev_roll_dumbbell", 
    "stddev_roll_forearm", "stddev_yaw_arm", "stddev_yaw_belt", "stddev_yaw_dumbbell", 
    "stddev_yaw_forearm", "user_name", "var_accel_arm", "var_accel_dumbbell", 
    "var_accel_forearm", "var_pitch_arm", "var_pitch_belt", "var_pitch_dumbbell", 
    "var_pitch_forearm", "var_roll_arm", "var_roll_belt", "var_roll_dumbbell", 
    "var_roll_forearm", "var_total_accel_belt", "var_yaw_arm", "var_yaw_belt", 
    "var_yaw_dumbbell", "var_yaw_forearm", "X")
# Filter out useless variables
training.filt <- training[, !(names(training) %in% drops)]
dim(training.filt)
```

```
## [1] 19622    87
```

```r
rm(drops)
```
By removing these variables, the number of possible predictors to sort through is 86.

### Discard Zero or Near Zero Variance Predictors
Create subset of training data that removes variables whose variance are small.

```r
nzv <- nearZeroVar(training.filt[, !(names(training.filt) %in% c("classe"))])
training.nzv <- training.filt[, -nzv]
dim(training.nzv)
```

```
## [1] 19622    53
```

```r
rm(nzv)
```

### Discard Highly Correlated Predictors
Reduce the level of correlation between the predictors. The code chunk below shows the effect of removing descriptors with absolute correlations above 0.75. 


```r
corr <- cor(training.nzv[, !(names(training.nzv) %in% c("classe"))])
training.corr75 <- training.nzv[, -findCorrelation(corr, cutoff = 0.75)]
training.corr50 <- training.nzv[, -findCorrelation(corr, cutoff = 0.5)]
dim(training.corr75)
```

```
## [1] 19622    33
```

```r
dim(training.corr50)
```

```
## [1] 19622    22
```

```r
rm(corr)
```
The number of possible predictors are now at 86.  This will be the final set of predictors used to build the eventual prediction model.

## Models
__caret__ has a function called __train__ for use over 147 models with different resampling methods.  Using the training data to build 3 prediction models with the __train__ function. The performance of each will be evaluated.  Only 3 models were built because The computing/memory capacity of the system used to perform this analysis limits the scope of the models that can be reasonably calculated.  The 3 models built were:

1. Kth Nearest Neigbhor
2. Neural Network
3. Tree Partition



### Classification Performance
Display the "confusion matrix" for each of the created models.  A “confusion matrix” is a cross–tabulation of the observed and predicted classes. The higher the sum of the values across the table's diagonal then the better that overall model's overall classification performance. 


```r
confusionMatrix(model1);  # kth Nearest Neighbors
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 27.6  0.6  0.2  0.1  0.1
##          B  0.4 17.6  0.4  0.1  0.3
##          C  0.2  0.7 16.1  0.8  0.3
##          D  0.2  0.2  0.6 15.2  0.2
##          E  0.0  0.2  0.2  0.1 17.4
```

```r
confusionMatrix(model2);  # Neural Network
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 23.5  3.5  1.5  1.3  1.2
##          B  1.7  9.1  1.3  1.4  3.0
##          C  1.4  2.9 12.3  2.2  2.8
##          D  1.5  1.6  1.5 10.1  2.0
##          E  0.5  2.1  0.9  1.3  9.4
```

```r
confusionMatrix(model3);  # Decision Trees
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 19.5  3.0  0.6  1.0  0.6
##          B  1.3  8.5  1.6  1.6  2.7
##          C  4.4  4.9 13.5  5.4  3.4
##          D  1.8  1.6  0.6  6.4  1.0
##          E  1.5  1.2  1.1  2.1 10.6
```

### Prediction on Testing Dataset
Apply models against the testing dataset.


```r
p1 <- predict(model1, newdata = testing)
p2 <- predict(model2, newdata = testing)
p3 <- predict(model3, newdata = testing)
```

The prediction results for the three models are shown in the table below.

Case | Model#1 | Model#2 | Model#3
---- | ------- | ------- | -------
  1  |    B    |    B    |    C  
  2  |    A    |    A    |    A 
  3  |    A    |    B    |    C 
  4  |    A    |    D    |    A 
  5  |    A    |    B    |    C 
  6  |    C    |    C    |    C 
  7  |    D    |    D    |    C 
  8  |    B    |    D    |    A 
  9  |    A    |    A    |    A 
 10  |    A    |    A    |    C 
 11  |    B    |    C    |    B 
 12  |    C    |    C    |    C 
 13  |    B    |    B    |    C 
 14  |    A    |    A    |    A 
 15  |    E    |    E    |    C 
 16  |    E    |    E    |    A 
 17  |    A    |    A    |    A 
 18  |    B    |    B    |    B  
 19  |    B    |    B    |    A  
 20  |    B    |    E    |    C   

## Conclusion
The prediction model built using the K–Nearest Neighbors method had the best classification out of the 3 that were studied.

### Cross-Validation
K–Nearest Neighbors method incorporates cross validation over bootstrapped samples.

### Assignment Writeup
Apply the best machine learning model built to each of the 20 test cases in the testing data set. For each test case create a text file with a single capital letter (A, B, C, D, or E) corresponding to the prediction for the corresponding problem in the test data set. 


```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("text/problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(as.character(pred1))
```

***

## Appendix
Plot of Tr

```r
fancyRpartPlot(model3$finalModel)
```

![](./Project_files/figure-html/appendix-1.png) 

 
