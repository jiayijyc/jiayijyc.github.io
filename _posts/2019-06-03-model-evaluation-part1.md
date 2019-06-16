---
layout: post
title:  "Model Evaluation - Part 1"
---


# **Introduction**

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/DataScience1.jpg)

The world has witnessed an increasingly important role that machine learning (ML) plays in both academic and business areas to analyze data, make predictions, make data-driven recommendations and decisions. While designing an algorithm and training a model is a key step, the model evaluation plays an essential role as well and cannot be neglected in machine learning pipeline. The problems we need to figure out in model evaluation include but not limited to 

* Is the model accurate enough and can we trust the predictions the model makes?
* Could the model be just memorizing the training data, and consequently not able to make good predictions on unseen data?
* Which model should I choose among the pool of candidates? 

In this article, we will introduce a common technique used to assess how well a model genralizes to out-of-sample data. Additionally, we will explain the model evaluation metrics for supervised learning. The cluster validity techniques will be illustrated in Part 2 of this series.

# **Model Evaluation Technique**
Once a new model is trained, it can't be assumed that the model will achieve the desired accuracy and variance on the unseen data. One important technique for testing the model on unseen data to check whether it's underfitting/overfitting/well generalized is cross validation (CV).

Here are two common techniques used in CV.

* **Train_Test Split Approach**
The data will be split randomly into train and test set, idealy split the data to 70:30 or 80:20. The training model will be trained on train set and validation will be performed on test set. However, this method will cause high bias if the data size is small since less data will be used to train the model.


* **K-folds Cross Validation**

K-folds cross validation is one of the best approaches for limited data. The method goes as follows.

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/K-foldCV.png)

1. Split the whole data set randomly into K folds, where the choice of K is based on data size (ideally 5 or 10).
2. Train the model on K-1 folds and do the validation using the remaining Kth fold. Calculate scores/erros.
3. Repeat this process until each fold serve as the test set. Abtain the final performance metric by taking the average of K scores/errors.

The various score/error metrics will be introduced in below sections.

# **Model Evaluation Metrics - Supervised Learning**

## **1 Regression**
### **1.1 Mean Absolute Error (MAE)**

$$MAE = \frac{1}{n}\sum_{j=1}^n|y_{j}-\hat{y_{j}}|$$

MAE is calculated as the average of the absolute differences between prediction and actual observation.

### **1.2 Root Mean Squared Error (RMSE)** 

$$RMSE = \sqrt{\frac{1}{n}\sum_{j=1}^n (y_{j}-\hat{y_{j}})^2}$$

RMSE is calculated as the square root of the average of squared differences between prediction and actual observations.

**Comparison of MAE and RMSE**

**Similarities**
 * They range from 0 to infinite and lower values are better.
 * They express average prediction error in units of the variable of interest.
 * They are indifferent to the directions of errors.

**Differences**
 * Compared to MAE, RMSE penalizes large errors more as it takes square of errors.
 * For MAE, the method of taking absolute value is undesiable in many mathematical calculations (e.g, differentiation of error function).
 * It is easier to interprete MAE.


### **1.3 Coeficient of Determination ($$R^2$$)** 

$$R^2=1-\frac{SSE}{SST}$$

$$Sum\ of\ Squares\ Errors:\ SSE=\sum_{i=0}^n (y_{i}-\hat{y_{i}})^2$$

$$Sum\ of\ Squares\ Total:\ SST=\sum_{i=0}^n (y_{i}-\overline{y_{i}})^2$$


The **Coeficient of Determination ($$R^2$$)** summarizes the explanatory power of the regression model. It decribes the proportion of variance explained by the model, or reduction in error over null model.

* $$R^2$$ ranges from 0 to 1. The closer to 1, the better the regression model.
* If the regression model is a total failure, SSE is equal to SST, and $$R^2$$ is zero.
* If the regression model is a total failure, SSE is 0, and $$R^2$$ is 1.

Howerver, there are some issues with $$R^2$$.
* We can't use $$R^2$$ to determin if coefficients estimates and predictions are biased.
* $$R^2$$ always increase as the number of features increase, which will lead to overfitting. One possible solution is to use adjusted $$R^2$$.

$$adjusted\ R^2 = 1 - (1- R^2)(\frac{n-1}{n-(k+1)})$$

$$where\ k\ is\ number\ of\ features$$


### **1.4 Standardized Residuals Plot**
The **Standardardized Residual Plot** is the plot of a series of standardized residuals ($$d_{i}$$).

$$d_{i}=\frac{e_{i}}{S_{e}}$$

$$e_{i}=y_{i}-\hat{y_{i}}$$

$$S_{e}=\sqrt{\frac{SSE}{n-k-1}}$$

The regression model is good if the Standardardized Residual Plot shows
* no pattern.
* flunctuation around zero.
* randomness and unpredictability.


## **2 Classification**

### **2.1 Confusion Matrix**

The **confusion matrix** is N$$\times$$N matrix, where N is the number of classes. For N=2, the 2$$\times$$2 matrix is shown as below. 

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/MatrixConfusion.png)


Here are few definitions.

 * **Accuracy**

$$\frac{TP+TN}{TP+FP+FN+TN}$$

It is the proportion of the total number of predictions that were correct. It doesn't provide much value when the data is highly biased one class (e.g, 99% of data belong to class 1).

* **Precision**

$$\frac{TP}{TP+FP}$$

It is the proportion of positive cases that are correctly identified.


* **Recall/Sensitivity/True Positive rate**: 

$$\frac{TP}{TP+FN}$$

It is the proportion of actual positive cases that are correctly identified.

* **Specifity / True Negative Rate**

$$\frac{TN}{TN+FP}$$

It is the proportion of actual negative cases that are correctly identified.

* **F-beta Score**

$$F_{\beta}=\frac{(1+\beta^2)\times Precision\times Recall}{\beta^2 \times Precision + Recall}$$

$$\beta$$ ranges from 0 to $$\propto$$. When $$\beta <1$$, it gives more weights to precision ($$\beta = 0$$: $$F_{\beta}=$$ precision). When $$\beta>1$$, it gives more weights to recall ($$\beta \to \propto$$: $$F_{\beta}=$$ recall). When $$ \beta = 1$$, it becomes harmonic mean of precision and recall.


### **2.2 ROC/AUC (Area Under ROC Curve)**
The ROC curve is the plot between true positive rate and false positive rate, where

$$True\ Positive\ Rate = \frac{TP}{all\ actual\ positives}$$

$$False\ Positive\ Rate = \frac{FP}{all\ actual\ negatives}$$

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/ROC.png)

The ROC curve is shown as above. The blue line shows how TPR changes according to different FTP. The TPR increases fast at first and slows down as FPR increases. The diagonal line shows result for random model. We should choose the model with TPR and FPR at 'turning point' at left upper coner. Moreover, the area under the ROC curve is calculated as AUC. The larger the AUC, the more accuracy of the model.

### **2.3 Gini Coefficient**

It is the ratio between the ROC curve and the disgnal line and the area of the above triangle. It can be calculated as 

$$Gini = 2AUC -1$$

Commonly, a gini coeafficient with value more than 60% denotes a good model. 


### **2.4 Concordant-Disconcordant Ratio**

It identifies the ability of the model to differentiate between event happening and not happening. Let's take a 2 class logistic regression classification as an example. The predicted probabilities for event(1) and event(0) are 

$$P_{i}, i=1,2,...,5\ for\ event(1)$$

$$P_{j}, j=1,2,...,5\ for\ event(0)$$

We can get 25 pairs of (1,0) with $$(P_{i},P_{j})$$,
* concordant pairs are pairs with $$P_{i} > P_{j}$$.
* disconcordant pairs are pairs with $$P_{i} < P_{j}$$.
* tied pairs are pairs with $$P_{i} = P_{j}$$.

$$concordant\ ratio =\frac{No.\ of\ concordant\ pairs}{No.\ of\ total\ pairs}$$

The larger the concordant ratio, the better the model.



### **2.5 Lift and Gain Chart**

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/LifeAndGain.png)

The above cumulative gains and lift charts can be used for measuring model performance. 
* The y-axis shows the percentage of positive responses. 
* The x-axis shows the percentage of data points contacted.
* The red line is **baseline**: If we contact X% of data points then we will receive X% of the total positive responses.
* The blue line is **lift curve**: using the predictions of the response model, calculate the percentage of positive responses for the percent of data points contacted and map these points to create the lift curve.

So, the greater the area between the life curve and the baseline, the better the model.


