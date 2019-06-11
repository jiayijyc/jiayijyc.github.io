---
layout: post
title:  "Data Science"
---

# **II Model Evaluation**

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/DataScience1.jpg)

## **Introduction**
The world has witnessed an increasingly important role that machine learning (ML) plays in both academic and business areas to analyze data, make predictions, make data-driven recommendations and decisions. While designing an algorithm and training a model is a key step, the model evaluation plays an essential role as well and cannot be neglected in machine learning pipeline. The problems we need to figure out in model evaluation include but not limited to 

* Is the model accurate enough and can we trust the predictions the model makes?
* Could the model be just memorizing the training data, and consequently not able to make good predictions on unseen data?
* Which model should I choose among the pool of candidates? 

In this article, we will introduce a common technique used to assess how well a model genralizes to out-of-sample data. Additionally, we will explain the model evaluation metrics for both supervised and unsupervised learning. 

## **Model Evaluation Technique**
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

## **Model Evaluation Metrics**

## **1. Supervisned Learning**
### **1.1 Regression**
#### **1.1.1 Mean Absolute Error (MAE)**

$$MAE = \frac{1}{n}\sum_{j=1}^n|y_{j}-\hat{y_{j}}|$$

MAE is calculated as the average of the absolute differences between prediction and actual observation.

#### **1.1.2 Root Mean Squared Error (RMSE)** 

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


#### **1.1.3 Coeficient of Determination ($$R^2$$)** 

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


#### **1.1.4 Standardized Residuals Plot**
The **Standardardized Residual Plot** is the plot of a series of standardized residuals ($$d_{i}$$).

$$d_{i}=\frac{e_{i}}{S_{e}}$$

$$e_{i}=y_{i}-\hat{y_{i}}$$

$$S_{e}=\sqrt{\frac{SSE}{n-k-1}}$$

The regression model is good if the Standardardized Residual Plot shows
* no pattern.
* flunctuation around zero.
* randomness and unpredictability.


### 1.2 **Classification**

* **1.2.1 Confusion Matrix**

The **confusion matrix** is N$$\times$$N matrix, where N is the number of classes. For N=2, the 2$$\times$$2 matrix is shown as below. 

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/MatrixConfusion.png)


Here are few definitions.

 **Accuracy**

$$\frac{TP+TN}{TP+FP+FN+TN}$$

It is the proportion of the total number of predictions that were correct. It doesn't provide much value when the data is highly biased one class (e.g, 99% of data belong to class 1).

**Precision**

$$\frac{TP}{TP+FP}$$

It is the proportion of positive cases that are correctly identified.


**Recall/Sensitivity/True Positive rate**: 

$$\frac{TP}{TP+FN}$$

It is the proportion of actual positive cases that are correctly identified.

**Specifity / True Negative Rate**

$$\frac{TN}{TN+FP}$$

It is the proportion of actual negative cases that are correctly identified.

**F-beta Score**

$$F_{\beta}=\frac{(1+\beta^2)\times Precision\times Recall}{\beta^2 \times Precision + Recall}$$

$$\beta$$ ranges from 0 to $$\propto$$. When $$\beta <1$$, it gives more weights to precision ($$\beta = 0$$: $$F_{\beta}=$$ precision). When $$\beta>1$$, it gives more weights to recall ($$\beta \to \propto$$: $$F_{\beta}=$$ recall). When $$ \beta = 1$$, it becomes harmonic mean of precision and recall.


* **1.2.2 ROC/AUC (Area Under ROC Curve)**
The ROC curve is the plot between true positive rate and false positive rate, where

$$True\ Positive\ Rate = \frac{TP}{all\ actual\ positives}$$

$$False\ Positive\ Rate = \frac{FP}{all\ actual\ negatives}$$

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/ROC.png)

The ROC curve is shown as above. The blue line shows how TPR changes according to different FTP. The TPR increases fast at first and slows down as FPR increases. The diagonal line shows result for random model. We should choose the model with TPR and FPR at 'turning point' at left upper coner. Moreover, the area under the ROC curve is calculated as AUC. The larger the AUC, the more accuracy of the model.

* **1.2.3 Gini Coefficient**

It is the ratio between the ROC curve and the disgnal line and the area of the above triangle. It can be calculated as 

$$Gini = 2AUC -1$$

Commonly, a gini coeafficient with value more than 60% denotes a good model. 


* **1.2.4 Concordant-Disconcordant Ratio**

It identifies the ability of the model to differentiate between event happening and not happening. Let's take a 2 class logistic regression classification as an example. The predicted probabilities for event(1) and event(0) are 

$$P_{i}, i=1,2,...,5\ for\ event(1)$$

$$P_{j}, j=1,2,...,5\ for\ event(0)$$

We can get 25 pairs of (1,0) with $$(P_{i},P_{j})$$,
* concordant pairs are pairs with $$P_{i} > P_{j}$$.
* disconcordant pairs are pairs with $$P_{i} < P_{j}$$.
* tied pairs are pairs with $$P_{i} = P_{j}$$.

$$concordant\ ratio =\frac{No.\ of\ concordant\ pairs}{No.\ of\ total\ pairs}$$

The larger the concordant ratio, the better the model.



* **1.2.5 Lift and Gain Chart**

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/LifeAndGain.png)

The above cumulative gains and lift charts can be used for measuring model performance. 
* The y-axis shows the percentage of positive responses. 
* The x-axis shows the percentage of data points contacted.
* The red line is **baseline**: If we contact X% of data points then we will receive X% of the total positive responses.
* The blue line is **lift curve**: using the predictions of the response model, calculate the percentage of positive responses for the percent of data points contacted and map these points to create the lift curve.

So, the greater the area between the life curve and the baseline, the better the model.


## **2. Unsupervised Learning (Clustering)**

The purpose pf clustering validity can be
* evaluating the goodness of clustering algorithm.
* avoiding finding patterns in noise.
* comparison of two clustering algorithm.
* comparison of two sets of clusters.
* finding the optimal No. of clusters.

And, there are various measures of cluster validity.

* **External Index**: measure the extent to which cluster labels match externally supplied class labels.
* **Internel Index**: measure the goodness of a clustering without externel information.
* **Relative Index**: compare two different clusterings or clusters.

### **2.1 Internal Validity**
Internal validity measures the compactness, connectedness and the separation of the clusters seperation.

* **compactness (cluster cohesion)**: It measrues how close are the objects within the same cluster. A lower within cluster variation means a good compactness. Examples can be within cluster sum of squares, within cluaster average/ median distances and so on.

* **Seperation**: it measures how well-seperated a cluster is from other clusters. it includes distance between cluster centers and pairwise minumum distances between objects in different clusters. e.g, between cluster sum of squares.

* **Connectivity**: it measures the extent to which items are placed in the same cluster as their nearest neighbors in the data set. The connectivity ranges from 0 to $$\propto$$ and should be minimized.

#### **2.1.1 total within-cluster sum of square(WSS) and Elbow method**

$$wss=\sum_{i}\sum_{x \in C_{i}} (x-\bar{x_{i}})^2$$

The total WSS (intra-cluster variation) measures the compactness of the clustering and we want to minimize the value. The Elbow method looks at the total WSS as a function of the No. of clusters. As shown in below graph, as the No. of clusters increase, the total WSS will always decrease. The optimual number of clusters is at 'elbow point', where by adding more clusters no longer drops the total WSS signifcantly.

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/ElbowMethod.png)

#### **2.1.2 Gap statistic**
The gap statistics compares the total WSS with their expected values under null reference distribution of the data. The opitimal culatering structure (e.g, the optimal number of clusters) will be the one maximize the gap statistic. The detailed method goes as follows.

* **Step 1**: clustering the data with various number of clusters from k= 1,2,..,m, and compute correspondin $$total\ WSS_{k}$$.

* **Step 2**: generate N data sets with a random uniform distribution. For each data set, culstering the data set with vairous number of clusters form k=1,2,..,m, and compute correspondin $$total\ WSS_{kn}$$.

* **Step 3**: calculate the gap statistic for varying k as 

$$GAP(k)=\frac{1}{N}\sum_{n=1}^N log(W_{kb})-log(W_{k})$$

and also calculate the stadard deviation of the statistics as $$s_{k}$$.

* **Step 4**: the optimum number of clusters is the smallest k such that:

$$Gap(k) \geq Gap(k+1)- s_{k+1}$$





#### **2.1.3 Silhouette Coefficient**

The Silhouette measures how similar an object is to its own cluster (cohesion) compared to other clusters (seperation). It ranges from -1 to 1. A high value shows the point is well matched to its own clusters and poorly matched to neighbouring clusters. For a given data set, if many data points have high values, then the clustering partition is appropriate.

The Silhouette Coefficient $$s(i)$$ for each data point $$i$$ is calculated below ways. Let

$$a(i)=\frac{1}{|C_{i}|-1}\sum_{j\in C_{n},i \neq j} d(i,j)$$

where $$C_{i}$$ is the Cluster data $$i$$ belongs to. $$d(i,j)$$ is the distance (e.g, Euclidean distance or Manhattan distance) between data points $$i$$ and $$j$$ in the cluster $$C_{i}$$. $$a(i)$$ is the average distance between $$i$$ and all other data points in the same cluster, thus measure how well $$i$$ is assigned to its cluster.

Let

$$b(i)=\min_{i \neq j}\frac{1}{|C_{j}|}\sum_{j\in C_{j}} d(i,j)$$

be the smallest average distance between $$i$$ and all points in any other clusters that $$i$$ doesn't belong to.
Then the Silhouette of data point $$i$$ is defined as

$$s(i)=\frac{b(i)-a(i)}{\max {(a(i),b(i))}}, if |C_{i}|>1$$

and 

$$s(i)=0, if\ |C_{i}|=1$$

* large $$s(i)$$ (close to 1): the point $$i$$ is very well clustered.
* $$s(i)=0$$: point $$i$$ lies between two clusters.
* $$s(i)<0$$: point $$i$$ is placed in wrong clusters.

Then average $$s(i)$$ over all points to measure how well the data has been clustered. 

#### **2.1.4 Dunn Index**
 It tries to identify sets of clusters that are compact (small variance between members of the cluster) and well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster variance. The Dunn Index is calculted as below way.

 $$DI = \frac{min.Seperation}{max.Diameter} =\frac{\min_{x \in C_{i},y \in C_{j}, i \neq j} d(x,y)}{\max_{x,y \in C_{i}} d(x,y)}$$

 The Dunn Index should be maximized.




### **2.2 External Validity**

#### **Rand Index**
The **Rand Index** or **Rand measure** measures the similarity between two data clustering. It defines in the way below. Given a set of $$n$$ elements $$S=\{ o_{1},...,o_{n}\}$$, and two clusterings of $$s$$, $$X=\{ X_{1},...,X_{r}\}$$, a partition of $$S$$ into $$r$$ clusters, and $$Y=\{Y_{1},...,Y_{k}\}$$, a partition of $$S$$ into $$k$$ clusters, then define the following:

* $$a$$, the number of pairs of elements in $$S$$ that are in the same cluster in $$X$$ and in the same clusters in $$Y$$
* $$b$$, the number of pairs of elements in $$S$$ that are in the different clusters in $$X$$ and in the different clusters in $$Y$$

The Rand Index,$$R$$, is

$$R=\frac{a+b}{\pmatrix{n \\ 2}}$$

It ranges from 0 to 1.
* $$R=0$$: two clustering do not agree on any pairs.
* $$R=1$$: two clusterings are exactly same.

The **Adjusted Rand Index** is the corrected-for-chance version of **Rand Index**. It establishes a baseline by random model. It varies from -1 (no agreement) to 1 (perfect agreement).














