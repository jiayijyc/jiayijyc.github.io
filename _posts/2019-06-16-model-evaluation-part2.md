---
layout: post
title:  "Model Evaluation - Part 2"
---


# **Introduction**

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/DataScience1.jpg)

For supervised regression and classification, we already have a variety of measures to evaluate how good the model is. But, how do we measure the "goodness-of-fit" for cluster analysis. And, why do we need to do the cluster validity? The purpose pf clustering validity can be
* evaluating the goodness of clustering algorithm.
* avoiding finding patterns in noise.
* comparison of two clustering algorithm.
* comparison of two sets of clusters.
* finding the optimal No. of clusters.

And, there are various measures of cluster validity.

* **External Index**: measure the extent to which cluster labels match externally supplied class labels.
* **Internel Index**: measure the goodness of a clustering without externel information.
* **Relative Index**: compare two different clusterings or clusters.

As a result, this article will introduce various model evaluation metrics used for unsupervised learning. 

# **Model Evaluation Metrics - Unsupervised Learning**

## **1 Internal Validity**
Internal validity measures the compactness, connectedness and the separation of the clusters seperation.

* **compactness (cluster cohesion)**: It measrues how close are the objects within the same cluster. A lower within cluster variation means a good compactness. Examples can be within cluster sum of squares, within cluaster average/ median distances and so on.

* **Seperation**: it measures how well-seperated a cluster is from other clusters. it includes distance between cluster centers and pairwise minumum distances between objects in different clusters. e.g, between cluster sum of squares.

* **Connectivity**: it measures the extent to which items are placed in the same cluster as their nearest neighbors in the data set. The connectivity ranges from 0 to $$\propto$$ and should be minimized.

### **1.1 total within-cluster sum of square(WSS) and Elbow method**

$$wss=\sum_{i}\sum_{x \in C_{i}} (x-\bar{x_{i}})^2$$

The total WSS (intra-cluster variation) measures the compactness of the clustering and we want to minimize the value. The Elbow method looks at the total WSS as a function of the No. of clusters. As shown in below graph, as the No. of clusters increase, the total WSS will always decrease. The optimual number of clusters is at 'elbow point', where by adding more clusters no longer drops the total WSS signifcantly.

![](https://raw.githubusercontent.com/jiayijyc/jiayijyc.github.io/master/_assets/images/ElbowMethod.png)

### **1.2 Gap statistic**
The gap statistics compares the total WSS with their expected values under null reference distribution of the data. The opitimal culatering structure (e.g, the optimal number of clusters) will be the one maximize the gap statistic. The detailed method goes as follows.

* **Step 1**: clustering the data with various number of clusters from k= 1,2,..,m, and compute correspondin $$total\ WSS_{k}$$.

* **Step 2**: generate N data sets with a random uniform distribution. For each data set, culstering the data set with vairous number of clusters form k=1,2,..,m, and compute correspondin $$total\ WSS_{kn}$$.

* **Step 3**: calculate the gap statistic for varying k as 

$$GAP(k)=\frac{1}{N}\sum_{n=1}^N log(W_{kb})-log(W_{k})$$

and also calculate the stadard deviation of the statistics as $$s_{k}$$.

* **Step 4**: the optimum number of clusters is the smallest k such that:

$$Gap(k) \geq Gap(k+1)- s_{k+1}$$





### **1.3 Silhouette Coefficient**

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

### **1.4 Dunn Index**
 It tries to identify sets of clusters that are compact (small variance between members of the cluster) and well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster variance. The Dunn Index is calculted as below way.

 $$DI = \frac{min.Seperation}{max.Diameter} =\frac{\min_{x \in C_{i},y \in C_{j}, i \neq j} d(x,y)}{\max_{x,y \in C_{i}} d(x,y)}$$

 The Dunn Index should be maximized.




## **2 External Validity**

### **2.1 Rand Index**
The **Rand Index** or **Rand measure** measures the similarity between two data clustering. It defines in the way below. Given a set of $$n$$ elements $$S=\{ o_{1},...,o_{n}\}$$, and two clusterings of $$s$$, $$X=\{ X_{1},...,X_{r}\}$$, a partition of $$S$$ into $$r$$ clusters, and $$Y=\{Y_{1},...,Y_{k}\}$$, a partition of $$S$$ into $$k$$ clusters, then define the following:

* $$a$$, the number of pairs of elements in $$S$$ that are in the same cluster in $$X$$ and in the same clusters in $$Y$$
* $$b$$, the number of pairs of elements in $$S$$ that are in the different clusters in $$X$$ and in the different clusters in $$Y$$

The Rand Index,$$R$$, is

$$R=\frac{a+b}{\pmatrix{n \\ 2}}$$

It ranges from 0 to 1.
* $$R=0$$: two clustering do not agree on any pairs.
* $$R=1$$: two clusterings are exactly same.

The **Adjusted Rand Index** is the corrected-for-chance version of **Rand Index**. It establishes a baseline by random model. It varies from -1 (no agreement) to 1 (perfect agreement).














