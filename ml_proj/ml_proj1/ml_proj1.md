1.1 In raw dataset, all the class labeled as  *No stroke* are arranged together, completely seperated from the class label as *Having stroke*, which means data is sorted by their classes. We should shuffle it to make sure our training/testing/validation sets are representative of the overall distribution of the data. 

When using Naive Bayes Classifier, it makes strong assumption that any two features are idependent. Then, using chain rule, it could just easily multiply the conditional distribution over the given class of all features

And we could see that in data set, it is not gurantee that any two features are idependent. In fact, some of them are high depedent on others, such as *Age vs. Hypertension*, *Age vs. BMI*, etc. It is the main issue with the data. Using Naive Bayes classifier, it makes strong assumption that any two features are idependent. When making an interference on a new sample with missing feature, we could easily drop the missing feature uniformly for all classes. And it could work because of chain rule. Finally, pick the class that could maximize the posterior probability.
$$
\begin{aligned}
P\left(x_{1}, x_{2}, \ldots, x_{M} \mid y\right) P(y) & \approx P\left(x_{1} \mid y\right) P\left(x_{2} \mid y\right) \ldots P\left(x_{M} \mid y\right) P(y) \\
&=P(y) \prod_{m=1}^{M} P\left(x_{m} \mid y\right)
\end{aligned}
$$

$$
\begin{aligned}
\hat{y}=\underset{y \in Y}{\operatorname{argmax}} P\left(x_{1}, x_{2}, \ldots, x_{M} \mid y\right) P(y)
\end{aligned}
$$



After having amn observation of the given dataset, we found that the main issue with the data is some features are highly dependent on others.

 It is the main issue with the data. Using Naive Bayes classifier, it makes strong assumption that any two features are idependent. When making an interference on a new sample with missing feature, we could easily drop the missing feature uniformly for all classes. And it could work because of chain rule. 







1.2 I don't think accuracy is an appropriate metric to evaluate the models created for this data. Let's label *Having stroke* as **Positive** while lable *No stroke* as **Negetive**. We know that $Accuracy=\frac{TP + TN}{TP + FP + FN + TN}$. A model could only predict **Negetive** may also achieve high accuracy, because in the real world, people having stroke are the minority, leading *FN* becomes far way smaller than *TN*. However, the cost of *False Negetive* (i.e., telling a patient who might suffer from stroke he/she is totally fine) is really high, the symptoms of stroke are usually permanent.  We need to reduce the probibility of *FN*; therefore, I would suggest that using *Recall*, where $Recall = \frac{TP}{TP + FN}$,  as the most appropriate metric in evaluation.  Also, *F1-score* is also a good method to select a best model when we have tie on using *Recall* to compare models. Because high *Precision*, where $Precision=\frac{TP}{TP+FP}$, also has important  medical significance that it could reduce the anxiety of those who are doing this test.



2.1 

Adv: 

- Super easy to implement.

DisAdv:

- It makes a very strong assumption that any two features are independent, but apparently , features in our data are far away from that assumption. Due to this, the result can be (potentially) very bad.
- The performance of pure Naive Bayes may be highly affected by data scarcity. Basically , we count the frequency of each feature and calculate the probability given by classes to estimate a maximum likelihood value. Due to chain rules, total conditional probability will become 0 when conditional probability of some features are 0. We should modify it by smoothing our data.

2.2

I choose Categorical Naive Bayes (CNB) because it is easier to implement. When using Hold-out to split data, I found that when the ratio between training data and testing data is 7:3, the gap between the recall of traning data and the recall of test data is smallest. And the recall of traning data is always lager than the recall of test data. What's more, the accuracy of traning data is also larger than the accuracy of test data, but they are quite close. 

And it seems like the amount of test data doesn't have a significant influcent on its accuracy which is around 80.

CNB and Zero-R has similar accuracy. But Zero-R cannot evaluted by *Recall* and *F1* of the class *Having stroke*, because it always predicts that patients won't have a stroke. By contrast, we could evaluate CNB with *Recall* and *F1* appropriately. And because our goal is to make efforts to find out potential patients , we care more about *Recall* and *F1* than accuracy. 

2.3

In *epsilon smoothing*, we replace the probality of features with probability 0 with $\epsilon$;

In *Laplace smoothing*, we add "pseudocount" $\alpha$ to each feature count observed during training, and usually set $\alpha$ to 1



3.1 K = 7

3.2 

Precision , recall and F1of class *No stroke* are overwhelmingly larger than those of class *Having stroke*, respectively. I think it is becasue in KNN, parameter *weight* is set to *uniform* by defualt, and 

The same as comparison between CNB and Zero-R, KNN could make prediction on *Having stroke*, although the performance is poor in this case.

3.3

Both NB and KNN achieve relative high accuracy, around 0.8. NB shows higher Recall than KNN for class *Having stroke* but shows lower recall than KNN for class *No stroke*. As for other evaluation metrics, NB and KNN show similar outcomes. In this dataset, the numer of  *No stroke* is 3 times as many as the number of *Having stroke*. The imbalance data distribution may significantly affect the performance of KNN because KNN does not have any prior knowledge. 

 It suffers from skewed class distributions meaning if a specific class occurs frequently in the training set then it is most likely to dominate the majority voting of the new example. The accuracy of KNN deteriorates with high-dimension data as there is hardly any difference between the nearest and farthest neighbor.



So, KNN prones to specify new example as *No stroke*, leading to low recall.



Precision , recall and F1 of class *No stroke* are overwhelmingly larger than those of class *Having stroke*, respectively. I think it is because the number of instance labeled as *No stroke* is four times as larger as the number of instance labeled as *Having stroke*. And because of the nature of K-NN, when making predictions on a new instance, the classifier will prone to classify it as the majority.  In this point, the result of K-NN is similar with the outcome of Zero-R, because both of them are significantly affected by the size of each class.









**Advantages**:

- *Simplicity and Efficiency* Having independence assumption makes Naive Bayes much easier to understand and to be implemented. And independence assumption also significantly reduce the number of conditional distribution, speeding up the whole learning and precition process. In this case, some features are dependent on others. If we use a model fully based on actual condictional probability distribution rather than assuming that features are mutually independent, the whole process would be far more complicated and time-consuming, and the model may also have high variance.

- *Easily dealing with missing/extra features* When making an interference on a new instance with missing features or with extra features. With independence assumption and chain rule, we could drop the missing feature uniformly for all classes or drop the extra feature for the new instance without dropping any instance. For example, say we take a mountain of data from data warehouse to predict the probability of those people having stroke, it may probably that some instances miss some features or have extra features, i.e., the data we get is very likely unorganized. In this way, data is much easier to be preproceeded.

**Disadvantages**:

- Naive Bayes classifier strongly assumes that that all features in $X$ are mutually independent. Once tha shape of our data distribution violates this assumption, the performace of the model could be potentially bad. Unfortunately, our data is far away from the assumption.