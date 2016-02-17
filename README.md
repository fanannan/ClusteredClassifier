ClusteredClassifier
---

This is an experimental implementation based on my idea for an efficient and convinient classifier.
- Some of the clustering methodologies do not always make the same results. In other words, such clustering methodologies may suggest the different set of clusters at each execution.
- Some of the classification methodologies are useful at noisy high dimensionary data but may not converge well enough.
- The proposed approach firstly makes some clusters and then applies classification on each cluster. After recording the clustering model and the classification models, repeat the process many times.
- The classifier may find the key features which may not be used well at the clustering process. A cluster may likely have unbalanced samples so that the classifier must have capability to handle unbalanced data set. 
- In addition, when investigating the clusters where the classification models do not perform well at, we may understand the characteristics of the clusters and find the missing features to imporve the models. Or, we may find certain set of conditions where we should not adopt the classification methodology employed.
- This approach does not suit for classification of low dimensionary data.

