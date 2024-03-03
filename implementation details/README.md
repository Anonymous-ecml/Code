# Implementation Details
For training the model, we used Adam with the mini-batch of 256 patients and the learning rate is set to 1e−3. To fairly compare different approaches, the hyper-parameters of the baseline models are fine-tuned by grid-searching strategy. Detailed experimental settings for different datasets refer to [](). The analysis of hyperparameters refer to [MIMIC-III_Hyperparameters](MIMIC-III_Hyperparameters.pdf) and [eICU Hyperparameters](eICU_Hyperparameters.pdf).

We include several state-of-the-art models as our baseline approaches. For SimCLR, we augment the data by two simple approaches: random time shift and reversion (same as ConCAD). For GraphCL and GRACE, we use the same graph structure learning as in our model to construct the graph.
