The Adam optimizer with a learning rate of 1e-3 and a batch size of 256 (patients) is employed to train the proposed approach. 
The hyperparameters of baseline approaches are optimized using a grid-searching strategy. 
The data augmentation techniques (i.e., random time shift and reversion) used by SimCLR are similar to those used by ConCAD. 
The outputs of patient graph structure learning in the proposed network architecture are used as input for GraphCL and GRACE. 
The objective function of SimCLR and GRACE in the self-supervised learning setting is contrastive loss. When properly trained and optimized, the obtained patient representations are fed into logistic regressors for downstream prediction tasks.
