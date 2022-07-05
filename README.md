# Empirical study of the value of weak supervision 

This code allows the reader to replicate all the experiments carried out in our study, and obtain the figures.

- `figA` relates the size of the candidate sets with the proportion of weakly labeled examples.
- `figB` relates the number of fully labeled samples with the proportion of weakly labeled ones.
- `figC` relates the probability of co-occurrence with other parameters (size of candidate sets, proportion of weakly labeled samples).

- `onlyfull` refers to baseline experiments where only the fully labeled subsets is used to learn the model.
- `real_model` refers to experiments that use the real generative model to establish the maximum reachable.
  - With real data, this real model is simulated.
- All the models and datasets are generated once using `gen_models_and_data` scripts.

The two notebooks collect experimental results and make the plots for synthetic and real data.