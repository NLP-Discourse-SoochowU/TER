# TER
Empowering Tree-structured Entailment Reasoning: Rhetorical Perception and LLM-driven Interpretability.


# Environment
Inherited from MetGen


# Data
[EntailmentBank dataset](https://allenai.org/data/entailmentbank).
Binarized EntailmentBank dataset can be found in `data/entailment_trees_binary`

```

# Fact retrieval and RST relation prediction

Follow the `./scripts/tune_controller_task1.sh` to train the model for task 1.
Follow the `./scripts/tune_controller_task2.sh` to train the model for task 2 and 3.

# Model selection and testing
Follow the `./scripts/test_*.sh` scripts to estimate the system on DEV and then test the selected models based on TEST; 
Follow the `eval_*.sh` scripts to evaluate the quality of the system outputs.

Contact [Longyin Zhang](zhangly@i2r.a-star.edu.sg) for more info.
