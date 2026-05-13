# Neural Traces after Attrition

Previous studies have shown that international adoptees can retain for many years neural traces of the language spoken in their environment before the adoption occurred, even in cases where the adoption was as early as 12-18 months of age. This repo develops evidence that deep learning systems also retain "neural" traces of their early training environment by switching language training environment and then evaluating model similarity with Representational Similarity Analysis (RSA).

The experiments are written using the [SpeechBrain](http://github.com/speechbrain/speechbrain) Toolkit, to run an experiment you can execute the following:

```bash
python speechbrain_main.py experiments/pretrain_fr.yaml
```

The figures demonstrating Neural Traces can be seen in `neural_traces.ipynb`. 
