# Significance Test

## Accuracy Comparison
To calculate whether one setup is significantly better than the other in terms of accuracy for all non-lexical datasets except speakbench (pronunciation, speed, speaker, tmhintq, thaimos, somos), run:

```bash
python bootstrap_nonlexical.py
```

## Positional Bias Significance
To calculate whether the positional bias of one setting is statistically significant when judging chatbotarena:

```bash
python chatbotarena_positonal_bias.py
```

To calculate whether the positional bias of one setting is statistically significant when judging non-lexical datasets:

```bash
python nonlexical_positonal_bias.py
```

## Verbosity Bias Significance
To calculate whether the verbosity bias of one setting is statistically significant:

```bash
python chatbotarena_verbosity_bias.py
```