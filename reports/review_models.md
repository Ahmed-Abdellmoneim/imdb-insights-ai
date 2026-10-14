# Review rating models

## Stanford validation (movie-grouped slice of train)

2,679 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 3.328 [3.286–3.375] | 3.523 | – | 9.6% | 50.8% [48.9%–52.7%] | +0.10 |
| embeddings | 1.269 [1.225–1.316] | 1.723 | 0.830 | 68.2% | 92.2% [91.2%–93.2%] | +0.02 |
| finetune | 1.052 [1.007–1.098] | 1.606 | 0.865 | 80.1% | 92.4% [91.4%–93.4%] | +0.10 |
| tfidf | 1.595 [1.547–1.639] | 2.050 | 0.802 | 54.1% | 89.0% [87.8%–90.2%] | -0.03 |

## Stanford test, official (25,000 reviews)

25,000 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 3.290 [3.276–3.304] | 3.491 | – | 10.5% | 50.0% [49.4%–50.6%] | -0.02 |
| embeddings | 1.251 [1.237–1.265] | 1.684 | 0.830 | 68.7% | 92.9% [92.6%–93.2%] | -0.01 |
| finetune | 1.031 [1.017–1.045] | 1.549 | 0.863 | 80.6% | 93.6% [93.3%–93.9%] | +0.05 |
| tfidf | 1.604 [1.588–1.620] | 2.048 | 0.795 | 53.9% | 89.4% [89.0%–89.8%] | -0.08 |

## Stanford test without leaked reviews

24,873 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 3.290 [3.274–3.305] | 3.491 | – | 10.5% | 49.9% [49.2%–50.4%] | -0.03 |
| embeddings | 1.251 [1.238–1.265] | 1.684 | 0.830 | 68.6% | 92.9% [92.6%–93.2%] | -0.01 |
| finetune | 1.032 [1.017–1.048] | 1.550 | 0.863 | 80.6% | 93.6% [93.3%–93.9%] | +0.05 |
| tfidf | 1.609 [1.594–1.624] | 2.052 | 0.794 | 53.7% | 89.4% [89.0%–89.7%] | -0.08 |

## Stanford test (clean) where the review does not state its x/10 score

22,742 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 3.316 [3.300–3.331] | 3.514 | – | 10.2% | 50.1% [49.4%–50.7%] | -0.02 |
| embeddings | 1.261 [1.247–1.275] | 1.697 | 0.829 | 68.2% | 92.8% [92.5%–93.1%] | -0.01 |
| finetune | 1.041 [1.026–1.056] | 1.563 | 0.862 | 80.3% | 93.5% [93.1%–93.8%] | +0.06 |
| tfidf | 1.629 [1.612–1.645] | 2.072 | 0.793 | 53.2% | 89.4% [89.0%–89.7%] | -0.08 |

## TMDB reviews (out of domain, includes 5-6 ratings)

9,850 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 2.264 [2.238–2.288] | 2.599 | – | 19.0% | 8.3% [7.8%–9.0%] | -1.83 |
| embeddings | 1.231 [1.211–1.251] | 1.578 | 0.577 | 67.7% | 94.3% [93.8%–94.8%] | -0.00 |
| finetune | 1.135 [1.115–1.154] | 1.506 | 0.690 | 73.7% | 95.5% [95.1%–95.9%] | +0.22 |
| tfidf | 1.329 [1.308–1.352] | 1.714 | 0.500 | 64.6% | 91.2% [90.5%–91.8%] | -0.32 |

## Head-to-head sample: 1,000 clean Stanford test reviews (all models)

999 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 3.292 [3.221–3.358] | 3.493 | – | 10.5% | 49.8% [46.6%–53.3%] | -0.03 |
| embeddings | 1.314 [1.238–1.389] | 1.766 | 0.819 | 66.5% | 91.8% [90.2%–93.5%] | +0.05 |
| finetune | 1.101 [1.029–1.180] | 1.651 | 0.852 | 78.6% | 92.2% [90.6%–93.7%] | +0.07 |
| llm-calibrated | 1.847 [1.753–1.940] | 2.390 | 0.714 | 52.4% | 86.1% [84.1%–88.3%] | -0.02 |
| llm-zero-shot | 2.072 [1.946–2.199] | 2.829 | 0.714 | 51.3% | 75.0% [72.2%–77.7%] | +1.42 |
| tfidf | 1.636 [1.556–1.717] | 2.104 | 0.779 | 53.2% | 88.4% [86.5%–90.3%] | -0.01 |

## Head-to-head sample: 500 TMDB reviews (all models)

500 reviews. 95% bootstrap intervals in brackets.

| model | MAE ↓ | RMSE ↓ | Spearman ↑ | within ±1 ↑ | polarity acc. ↑ | bias |
|---|---|---|---|---|---|---|
| mean baseline | 2.265 [2.153–2.378] | 2.601 | – | 19.0% | 8.4% [5.8%–11.2%] | -1.82 |
| embeddings | 1.254 [1.168–1.341] | 1.602 | 0.595 | 67.4% | 92.6% [90.1%–95.0%] | -0.02 |
| finetune | 1.176 [1.093–1.267] | 1.548 | 0.683 | 72.8% | 94.0% [91.5%–96.1%] | +0.19 |
| llm-calibrated | 1.837 [1.705–1.972] | 2.369 | 0.607 | 51.2% | 79.9% [75.7%–83.5%] | -1.42 |
| llm-zero-shot | 1.283 [1.187–1.377] | 1.686 | 0.607 | 63.6% | 88.3% [85.0%–91.2%] | -0.02 |
| tfidf | 1.271 [1.176–1.371] | 1.693 | 0.553 | 67.4% | 88.3% [85.4%–91.0%] | -0.41 |
