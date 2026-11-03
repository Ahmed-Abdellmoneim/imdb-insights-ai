# Community reviews

Reviews written in the app and moderated by hand. The model's guess is made from the text alone, after the writer chose a rating, and is only shown to them afterwards.

1 submissions: 1 approved, 0 rejected, 0 pending.

## Rating model on first-party reviews

Step 3's embedding model (trained on Stanford IMDb reviews) against the rating each writer gave. 95% bootstrap intervals in brackets.

| model version | reviews | MAE ↓ | within ±1 ↑ | polarity acc. ↑ | Spearman ↑ | bias |
|---|---|---|---|---|---|---|
| `90cfef8f7a` | 1 | 2.61 [2.61–2.61] | 0% | 100% | – | -2.61 |

Ratings given: 10★ 1
