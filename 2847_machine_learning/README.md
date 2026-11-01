# H&M Personalized Fashion Recommendations

What might a customer buy next? This project uses past purchases, product
information and clothing images to recommend **twelve articles for each
customer for the following week**. It began as a university group project for
the H&M recommendation challenge on Kaggle.

The main idea is to combine several clues: what the customer bought before,
what other customers bought, what is popular now, and which products fit the
customer's preferences. A machine learning model then learns how to combine
these clues into a ranked list.

The published pipeline focuses on one selected model:
**50k/expanded + BPR + repeat/activity** (`bpr_repeat_50k_expanded`). It retrieves
articles from recent popularity, personal history, and similar purchase
patterns, then learns their order with gradient-boosted classifiers. BPR learns
customer–article compatibility; repeat and activity features describe purchase
frequency and timing. Two classifiers serve customers with recent versus older
purchase histories.

## How the recommender works

### 1. Find promising articles

Instead of scoring the entire catalogue for every customer, the system first
builds a shortlist from three sources:

- **Popular articles:** products bought by many customers in the previous week.
- **Purchase history:** articles the customer has bought before, since repeat
  purchases are allowed and common in this task.
- **Collaborative filtering:** articles bought by similar groups of customers.
  Products are considered similar when their buyers overlap.

### 2. Describe each customer–article pair

For each shortlisted article, the system builds features describing why it
might be relevant:

- **Purchase timing and activity:** how recently and frequently the customer
  bought the article, their usual purchase intervals, and recent shopping activity.
- **Learned preferences:** Bayesian Personalized Ranking (**BPR**) learns a
  numerical representation of customers and articles from purchase interactions,
  producing a customer–article compatibility score.
- **Product preferences:** how well the article matches the customer's history
  of product types, colours and patterns.
- **Visual preferences:** a pretrained image model represents clothing images
  as numerical vectors, which help describe the customer's visual preferences.
  The image model is kept fixed rather than trained on this dataset.
- **Seasonality:** calendar patterns in demand for different product groups.

### 3. Learn the final ranking

**Gradient-boosted decision trees** learn from these features to estimate which
candidate articles will be bought next. The twelve highest-scoring distinct
articles become the recommendations.

The system uses two classifiers: one for customers who bought within the last
seven days, also used for customers with no purchase history, and another for
customers with older histories. The second classifier also uses seasonal features.

At each prediction date, both classifiers are trained again using the two most
recent completed weeks. Each training example uses only information available
before the week it tries to predict. The selected model samples **50,000 buyers
per training week**, keeping all shortlisted articles for those buyers.

## Results

The latest submission received the following **Kaggle scores**:

| Leaderboard | MAP@12 |
|---|---:|
| Public | **0.03107** |
| Private | **0.03135** |

Locally, the following comparisons helped build the final model:

| Model | September validation MAP@12 |
|---|---:|
| Profile model, 10k buyers per training week | 0.030495 |
| Profile model, 25k buyers | 0.031148 |
| 25k + BPR | 0.033723 |
| 25k + BPR + repeat/activity | 0.034144 |
| **50k + larger boosted trees + BPR + repeat/activity** | **0.034450** |

These validation scores average the weeks **2–8 and 9–15 September 2020**.
The selected model also averaged **0.028251** across the July/August backtests.
All local validation weeks were inspected during development, so these are
exploratory comparisons.

**MAP@12** measures whether the articles a customer actually buys appear near
the top of the twelve recommendations. Evaluation averages this score over
next-week buyers, including those with no earlier purchase history.

## What we learned

Learning the order of a shortlist was more effective than relying only on fixed
rules for combining recommendations. BPR supplied a useful preference signal,
while repeat-purchase timing and customer activity improved the ranking further.
More training data helped most when paired with larger boosted trees; increasing
the sample size alone did not consistently improve results.

For model settings and running instructions, see the
[pipeline guide](pipeline/pipeline.md). The
[development history](docs/development_history.md) describes the experiments in
more detail.

## Authors and reuse

Group 12:

- Ana Catarina Velez Mendes
- Federico Maria Brignoli
- Isabella Russo
- Mariana Alexandra Feritas Filipe
- Miguel Parra-Castro

Dataset source: [H&M Personalized Fashion Recommendations on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
