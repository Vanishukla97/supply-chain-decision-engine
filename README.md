\# Supply Chain Decision Intelligence \& Replay Engine



A data-driven decision-support system that predicts late-delivery risk, prioritizes operational alerts, and compares supply-chain decision strategies using historical data.



\## Business Problem



Late deliveries can affect customer satisfaction and operational planning. Supply-chain teams need a way to identify potentially late orders, prioritize investigations, and understand the trade-off between missed delays and unnecessary alerts.



This project explores how machine learning, SQL, and cost-sensitive evaluation can support those decisions.



\## Key Features



\- \*\*Late-delivery risk prediction:\*\* Estimates the probability that an order will be delivered late.

\- \*\*Alert prioritization:\*\* Categorizes orders into High, Medium, Low, and No Alert groups.

\- \*\*Shipping-mode analysis:\*\* Compares historical late-delivery patterns across shipping modes.

\- \*\*Model evaluation:\*\* Reports accuracy, precision, recall, F1-score, and confusion-matrix results.

\- \*\*Baseline comparison:\*\* Evaluates the full model against a shipping-mode-only baseline.

\- \*\*Cost scenario analysis:\*\* Compares strategies under different illustrative false-alarm and missed-delivery penalties.

\- \*\*Interactive dashboard:\*\* Presents model performance, risk monitoring, shipping-mode insights, and decision-cost comparisons.



\## Technology Stack



\- Python

\- pandas

\- scikit-learn

\- PostgreSQL

\- SQL

\- Streamlit

\- Altair



\## Dataset and Methodology



The project uses the DataCo Smart Supply Chain dataset.



The modeling dataset contains 62,897 non-cancelled orders. The target variable is `is\_late`.



To reduce target leakage, outcome-related fields such as delivery status, actual delivery duration, and `Late\_delivery\_risk` are excluded from prediction features.



A chronological evaluation approach is used:



\- Training period: Orders before October 2017

\- Validation period: October–December 2017

\- Historical evaluation period: January 2018



The January 2018 evaluation period was inspected during development, so it should not be considered a completely untouched holdout.



\## Model Performance



\*\*Historical evaluation results — January 2018\*\*



| Metric | Result |

|---|---:|

| Orders evaluated | 2,037 |

| Accuracy | 61.32% |

| Precision | 62.21% |

| Recall | 86.78% |

| F1-score | 72.47% |

| Decision threshold | 0.40 |



The selected threshold prioritizes recall, helping identify a larger proportion of late orders at the cost of additional false alarms.



\### Comparison with the Shipping-Mode Baseline



| Metric | Full Model | Baseline |

|---|---:|---:|

| Accuracy | 61.32% | 69.32% |

| Precision | 62.21% | 83.22% |

| Recall | 86.78% | 59.75% |

| F1-score | 72.47% | 69.56% |



The full model identifies more late orders, while the baseline generates fewer false alarms. The preferred strategy depends on operational priorities and investigation costs.



\## Cost-Sensitive Decision Analysis



The project compares both strategies using three illustrative cost scenarios:



| Scenario | Full Model Cost | Baseline Cost | Lower-Cost Strategy |

|---|---:|---:|---|

| Balanced | 1,104 | 1,587 | Full Model |

| Service-focused | 1,420 | 2,549 | Full Model |

| Investigation-cost-focused | 2,364 | 1,875 | Baseline |



These values are illustrative cost units based on assumed penalties. They do not represent measured financial savings.



\## Project Structure



```text

supply-chain-decision-engine/

├── app.py

├── config/

├── data/

│   └── processed/

├── docs/

├── models/

├── sql/

├── src/

├── .gitignore

└── README.md

```



\## Running the Dashboard



1\. Clone the repository.

2\. Create and activate a Python virtual environment.

3\. Install the dependencies from `requirements.txt` (if included in the repository).

4\. Ensure the processed files and trained model required by the application are available.

5\. Start the dashboard:



```bash

streamlit run app.py

```



The dashboard uses historical evaluation outputs; it is not a live supply-chain monitoring service.



\## Limitations



\- Evaluation results reflect the selected dataset and historical period.

\- The January 2018 evaluation period was inspected during development.

\- Lower-probability calibration bands show some mismatch between predicted probabilities and actual outcomes.

\- Shipping-mode patterns are dataset-specific and should not be treated as universal operational rules.

\- Cost comparisons depend on assumed penalties rather than actual business costs.



\## Future Improvements



\- Investigate calibration and alert-category ordering.

\- Evaluate the model on a genuinely untouched later period.

\- Incorporate actual operational cost estimates when available.

\- Add reproducible data preparation and model-training instructions.

\- Improve automated validation and testing.



\## Author



Built as a portfolio project focused on supply-chain analytics, machine learning, and operational decision support.

