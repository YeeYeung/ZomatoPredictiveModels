# Zomato Sydney Restaurant Predictive Models

Predictive modelling project analysing 10,500+ Sydney restaurants from Zomato. Covers exploratory data analysis, geospatial visualisation, and machine learning models for rating prediction and restaurant classification.

## Overview

| | Part A: EDA & Visualisation | Part B: Predictive Modelling |
|---|---|---|
| **Focus** | Data cleaning, exploration, feature engineering | Regression & classification models |
| **Key tools** | Pandas, Plotly, GeoPandas, Seaborn | Scikit-learn (Linear Regression, Logistic Regression, SVM, Decision Tree, KNN) |
| **Outputs** | Interactive maps, distribution plots, Tableau dashboard | Model comparison with accuracy metrics |

### Key Results

- **Rating prediction** (Linear Regression): MSE = 0.167
- **Restaurant classification** (Logistic Regression): 84% accuracy
- **134 unique cuisines** identified across Sydney suburbs
- **Top suburbs**: CBD (476 restaurants), Surry Hills (260), Parramatta (225)

## Project Structure

```
.
├── Part_A_Jupyter.ipynb          # EDA, data cleaning, geospatial analysis
├── Part_B_Jupyter.ipynb          # Predictive modelling (regression + classification)
├── part_a_analysis.py            # Part A as standalone Python script
├── part_b_analysis.py            # Part B as standalone Python script
├── zomato_df_final_data.csv      # Raw dataset
├── zomato_cleaned_data_for_tableau.csv  # Cleaned data for Tableau
├── sydney.geojson                # GeoJSON for suburb boundaries
├── Zomato_Sydney_Data_Analysis.twb      # Tableau workbook
├── Dockerfile                    # Docker configuration
└── requirements.txt              # Python dependencies
```

## Quick Start

### Option 1: Local Setup

```bash
git clone https://github.com/YeeYeung/ZomatoPredictiveModels.git
cd ZomatoPredictiveModels
pip install -r requirements.txt
jupyter lab
```

### Option 2: Docker

```bash
docker pull yeeyeung/zomato_predictive_model
docker run -it -p 8888:8888 yeeyeung/zomato_predictive_model
```

## Tech Stack

Python 3.12 | Pandas | NumPy | Scikit-learn | Plotly | Matplotlib | Seaborn | GeoPandas | Docker | Tableau

## License

[MIT](LICENSE)
