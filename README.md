# Mutual-Fund-Returns-Prediction-Model
# Mutual Fund Returns Prediction Model

## Overview
This machine learning model predicts mutual fund returns using a Random Forest Regressor. The model analyzes various fund characteristics and performance metrics to forecast future returns with high accuracy, achieving an R² score of 0.8420.

## Model Performance
- **Mean Squared Error (MSE)**: 4.7628
  - Root MSE: 2.18% (square root of MSE)
  - This indicates that our predictions deviate by approximately 2.18 percentage points from actual returns on average
- **R² Score**: 0.8420 (84.20%)
  - The model explains 84.20% of the variance in mutual fund returns
  - This high R² score suggests strong predictive capabilities for financial market data

## Feature Importance Analysis
The model identifies the following features as key predictors (in order of importance):

1. **Risk-Adjusted Returns** (59.33%)
   - Most influential feature
   - Demonstrates the importance of risk-weighted performance metrics

2. **Beta** (20.16%)
   - Second most important feature
   - Measures fund's market sensitivity

3. **Alpha** (13.38%)
   - Indicates the significance of excess returns
   - Third most impactful feature

4. **Other Features** (Combined 7.13%)
   - Expense Ratio (2.24%)
   - Category Encoded (2.14%)
   - Fund Size (1.09%)
   - Sharpe Ratio (0.83%)
   - Sub-Category Encoded (0.69%)
   - Fund Age (0.14%)

## Key Insights
1. Risk-adjusted metrics dominate the prediction process, accounting for nearly 60% of the model's decisions
2. Market sensitivity (Beta) and excess returns (Alpha) together account for about 33% of the prediction weight
3. Operational characteristics (expense ratio, fund size, age) have relatively lower impact on predictions

## Model Implementation
```python
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

## Data Preprocessing Steps
1. Numeric conversion of key metrics (sortino, sharpe, alpha, beta, sd)
2. Median imputation for missing values
3. Feature engineering including:
   - Risk-adjusted returns calculation
   - Category encoding
   - Sub-category encoding

## Usage
```python
# Prepare your data
df = pd.read_csv('mutual_funds.csv')
# Preprocess features
# ... [preprocessing steps]
# Make predictions
predictions = model.predict(X_new)
```

## Limitations and Considerations
1. Past performance doesn't guarantee future results
2. Market conditions and external factors can impact model accuracy
3. Model should be regularly retrained with recent data
4. Performance may vary across different market cycles

## Future Improvements
1. Incorporate market sentiment analysis
2. Add macroeconomic indicators
3. Implement time-series cross-validation
4. Develop ensemble approach with other algorithms

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- numpy
