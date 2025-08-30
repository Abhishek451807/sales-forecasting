Retail Sales Forecasting

Overview

This project builds a predictive analytics dashboard to help retail businesses forecast future sales trends using historical transaction data. The model uses Facebook Prophet for time series forecasting, and the results are visualized in an interactive Power BI dashboard.

Features

Sales Forecasting: Predict future sales based on historical data.

Monthly & Yearly Comparisons: Visualize sales trends across different time periods.

Business Insights: Extract actionable insights like top-selling items, best/worst months, and sales trends.

Interactive Dashboard: Display actual vs. forecasted data, insights, and comparisons in Power BI.

Tools Used

Python: For model building and data analysis (Prophet, Pandas, Scikit-learn).

Power BI: For creating an interactive and shareable dashboard.

Jupyter Notebook: For exploratory data analysis (EDA) and visualizations.

Facebook Prophet: A time series forecasting tool.

Excel: (Optional) For data preprocessing and cleaning.

Files

forecast.py: Python script for training the forecasting model and generating insights.

sample_data.csv: Sample retail sales data (columns: date, sales).

artifacts/: Folder containing the following output files:

forecast_overall.csv: Forecasted sales data.

metrics_overall.csv: Model evaluation metrics (MAE, RMSE).

forecast_overall.png: Actual vs forecasted sales plot.

monthly_sales_overall.png: Monthly sales comparison chart.

yearly_sales_overall.png: Yearly sales comparison chart.

insights_overall.txt: Text file with business insights.

How to Run

Clone the repository:

git clone https://github.com/Abhishek451807/sales-forecasting.git


Install the required Python libraries:

pip install -r requirements.txt


Run the forecasting script:

python forecast.py --data sample_data.csv --freq D --horizon 30 --outdir artifacts


--freq D: Use daily data.

--horizon 30: Forecast the next 30 days.

Results

The script generates:

Forecasted values (forecast_overall.csv)

Performance metrics (MAE, RMSE)

Forecast plots and business insights saved in the artifacts/ folder.

Future Improvements

Enhance the Power BI dashboard with more interactivity (filters by category, store, region).

Implement other forecasting models like ARIMA or XGBoost for comparison.

Include insights like marketing campaign impacts or inventory prediction.