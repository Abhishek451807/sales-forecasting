🛍️ Sales Forecasting Dashboard
📊 Overview

This project aims to build a predictive analytics dashboard that helps retail businesses forecast their future sales using historical transaction data. The model leverages Facebook Prophet for time series forecasting and generates predictions, performance metrics, and business insights.

🔑 Features:

Sales Forecasting: Predict future sales using historical data.

Monthly & Yearly Comparisons: Visualize trends across different time periods.

Business Insights: Extract key insights such as the highest/lowest sales months and sales trends.

Interactive Dashboard: Visualize forecasted data with charts and insights in Power BI.

🧰 Tools Used:

Facebook Prophet: For time series forecasting.

Python: For data processing and model building.

Power BI: For building interactive dashboards.

Jupyter Notebook: For exploratory data analysis (EDA) and visualizations.

Pandas & Matplotlib: For data manipulation and charting.

📁 Project Files:

forecast.py: Python script for training the forecasting model (Prophet), generating insights, and saving results.

sample_data.csv: Sample retail sales data (date, sales).

artifacts/: Folder containing output files:

forecast_overall.csv → Forecast data.

metrics_overall.csv → Model evaluation metrics (MAE, RMSE).

forecast_overall.png → Sales forecast plot (actual vs predicted).

monthly_sales_overall.png → Monthly sales comparison.

yearly_sales_overall.png → Yearly sales comparison.

insights_overall.txt → Business insights for decision-making.

💻 How to Run:

Clone the repository:

git clone https://github.com/Abhishek451807/sales-forecasting.git


Install dependencies:

pip install -r requirements.txt


Run the forecasting script:

python forecast.py --data sample_data.csv --freq D --horizon 30 --outdir artifacts


--freq D: Daily frequency for resampling data.

--horizon 30: Forecast the next 30 days.

🌱 Future Improvements:

Enhance the Power BI dashboard with more interactivity (filters by category, store, region).

Implement advanced forecasting models like XGBoost or ARIMA.

Include additional business insights like marketing campaign impact or inventory prediction.

📈 Results:

The model will generate the following files in the artifacts/ folder:

Forecasted values (forecast_overall.csv).

Performance metrics (MAE, RMSE) in metrics_overall.csv.

Visuals: line charts, bar charts, and insights cards saved as images.