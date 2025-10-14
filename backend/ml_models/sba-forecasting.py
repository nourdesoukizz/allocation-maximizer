import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class SBAForecasting:
    """
    Syntetos-Boylan Approximation (SBA) for Intermittent Demand Forecasting
    
    SBA is an improvement over Croston's method, specifically designed for spare parts
    and intermittent demand patterns. It addresses the bias in Croston's method.
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        """
        Initialize SBA forecasting model
        
        Args:
            alpha: Smoothing parameter for demand size (0 < alpha < 1)
            beta: Smoothing parameter for inter-demand interval (0 < beta < 1)
        """
        self.alpha = alpha
        self.beta = beta
        self.fitted_params = {}
        self.forecast_results = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load spare parts data from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            DataFrame with processed data
        """
        try:
            # Read Excel file
            df = pd.read_excel(file_path, header=[0, 1])
            
            # The actual structure has years in row 0 and months in row 1
            # Let's read it properly
            xl_file = pd.ExcelFile(file_path)
            raw_data = pd.read_excel(xl_file, header=None)
            
            # Extract years and months
            years = raw_data.iloc[0, 3:].values
            months = raw_data.iloc[1, 3:].values
            
            # Create proper column names
            time_columns = []
            for i, (year, month) in enumerate(zip(years, months)):
                if pd.notna(year) and pd.notna(month) and month != 'Total':
                    time_columns.append(f"{int(year)}-{month}")
            
            # Extract item information and data
            items_data = []
            for idx in range(2, len(raw_data)):
                row = raw_data.iloc[idx]
                if pd.notna(row.iloc[0]):  # If item ID exists
                    item_info = {
                        'item_id': row.iloc[0],
                        'item_name': row.iloc[1],
                        'category': row.iloc[2]
                    }
                    
                    # Extract demand data (skip Total columns)
                    demand_data = []
                    col_idx = 3
                    for year in [2021, 2022, 2023, 2024, 2025]:
                        for month in ['January', 'February', 'March', 'April', 'May', 'June',
                                    'July', 'August', 'September', 'October', 'November', 'December']:
                            if col_idx < len(row) and f"{year}-{month}" in time_columns:
                                demand_data.append(row.iloc[col_idx] if pd.notna(row.iloc[col_idx]) else 0)
                            col_idx += 1
                        col_idx += 1  # Skip Total column
                        if year == 2025:  # 2025 data is incomplete
                            break
            
                    item_info['demand_data'] = demand_data[:len(time_columns)]
                    items_data.append(item_info)
            
            # Create DataFrame
            df = pd.DataFrame(items_data)
            
            # Create time series columns
            for i, col_name in enumerate(time_columns):
                df[col_name] = df['demand_data'].apply(lambda x: x[i] if i < len(x) else 0)
            
            # Drop the temporary demand_data column
            df = df.drop('demand_data', axis=1)
            
            print(f"Loaded data for {len(df)} items with {len(time_columns)} time periods")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_sba_parameters(self, demand_series: np.array) -> Tuple[float, float, float]:
        """
        Calculate SBA parameters for a given demand series
        
        Args:
            demand_series: Array of historical demand values
            
        Returns:
            Tuple of (demand_estimate, interval_estimate, forecast)
        """
        # Remove leading zeros
        non_zero_start = 0
        for i, val in enumerate(demand_series):
            if val > 0:
                non_zero_start = i
                break
        
        demand_series = demand_series[non_zero_start:]
        
        if len(demand_series) == 0 or np.sum(demand_series) == 0:
            return 0, 1, 0
        
        # Initialize estimates
        first_demand_idx = np.where(demand_series > 0)[0]
        if len(first_demand_idx) == 0:
            return 0, 1, 0
        
        # Initial estimates
        z_0 = demand_series[first_demand_idx[0]]  # First non-zero demand
        x_0 = first_demand_idx[0] + 1  # Inter-demand interval
        
        z_t = z_0  # Demand size estimate
        x_t = x_0  # Inter-demand interval estimate
        
        # Track periods since last demand
        periods_since_demand = 0
        
        for t, demand in enumerate(demand_series):
            periods_since_demand += 1
            
            if demand > 0:
                # Update demand size estimate
                z_t = self.alpha * demand + (1 - self.alpha) * z_t
                
                # Update inter-demand interval estimate  
                x_t = self.beta * periods_since_demand + (1 - self.beta) * x_t
                
                # Reset counter
                periods_since_demand = 0
        
        # SBA forecast calculation (corrects Croston's bias)
        if x_t > 0:
            forecast = (z_t / x_t) * (1 - self.alpha / 2)
        else:
            forecast = 0
            
        return z_t, x_t, forecast
    
    def classify_demand_pattern(self, demand_series: np.array) -> str:
        """
        Classify demand pattern based on ADI and CV²
        
        Args:
            demand_series: Array of historical demand values
            
        Returns:
            Demand pattern classification
        """
        non_zero_demands = demand_series[demand_series > 0]
        
        if len(non_zero_demands) == 0:
            return "No Demand"
        
        # Average Demand Interval (ADI)
        total_periods = len(demand_series)
        demand_occasions = len(non_zero_demands)
        adi = total_periods / demand_occasions if demand_occasions > 0 else total_periods
        
        # Coefficient of Variation squared (CV²)
        if len(non_zero_demands) > 1:
            mean_demand = np.mean(non_zero_demands)
            std_demand = np.std(non_zero_demands, ddof=1)
            cv_squared = (std_demand / mean_demand) ** 2 if mean_demand > 0 else 0
        else:
            cv_squared = 0
        
        # Syntetos-Boylan classification
        if adi < 1.32 and cv_squared < 0.49:
            return "Smooth"
        elif adi < 1.32 and cv_squared >= 0.49:
            return "Erratic"
        elif adi >= 1.32 and cv_squared < 0.49:
            return "Intermittent"
        else:
            return "Lumpy"
    
    def fit_and_forecast(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict:
        """
        Fit SBA model and generate forecasts for all items
        
        Args:
            df: DataFrame with item data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        results = {
            'item_forecasts': {},
            'accuracy_metrics': {},
            'demand_classifications': {}
        }
        
        # Get time columns (exclude item info columns)
        time_columns = [col for col in df.columns if '-' in col and any(month in col for month in 
                       ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December'])]
        
        for idx, row in df.iterrows():
            item_id = row['item_id']
            
            # Extract demand series
            demand_series = np.array([row[col] for col in time_columns])
            
            # Classify demand pattern
            pattern = self.classify_demand_pattern(demand_series)
            results['demand_classifications'][item_id] = pattern
            
            # Calculate SBA parameters and forecast
            z_t, x_t, base_forecast = self.calculate_sba_parameters(demand_series)
            
            # Generate 12-month forecasts
            monthly_forecasts = [base_forecast] * forecast_periods
            
            # Store results
            results['item_forecasts'][item_id] = {
                'historical_demand': demand_series.tolist(),
                'demand_estimate': z_t,
                'interval_estimate': x_t,
                'base_forecast': base_forecast,
                'monthly_forecasts': monthly_forecasts,
                'demand_pattern': pattern,
                'item_name': row['item_name'],
                'category': row['category']
            }
            
            # Calculate simple accuracy metrics on historical data
            if len(demand_series) > 12:
                # Use last 12 months for validation
                train_data = demand_series[:-12]
                test_data = demand_series[-12:]
                
                if np.sum(train_data) > 0:
                    z_val, x_val, forecast_val = self.calculate_sba_parameters(train_data)
                    validation_forecast = [forecast_val] * 12
                    
                    # Calculate MAE and RMSE
                    mae = np.mean(np.abs(test_data - validation_forecast))
                    rmse = np.sqrt(np.mean((test_data - validation_forecast) ** 2))
                    
                    # Calculate MAPE (handling zeros)
                    mape_values = []
                    for actual, pred in zip(test_data, validation_forecast):
                        if actual != 0:
                            mape_values.append(abs((actual - pred) / actual))
                    mape = np.mean(mape_values) * 100 if mape_values else 0
                    
                    results['accuracy_metrics'][item_id] = {
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    }
        
        print(f"Completed SBA forecasting for {len(results['item_forecasts'])} items")
        return results
    
    def create_forecast_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for item_id, forecast_data in results['item_forecasts'].items():
            summary_data.append({
                'Item_ID': item_id,
                'Item_Name': forecast_data['item_name'][:50] + '...' if len(forecast_data['item_name']) > 50 else forecast_data['item_name'],
                'Category': forecast_data['category'],
                'Demand_Pattern': forecast_data['demand_pattern'],
                'Historical_Total': sum(forecast_data['historical_demand']),
                'Avg_Monthly_Historical': np.mean(forecast_data['historical_demand']),
                'Monthly_Forecast': round(forecast_data['base_forecast'], 2),
                'Annual_Forecast': round(forecast_data['base_forecast'] * 12, 2),
                'Demand_Estimate': round(forecast_data['demand_estimate'], 2),
                'Interval_Estimate': round(forecast_data['interval_estimate'], 2)
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_forecast_analysis(self, results: Dict, top_n: int = 5):
        """
        Create visualization of forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            top_n: Number of top items to plot
        """
        # Get top items by historical demand
        item_totals = {item_id: sum(data['historical_demand']) 
                      for item_id, data in results['item_forecasts'].items()}
        top_items = sorted(item_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Demand Pattern Distribution
        patterns = list(results['demand_classifications'].values())
        pattern_counts = pd.Series(patterns).value_counts()
        axes[0, 0].pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Demand Pattern Distribution')
        
        # Plot 2: Top Items Historical vs Forecast
        top_item_ids = [item[0] for item in top_items]
        historical_avgs = [np.mean(results['item_forecasts'][item_id]['historical_demand']) 
                          for item_id in top_item_ids]
        forecasts = [results['item_forecasts'][item_id]['base_forecast'] 
                    for item_id in top_item_ids]
        
        x_pos = np.arange(len(top_item_ids))
        width = 0.35
        
        axes[0, 1].bar(x_pos - width/2, historical_avgs, width, label='Historical Avg', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, forecasts, width, label='SBA Forecast', alpha=0.7)
        axes[0, 1].set_xlabel('Items')
        axes[0, 1].set_ylabel('Demand')
        axes[0, 1].set_title('Top Items: Historical vs SBA Forecast')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([f"Item_{i+1}" for i in range(len(top_item_ids))], rotation=45)
        axes[0, 1].legend()
        
        # Plot 3: Forecast Distribution
        all_forecasts = [data['base_forecast'] for data in results['item_forecasts'].values()]
        axes[1, 0].hist([f for f in all_forecasts if f > 0], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Monthly Forecast')
        axes[1, 0].set_ylabel('Number of Items')
        axes[1, 0].set_title('Distribution of Monthly Forecasts')
        
        # Plot 4: Time Series for Top Item
        if top_items:
            top_item_id = top_items[0][0]
            top_item_data = results['item_forecasts'][top_item_id]
            
            # Historical data
            historical = top_item_data['historical_demand']
            months = list(range(1, len(historical) + 1))
            
            axes[1, 1].plot(months, historical, 'b-o', alpha=0.7, label='Historical')
            
            # Forecast
            forecast_months = list(range(len(historical) + 1, len(historical) + 13))
            forecasts = top_item_data['monthly_forecasts']
            axes[1, 1].plot(forecast_months, forecasts, 'r--s', alpha=0.7, label='SBA Forecast')
            
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Demand')
            axes[1, 1].set_title(f'Top Item Forecast: {top_item_id}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate SBA forecasting
    """
    print("SBA (Syntetos-Boylan Approximation) Forecasting for Spare Parts")
    print("=" * 60)
    
    # Initialize SBA model
    sba = SBAForecasting(alpha=0.1, beta=0.1)
    
    # Load data
    file_path = 'Sample_FiveYears_Sales_SpareParts.xlsx'
    df = sba.load_data(file_path)
    
    if df.empty:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Fit model and generate forecasts
    print("\nRunning SBA forecasting...")
    results = sba.fit_and_forecast(df, forecast_periods=12)
    
    # Create summary
    summary_df = sba.create_forecast_summary(results)
    
    print("\nForecast Summary (Top 10 by Historical Demand):")
    print(summary_df.nlargest(10, 'Historical_Total').to_string(index=False))
    
    # Display demand pattern analysis
    patterns = list(results['demand_classifications'].values())
    pattern_counts = pd.Series(patterns).value_counts()
    print(f"\nDemand Pattern Analysis:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} items ({count/len(patterns)*100:.1f}%)")
    
    # Display accuracy metrics if available
    if results['accuracy_metrics']:
        print(f"\nAccuracy Metrics (Sample):")
        sample_items = list(results['accuracy_metrics'].keys())[:5]
        for item_id in sample_items:
            metrics = results['accuracy_metrics'][item_id]
            print(f"  {item_id}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
    
    # Create visualizations
    print("\nGenerating forecast visualizations...")
    sba.plot_forecast_analysis(results, top_n=5)
    
    # Save results to Excel
    output_file = 'SBA_Forecast_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Forecast_Summary', index=False)
        
        # Create detailed forecast sheet for top items
        detailed_data = []
        top_items = summary_df.nlargest(10, 'Historical_Total')['Item_ID'].tolist()
        
        for item_id in top_items:
            item_data = results['item_forecasts'][item_id]
            for month in range(12):
                detailed_data.append({
                    'Item_ID': item_id,
                    'Item_Name': item_data['item_name'],
                    'Forecast_Month': month + 1,
                    'Forecast_Value': item_data['monthly_forecasts'][month],
                    'Demand_Pattern': item_data['demand_pattern']
                })
        
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed_Forecasts', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("SBA forecasting completed successfully!")

if __name__ == "__main__":
    main()
