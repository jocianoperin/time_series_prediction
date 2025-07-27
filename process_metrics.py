import os
import pandas as pd
import re
from pathlib import Path

def get_metric_name(filename):
    """Extract metric name from filename."""
    # Match patterns like NN_MAE_por_categoria.csv or XGBoost_MAE_por_categoria.csv
    match = re.search(r'(?:NN|XGBoost)_([A-Z]+)_por_categoria\.csv', filename)
    if match:
        return match.group(1)  # Returns MAE, MAPE, RMSE, or SMAPE
    return None

def get_model_name(filename):
    """Extract model name from filename."""
    return 'NN' if filename.startswith('NN_') else 'XGBoost'

def process_market_metrics(market_path, output_dir):
    """Process metrics files for a single market."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing files in: {market_path}")
    
    # Get all CSV files for this market
    files = [f for f in os.listdir(market_path) if f.endswith('.csv')]
    print(f"Found {len(files)} files: {files}")
    
    if not files:
        print(f"No CSV files found in {market_path}")
        return
    
    # Group files by metric type (MAE, MAPE, RMSE, SMAPE)
    metrics = {}
    for f in files:
        metric = get_metric_name(f)
        model = get_model_name(f)
        
        if not metric:
            print(f"Warning: Could not parse metric from filename: {f}")
            continue
            
        if metric not in metrics:
            metrics[metric] = {}
        metrics[metric][model] = os.path.join(market_path, f)
    
    print(f"\nFound metrics: {list(metrics.keys())}")
    
    # Process each metric
    for metric, model_files in metrics.items():
        if len(model_files) != 2:  # Skip if we don't have both NN and XGBoost
            print(f"Warning: Missing files for metric {metric} in {market_path}. Found: {list(model_files.keys())}")
            continue
        
        print(f"\nProcessing {metric}...")
        print(f"  NN file: {model_files['NN']}")
        print(f"  XGBoost file: {model_files['XGBoost']}")
        
        try:
            # Read both files
            df_nn = pd.read_csv(model_files['NN'], index_col=0)
            df_xgb = pd.read_csv(model_files['XGBoost'], index_col=0)
            
            print(f"  NN columns: {df_nn.columns.tolist()}")
            print(f"  XGBoost columns: {df_xgb.columns.tolist()}")
            
            # Average the metrics between NN and XGBoost
            df_avg = (df_nn + df_xgb) / 2
            
            # Get all month columns (exclude non-numeric columns)
            month_columns = [col for col in df_avg.columns if str(col).isdigit() or 
                           (isinstance(col, str) and col.replace('.', '').isdigit())]
            
            if not month_columns:
                print(f"  Warning: No month columns found in {metric} files")
                continue
                
            # Sort columns numerically
            month_columns = sorted(month_columns, key=lambda x: float(x) if str(x).replace('.', '').isdigit() else float('inf'))
            print(f"  Month columns: {month_columns}")
            
            # Group into bimesters (2-month periods)
            bimonthly_data = {}
            
            for i in range(0, len(month_columns), 2):
                if i + 1 < len(month_columns):
                    # Get two months for the bimester
                    months = month_columns[i:i+2]
                    bimonth_name = f"{int(float(months[0])):02d}_{int(float(months[1])):02d}"  # e.g., "01_02"
                    
                    # Calculate average for the bimester
                    bimonthly_data[bimonth_name] = df_avg[months].mean(axis=1)
            
            # Create a list to store all dataframes (one per model)
            all_bimonthly_dfs = []
            
            # Process NN data
            for i in range(0, len(month_columns), 2):
                if i + 1 < len(month_columns):
                    months = month_columns[i:i+2]
                    bimonth_name = f"{int(float(months[0])):02d}_{int(float(months[1])):02d}"
                    df_nn_bimonth = df_nn[months].mean(axis=1).round(3)
                    df_nn_bimonth.name = bimonth_name
                    all_bimonthly_dfs.append(df_nn_bimonth)
            
            # Process XGBoost data
            for i in range(0, len(month_columns), 2):
                if i + 1 < len(month_columns):
                    months = month_columns[i:i+2]
                    bimonth_name = f"{int(float(months[0])):02d}_{int(float(months[1])):02d}"
                    df_xgb_bimonth = df_xgb[months].mean(axis=1).round(3)
                    df_xgb_bimonth.name = bimonth_name
                    all_bimonthly_dfs.append(df_xgb_bimonth)
            
            # Combine all data
            df_bimonthly = pd.concat(all_bimonthly_dfs, axis=1)
            
            # Add model and category information
            categories = df_avg.index.tolist()
            num_categories = len(categories)
            
            # Create MultiIndex for rows: (category, model)
            index_tuples = []
            for cat in categories:
                index_tuples.append((cat, 'NN'))
                index_tuples.append((cat, 'XGBoost'))
            
            # Create new index
            multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Categoria', 'Modelo'])
            
            # Prepare data in the correct order
            data = []
            for cat in categories:
                # NN data
                nn_row = {}
                for i in range(0, len(month_columns), 2):
                    if i + 1 < len(month_columns):
                        months = month_columns[i:i+2]
                        bimonth_name = f"{int(float(months[0])):02d}_{int(float(months[1])):02d}"
                        nn_row[bimonth_name] = df_nn[months].loc[cat].mean().round(3)
                data.append(nn_row)
                
                # XGBoost data
                xgb_row = {}
                for i in range(0, len(month_columns), 2):
                    if i + 1 < len(month_columns):
                        months = month_columns[i:i+2]
                        bimonth_name = f"{int(float(months[0])):02d}_{int(float(months[1])):02d}"
                        xgb_row[bimonth_name] = df_xgb[months].loc[cat].mean().round(3)
                data.append(xgb_row)
            
            # Create final dataframe with sorted index
            df_final = pd.DataFrame(data, index=multi_index)
            
            # Sort the index to ensure correct order
            df_final = df_final.sort_index(level=['Categoria', 'Modelo'])
            
            # Reset index to make Categoria and Modelo regular columns
            df_final = df_final.reset_index()
            
            # Save to new CSV
            output_file = os.path.join(output_dir, f"{metric}_por_categoria.csv")
            df_final.to_csv(output_file, index=False, float_format='%.3f')
            print(f"  Created: {output_file} with separate rows for each model")
            
        except Exception as e:
            print(f"  Error processing {metric}: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'analise/analysis_results')
    print(f"Base directory: {base_dir}")
    
    # Process each market
    for market in ['market1', 'market2']:
        market_path = os.path.join(base_dir, market)
        output_dir = os.path.join(base_dir, f"{market}_bimestral")
        
        if os.path.exists(market_path):
            print(f"\n{'='*50}")
            print(f"Processing {market}...")
            print(f"Source: {market_path}")
            print(f"Output: {output_dir}")
            print(f"{'='*50}")
            
            process_market_metrics(market_path, output_dir)
        else:
            print(f"\nWarning: {market_path} not found")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
