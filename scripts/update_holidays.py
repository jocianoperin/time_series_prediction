import os
import pandas as pd
from pathlib import Path

def update_holidays_in_csv(file_path):
    """
    Update holiday information for October 18th in a single CSV file.
    """
    try:
        print(f"\nProcessing file: {file_path}")
        
        # Read the CSV file with headers
        df = pd.read_csv(file_path)
        
        # Print the first few rows to understand the structure
        print("First few rows of the file:")
        print(df.head(3).to_string())
        
        # Get the column names
        columns = df.columns.tolist()
        print(f"Columns in the file: {columns}")
        
        # Find the date and holiday columns (case insensitive)
        date_col = next((col for col in columns if col.lower() == 'date'), None)
        holiday_col = next((col for col in columns if col.lower() == 'holiday'), None)
        
        if date_col is None or holiday_col is None:
            print("Error: Could not find 'Date' or 'Holiday' column in the file")
            return False
            
        # Convert the date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Find rows where month is October (10) and day is 18
        mask = (df[date_col].dt.month == 10) & (df[date_col].dt.day == 18)
        
        # Count how many rows will be updated
        num_updates = mask.sum()
        print(f"Found {num_updates} rows to update (October 18th)")
        
        if num_updates > 0:
            # Update the holiday column to 1 for matching rows
            df.loc[mask, holiday_col] = 1
            
            # Save the updated dataframe back to the same file with headers
            df.to_csv(file_path, index=False)
            print(f"Successfully updated {num_updates} rows in {file_path}")
            
            # Print the first few updated rows for verification
            print("First few updated rows:")
            print(df[mask].head(3).to_string())
            
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def process_market_directories(base_dirs):
    """
    Process all CSV files in the given market directories.
    """
    for base_dir in base_dirs:
        processed_dir = Path(base_dir) / 'processed'
        if not processed_dir.exists():
            print(f"Directory not found: {processed_dir}")
            continue
            
        print(f"Processing directory: {processed_dir}")
        csv_files = list(processed_dir.glob('produto_*.csv'))
        
        for i, csv_file in enumerate(csv_files, 1):
            if update_holidays_in_csv(csv_file):
                if i % 100 == 0 or i == len(csv_files):
                    print(f"Processed {i}/{len(csv_files)} files in {processed_dir}")

if __name__ == "__main__":
    # Define the base directories containing the processed folders
    base_directories = [
        '/home/jociano/Projects/time_series_prediction/data-Market1',
        '/home/jociano/Projects/time_series_prediction/data-Market2'
    ]
    
    process_market_directories(base_directories)
    print("Holiday update completed.")
