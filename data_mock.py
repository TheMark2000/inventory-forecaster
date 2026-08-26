import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

def create_mock_data():
    """Generate 2 years of fake inventory data"""
    
    np.random.seed(42)
    products = [f"SKU_{i:04d}" for i in range(1, 101)]
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys']
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=730),
        end=datetime.now(),
        freq='D'
    )
    
    data = []
    for product in products:
        base_demand = np.random.uniform(50, 500)
        category = np.random.choice(categories)
        
        for date in dates:
            month = date.month
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
            noise = np.random.normal(0, 0.1)
            day_of_week = date.weekday()
            weekly = 1 + 0.2 * (day_of_week == 5 or day_of_week == 6)
            
            demand = base_demand * seasonal * weekly * (1 + noise)
            demand = max(0, int(demand))
            current_stock = demand * np.random.uniform(0.3, 1.5)
            
            data.append({
                'sku': product,
                'date': date.strftime('%Y-%m-%d'),
                'category': category,
                'current_stock': max(0, int(current_stock)),
                'daily_demand': demand,
                'price': np.random.uniform(10, 200)
            })
    
    df = pd.DataFrame(data)
    
    # Save to SQLite
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/inventory.db')
    df.to_sql('inventory_history', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ Created {len(df)} records in data/inventory.db")
    return df

if __name__ == "__main__":
    create_mock_data()