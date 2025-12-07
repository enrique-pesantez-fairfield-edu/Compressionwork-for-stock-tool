"""
CSV File Inspector
Quick diagnostic tool to see what's in your stroke CSV files
"""

import pandas as pd
import glob
import os

print("="*60)
print("CSV FILE INSPECTOR")
print("="*60)

# Find all CSV files
csv_files = glob.glob("*.csv")

if not csv_files:
    print("\n✗ No CSV files found in current directory!")
    print(f"Current directory: {os.getcwd()}")
else:
    print(f"\n✓ Found {len(csv_files)} CSV file(s)")
    
    # Look specifically for stroke files
    stroke_files = [f for f in csv_files if '_strokes' in f.lower() or '_marked' in f.lower()]
    
    if stroke_files:
        print(f"\nStroke-related files ({len(stroke_files)}):")
        for f in stroke_files:
            print(f"  • {f}")
    
    print(f"\n{'='*60}")
    print("DETAILED INSPECTION")
    print(f"{'='*60}")
    
    # Inspect each file
    for i, file_path in enumerate(stroke_files[:3], 1):  # Inspect first 3
        print(f"\n[{i}] FILE: {file_path}")
        print("-" * 60)
        
        try:
            df = pd.read_csv(file_path)
            
            print(f"\n✓ Loaded successfully")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            
            print(f"\n📋 COLUMNS:")
            for j, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                non_null = df[col].notna().sum()
                null_count = df[col].isna().sum()
                
                print(f"\n  [{j}] {col}")
                print(f"      Type: {dtype}")
                print(f"      Non-null: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
                
                if null_count > 0:
                    print(f"      ⚠ NULL VALUES: {null_count}")
                
                # Show sample values
                sample = df[col].head(3).tolist()
                print(f"      Sample: {sample}")
                
                # Try to detect if numeric
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    numeric_count = numeric_values.notna().sum()
                    if numeric_count > 0:
                        print(f"      ✓ Can convert to numeric: {numeric_count}/{len(df)} values ({numeric_count/len(df)*100:.1f}%)")
                        if numeric_count == len(df):
                            print(f"        Range: [{numeric_values.min():.2f}, {numeric_values.max():.2f}]")
                except:
                    pass
            
            # Summary of numeric columns
            print(f"\n📊 NUMERIC COLUMNS SUMMARY:")
            numeric_cols = []
            for col in df.columns:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    numeric_count = numeric_values.notna().sum()
                    if numeric_count > len(df) * 0.5:  # At least 50% numeric
                        numeric_cols.append(col)
                        print(f"  ✓ {col:25s} ({numeric_count}/{len(df)} numeric)")
                except:
                    pass
            
            if not numeric_cols:
                print(f"  ✗ No numeric columns found!")
            else:
                print(f"\n  Total usable numeric columns: {len(numeric_cols)}")
            
            # Show first few rows
            print(f"\n📄 FIRST 3 ROWS:")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
    
    print(f"\n{'='*60}")
    print("INSPECTION COMPLETE")
    print(f"{'='*60}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not stroke_files:
        print("  • No stroke files found (*_strokes.csv or *_marked_data.csv)")
        print("  • Your files might have different names")
    
    print(f"\n  Files to use for clustering:")
    for f in stroke_files:
        print(f"    ✓ {f}")