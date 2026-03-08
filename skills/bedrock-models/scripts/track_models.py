#!/usr/bin/env python3
"""
AWS Bedrock Model Tracker
Scrapes AWS documentation and detects new model releases

Author: Cleber Rodrigues
Skill: bedrock-models
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from typing import Dict
from datetime import datetime
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data"
MEMORY_FILE = DATA_DIR / "bedrock_models_memory.csv"

def fetch_bedrock_models(url: str = "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html") -> Dict:
    """Fetch and extract model information from AWS Bedrock documentation"""
    print(f"Fetching content from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")
    
    models = []
    for table_idx, table in enumerate(tables):
        rows = table.find_all('tr')
        print(f"Table {table_idx}: {len(rows)} rows")
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3 and cells[0].name == 'td':
                model_data = {
                    'provider': cells[0].get_text(strip=True),
                    'model_name': cells[1].get_text(strip=True),
                    'model_id': cells[2].get_text(strip=True),
                    'single_region_support': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                    'cross_region_support': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                    'input_modalities': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                    'output_modalities': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                    'streaming': cells[7].get_text(strip=True) if len(cells) > 7 else '',
                }
                models.append(model_data)
    
    if not models:
        raise ValueError(f"No models extracted from {len(tables)} tables")
    
    df = pd.DataFrame(models)
    print(f"\nExtracted {len(models)} models from {len(df['provider'].unique())} providers")
    print(f"\nProviders: {', '.join(sorted(df['provider'].unique()))}")
    
    return {
        'models': models,
        'dataframe': df,
        'total_count': len(models),
        'providers': sorted(df['provider'].unique().tolist())
    }

def detect_new_models(current_models: list) -> Dict:
    """Detect new models by comparing with previous run"""
    DATA_DIR.mkdir(exist_ok=True)
    current_ids = {m['model_id'] for m in current_models}
    
    if not MEMORY_FILE.exists():
        df = pd.DataFrame(current_models)
        df['first_seen'] = datetime.now().isoformat()
        df.to_csv(MEMORY_FILE, index=False)
        return {'new_models': [], 'is_first_run': True}
    
    previous_df = pd.read_csv(MEMORY_FILE)
    previous_ids = set(previous_df['model_id'])
    
    new_ids = current_ids - previous_ids
    new_models = [m for m in current_models if m['model_id'] in new_ids]
    
    if new_models:
        new_df = pd.DataFrame(new_models)
        new_df['first_seen'] = datetime.now().isoformat()
        updated_df = pd.concat([previous_df, new_df], ignore_index=True)
        updated_df.to_csv(MEMORY_FILE, index=False)
    
    return {'new_models': new_models, 'is_first_run': False}

def save_models(data: Dict):
    """Save extracted models to various formats"""
    DATA_DIR.mkdir(exist_ok=True)
    df = data['dataframe']
    
    json_file = DATA_DIR / 'bedrock_models.json'
    with open(json_file, 'w') as f:
        json.dump(data['models'], f, indent=2)
    print(f"Saved {len(data['models'])} models to {json_file}")
    
    csv_file = DATA_DIR / 'bedrock_models.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved {len(data['models'])} models to {csv_file}")

def display_summary(data: Dict):
    """Display a summary of extracted models"""
    df = data['dataframe']
    print("\n" + "="*70)
    print("AWS BEDROCK MODELS SUMMARY")
    print("="*70)
    print(f"\nTotal Models: {data['total_count']}")
    print(f"Total Providers: {len(data['providers'])}")
    print("\nModels by Provider:")
    provider_counts = df['provider'].value_counts()
    for provider, count in provider_counts.items():
        print(f"  {provider}: {count} models")
    
    print("\nSample Models:")
    print(df[['provider', 'model_name', 'model_id']].head(10).to_string(index=False))
    
    if 'streaming' in df.columns:
        streaming_models = df[df['streaming'] == 'Yes'].shape[0]
        print(f"\nModels with Streaming Support: {streaming_models}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        data = fetch_bedrock_models()
        
        new_info = detect_new_models(data['models'])
        
        if new_info['is_first_run']:
            print("\n🆕 First run - all models saved to memory")
        elif new_info['new_models']:
            print(f"\n🚨 Found {len(new_info['new_models'])} NEW models:")
            for model in new_info['new_models']:
                print(f"  • {model['provider']}: {model['model_name']} ({model['model_id']})")
        else:
            print("\n✓ No new models detected")
        
        display_summary(data)
        save_models(data)
        print("\n✅ Script completed successfully!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
