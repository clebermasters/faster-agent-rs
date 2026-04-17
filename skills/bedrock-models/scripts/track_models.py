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
from typing import Dict
from datetime import datetime
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data"
MEMORY_FILE = DATA_DIR / "bedrock_models_memory.csv"
REPORT_FILE = DATA_DIR / "bedrock_models_report.html"

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

def generate_html_report(data: Dict, new_info: Dict) -> Path:
    """Generate a professional HTML report and save to fixed path"""
    df = data['dataframe']
    now = datetime.now()
    run_ts = now.strftime("%B %d, %Y at %I:%M %p")
    new_models: list[Dict] = new_info.get('new_models', [])
    is_first_run: bool = new_info.get('is_first_run', False)

    streaming_count = int((df['streaming'].str.lower() == 'yes').sum()) if 'streaming' in df.columns else 0
    multimodal_count = int(df['input_modalities'].str.contains('Image|Video|Audio', case=False, na=False).sum()) if 'input_modalities' in df.columns else 0

    provider_stats = (
        df.groupby('provider')
        .agg(
            count=('model_id', 'count'),
            streaming=('streaming', lambda x: (x.str.lower() == 'yes').sum()),
        )
        .reset_index()
        .sort_values('count', ascending=False)
    )

    new_badge = ""
    new_section = ""
    if new_models:
        new_badge = f'<span class="badge-new">{len(new_models)} NEW</span>'
        rows = "".join(
            f'<tr><td>{m["provider"]}</td><td>{m["model_name"]}</td>'
            f'<td><code>{m["model_id"]}</code></td>'
            f'<td>{m.get("input_modalities","")}</td>'
            f'<td>{m.get("streaming","")}</td></tr>'
            for m in new_models
        )
        new_section = f"""
        <section class="new-models-section">
          <h2>🚨 New Models Detected ({len(new_models)})</h2>
          <table>
            <thead><tr>
              <th>Provider</th><th>Model Name</th><th>Model ID</th>
              <th>Input Modalities</th><th>Streaming</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>"""
    elif is_first_run:
        new_section = '<section class="new-models-section first-run"><h2>🆕 First Run — Baseline Established</h2></section>'
    else:
        new_section = '<section class="no-changes"><h2>✅ No New Models Detected</h2></section>'

    provider_rows = "".join(
        f'<tr><td>{row["provider"]}</td><td class="num">{int(row["count"])}</td>'
        f'<td class="num">{int(row["streaming"])}</td></tr>'
        for _, row in provider_stats.iterrows()
    )

    model_rows = "".join(
        f'<tr>'
        f'<td>{r["provider"]}</td>'
        f'<td>{r["model_name"]}</td>'
        f'<td><code>{r["model_id"]}</code></td>'
        f'<td>{r.get("single_region_support","")}</td>'
        f'<td>{r.get("input_modalities","")}</td>'
        f'<td>{r.get("output_modalities","")}</td>'
        f'<td class="center">{r.get("streaming","")}</td>'
        f'</tr>'
        for r in data['models']
    )

    alert_color = "#e53e3e" if new_models else "#38a169"
    alert_label = f"{len(new_models)} NEW" if new_models else "No changes"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AWS Bedrock Model Tracker — {now.strftime("%Y-%m-%d")}</title>
  <style>
    *{{ margin:0; padding:0; box-sizing:border-box; }}
    body{{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:linear-gradient(135deg,#232f3e 0%,#1a1a2e 100%);
          padding:2rem; min-height:100vh; color:#333; }}
    .container{{ max-width:1400px; margin:0 auto; background:#fff;
                border-radius:12px; box-shadow:0 20px 60px rgba(0,0,0,.4);
                overflow:hidden; }}
    header{{ background:linear-gradient(135deg,#232f3e 0%,#ff9900 100%);
             color:#fff; padding:2.5rem 2rem; display:flex;
             justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; }}
    header h1{{ font-size:1.8rem; }}
    header p{{ opacity:.85; font-size:.95rem; margin-top:.3rem; }}
    .badge-new{{ background:#e53e3e; color:#fff; padding:.3rem .8rem;
                border-radius:20px; font-size:.85rem; font-weight:700; }}
    .stats{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
             gap:1rem; padding:1.5rem; background:#f7f8fa; }}
    .stat-card{{ background:#fff; padding:1.2rem 1.5rem; border-radius:8px;
                border-left:4px solid #ff9900;
                box-shadow:0 2px 8px rgba(0,0,0,.08); }}
    .stat-card h3{{ color:#ff9900; font-size:.85rem; font-weight:600;
                   text-transform:uppercase; letter-spacing:.05em; margin-bottom:.4rem; }}
    .stat-card .num{{ font-size:2rem; font-weight:700; color:#232f3e; }}
    .stat-card .sub{{ font-size:.8rem; color:#888; margin-top:.2rem; }}
    .alert-card{{ border-left-color:{alert_color}; }}
    .alert-card h3{{ color:{alert_color}; }}
    .alert-card .num{{ color:{alert_color}; }}
    section{{ padding:1.5rem 2rem; }}
    section h2{{ font-size:1.15rem; margin-bottom:1rem; color:#232f3e;
                padding-bottom:.5rem; border-bottom:2px solid #ff9900; }}
    .new-models-section{{ background:#fff5f5; border-left:4px solid #e53e3e; margin:1rem; border-radius:8px; }}
    .new-models-section h2{{ color:#e53e3e; border-bottom-color:#e53e3e; }}
    .first-run{{ background:#ebf8ff; border-left-color:#3182ce; }}
    .first-run h2{{ color:#3182ce; border-bottom-color:#3182ce; }}
    .no-changes{{ background:#f0fff4; border-left:4px solid #38a169; margin:1rem; border-radius:8px; }}
    .no-changes h2{{ color:#38a169; border-bottom-color:#38a169; }}
    table{{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    th{{ background:#232f3e; color:#fff; padding:.7rem 1rem; text-align:left;
         font-weight:600; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }}
    td{{ padding:.65rem 1rem; border-bottom:1px solid #eee; }}
    tr:hover td{{ background:#fffbf0; }}
    code{{ background:#f1f1f1; padding:.15rem .4rem; border-radius:4px;
           font-size:.82rem; color:#c7254e; }}
    .num{{ text-align:right; }}
    .center{{ text-align:center; }}
    .two-col{{ display:grid; grid-template-columns:1fr 2fr; gap:1.5rem; }}
    footer{{ background:#232f3e; color:#aaa; padding:1rem 2rem;
             font-size:.8rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem; }}
    @media(max-width:768px){{ .two-col{{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>🤖 AWS Bedrock Model Tracker</h1>
      <p>Automated scan — {run_ts}</p>
    </div>
    {new_badge}
  </header>

  <div class="stats">
    <div class="stat-card">
      <h3>Total Models</h3>
      <div class="num">{data['total_count']}</div>
      <div class="sub">across all providers</div>
    </div>
    <div class="stat-card">
      <h3>Providers</h3>
      <div class="num">{len(data['providers'])}</div>
      <div class="sub">{', '.join(data['providers'][:3])}{'…' if len(data['providers']) > 3 else ''}</div>
    </div>
    <div class="stat-card">
      <h3>Streaming Support</h3>
      <div class="num">{streaming_count}</div>
      <div class="sub">models with streaming</div>
    </div>
    <div class="stat-card">
      <h3>Multimodal Input</h3>
      <div class="num">{multimodal_count}</div>
      <div class="sub">image / video / audio input</div>
    </div>
    <div class="stat-card alert-card">
      <h3>Changes This Run</h3>
      <div class="num">{len(new_models) if not is_first_run else '—'}</div>
      <div class="sub">{alert_label}</div>
    </div>
  </div>

  {new_section}

  <div class="two-col" style="padding:1.5rem 2rem; gap:1.5rem;">
    <section style="padding:0;">
      <h2>Models by Provider</h2>
      <table>
        <thead><tr><th>Provider</th><th class="num">Models</th><th class="num">Streaming</th></tr></thead>
        <tbody>{provider_rows}</tbody>
      </table>
    </section>
    <section style="padding:0;">
      <h2>All Models</h2>
      <div style="overflow-x:auto; max-height:420px; overflow-y:auto;">
        <table>
          <thead><tr>
            <th>Provider</th><th>Model Name</th><th>Model ID</th>
            <th>Region</th><th>Input</th><th>Output</th><th class="center">Stream</th>
          </tr></thead>
          <tbody>{model_rows}</tbody>
        </table>
      </div>
    </section>
  </div>

  <footer>
    <span>Generated {run_ts}</span>
    <span>Source: docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html</span>
  </footer>
</div>
</body>
</html>"""

    DATA_DIR.mkdir(exist_ok=True)
    REPORT_FILE.write_text(html, encoding="utf-8")
    print(f"📊 HTML report saved to {REPORT_FILE}")
    return REPORT_FILE


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
        generate_html_report(data, new_info)
        print("\n✅ Script completed successfully!")
        print(f"REPORT_PATH:{REPORT_FILE}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
