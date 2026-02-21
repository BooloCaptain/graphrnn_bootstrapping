"""
HTML Report Generator for GraphRNN Experiments

Generates comprehensive HTML reports including:
- Training data visualization
- Training metrics (loss curves)
- Evaluation metrics (MMD scores)
- Generated graph samples
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def encode_image_base64(image_path: Path) -> str:
    """Encode image file as base64 string for embedding in HTML."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_report(
    output_path: Path,
    experiment_config: Dict[str, Any],
    training_metrics: Optional[Dict[str, Any]] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    training_viz_path: Optional[Path] = None,
    generated_viz_path: Optional[Path] = None,
    title: str = "GraphRNN Experiment Report"
) -> None:
    """
    Generate comprehensive HTML report for GraphRNN experiment.
    
    Args:
        output_path: Where to save the HTML report
        experiment_config: Configuration dictionary with experiment parameters
        training_metrics: Dict with 'epochs' and 'loss' lists
        evaluation_metrics: Dict with MMD scores (test/validation sets)
        training_viz_path: Path to training data visualization PNG
        generated_viz_path: Path to generated graphs visualization PNG
        title: Report title
    """
    # Prepare image embeds
    training_img_data = None
    generated_img_data = None
    
    if training_viz_path and training_viz_path.exists():
        try:
            training_img_data = encode_image_base64(training_viz_path)
        except Exception as e:
            print(f"Warning: Could not embed training visualization: {e}")
    
    if generated_viz_path and generated_viz_path.exists():
        try:
            generated_img_data = encode_image_base64(generated_viz_path)
        except Exception as e:
            print(f"Warning: Could not embed generated visualization: {e}")
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        
        .metadata p {{
            margin: 5px 0;
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .config-item {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
        
        .config-item strong {{
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .metric-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        
        .metric-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .no-data {{
            color: #95a5a6;
            font-style: italic;
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        th {{
            background: #34495e;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
        }}
        
        .chart-placeholder {{
            background: #f8f9fa;
            border: 2px dashed #bdc3c7;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #7f8c8d;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="metadata">
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Experiment:</strong> {experiment_config.get('graph_type', 'Unknown')} graphs</p>
        </div>

        <h2>📊 Experiment Configuration</h2>
        <div class="config-grid">
"""
    
    # Add configuration items
    config_items = [
        ("Graph Type", experiment_config.get('graph_type', 'N/A')),
        ("Training Epochs", experiment_config.get('epochs', 'N/A')),
        ("Batch Size", experiment_config.get('batch_size', 'N/A')),
        ("Learning Rate", experiment_config.get('lr', 'N/A')),
        ("Hidden Size", experiment_config.get('hidden_size_rnn', 'N/A')),
        ("Embedding Size", experiment_config.get('embedding_size_rnn', 'N/A')),
        ("Max Previous Nodes", experiment_config.get('max_prev_node', 'N/A')),
        ("Device", experiment_config.get('device', 'N/A')),
    ]
    
    for label, value in config_items:
        html_content += f"""            <div class="config-item">
                <strong>{label}</strong>
                <span>{value}</span>
            </div>
"""
    
    html_content += """        </div>

        <h2>🎯 Training Data</h2>
"""
    
    if training_img_data:
        html_content += f"""        <div class="visualization">
            <img src="data:image/png;base64,{training_img_data}" alt="Training Data Samples">
            <p style="margin-top: 10px; color: #7f8c8d; font-size: 0.9em;">Sample graphs from training set</p>
        </div>
"""
    else:
        html_content += """        <div class="no-data">Training data visualization not available</div>
"""
    
    # Training Metrics
    html_content += """
        <h2>📈 Training Metrics</h2>
"""
    
    if training_metrics and 'epochs' in training_metrics and 'loss' in training_metrics:
        epochs = training_metrics['epochs']
        losses = training_metrics['loss']
        
        if epochs and losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
            min_loss = min(losses)
            avg_loss = sum(losses) / len(losses)
            
            html_content += f"""        <div class="metrics-grid">
            <div class="metric-card success">
                <div class="metric-label">Final Loss</div>
                <div class="metric-value">{final_loss:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Initial Loss</div>
                <div class="metric-value">{initial_loss:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Min Loss</div>
                <div class="metric-value">{min_loss:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Loss</div>
                <div class="metric-value">{avg_loss:.2f}</div>
            </div>
        </div>
        
        <h3>Loss Over Time</h3>
        <table>
            <thead>
                <tr>
                    <th>Epoch</th>
                    <th>Loss</th>
                </tr>
            </thead>
            <tbody>
"""
            
            # Show every 10th epoch for long training runs, or all for short runs
            step = max(1, len(epochs) // 20)
            for i in range(0, len(epochs), step):
                html_content += f"""                <tr>
                    <td>Epoch {epochs[i]}</td>
                    <td>{losses[i]:.4f}</td>
                </tr>
"""
            
            html_content += """            </tbody>
        </table>
"""
    else:
        html_content += """        <div class="no-data">Training metrics not available</div>
"""
    
    # Evaluation Metrics
    html_content += """
        <h2>🔬 Evaluation Metrics</h2>
"""
    
    if evaluation_metrics:
        # Test set metrics
        if 'test' in evaluation_metrics:
            test_metrics = evaluation_metrics['test']
            html_content += """        <h3>Test Set Performance</h3>
        <div class="metrics-grid">
"""
            
            if 'degree_mmd' in test_metrics:
                html_content += f"""            <div class="metric-card">
                <div class="metric-label">Degree MMD</div>
                <div class="metric-value">{test_metrics['degree_mmd']:.4f}</div>
            </div>
"""
            
            if 'clustering_mmd' in test_metrics:
                html_content += f"""            <div class="metric-card">
                <div class="metric-label">Clustering MMD</div>
                <div class="metric-value">{test_metrics['clustering_mmd']:.4f}</div>
            </div>
"""
            
            if 'orbit_mmd' in test_metrics:
                html_content += f"""            <div class="metric-card">
                <div class="metric-label">Orbit MMD</div>
                <div class="metric-value">{test_metrics['orbit_mmd']:.4f}</div>
            </div>
"""
            
            html_content += """        </div>
"""
        
        # Validation set metrics
        if 'validation' in evaluation_metrics:
            val_metrics = evaluation_metrics['validation']
            html_content += """        <h3>Validation Set Performance</h3>
        <div class="metrics-grid">
"""
            
            if 'degree_mmd' in val_metrics:
                html_content += f"""            <div class="metric-card">
                <div class="metric-label">Degree MMD</div>
                <div class="metric-value">{val_metrics['degree_mmd']:.4f}</div>
            </div>
"""
            
            if 'clustering_mmd' in val_metrics:
                html_content += f"""            <div class="metric-card">
                <div class="metric-label">Clustering MMD</div>
                <div class="metric-value">{val_metrics['clustering_mmd']:.4f}</div>
            </div>
"""
            
            if 'orbit_mmd' in val_metrics:
                html_content += f"""            <div class="metric-card">
                <div class="metric-label">Orbit MMD</div>
                <div class="metric-value">{val_metrics['orbit_mmd']:.4f}</div>
            </div>
"""
            
            html_content += """        </div>
"""
    else:
        html_content += """        <div class="no-data">Evaluation metrics not available</div>
"""
    
    # Generated Graphs
    html_content += """
        <h2>🎨 Generated Graphs</h2>
"""
    
    if generated_img_data:
        html_content += f"""        <div class="visualization">
            <img src="data:image/png;base64,{generated_img_data}" alt="Generated Graph Samples">
            <p style="margin-top: 10px; color: #7f8c8d; font-size: 0.9em;">Graphs generated by the trained model</p>
        </div>
"""
    else:
        html_content += """        <div class="no-data">Generated graph visualization not available</div>
"""
    
    # Footer
    html_content += f"""
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✓ HTML report saved to {output_path}")


def save_training_metrics(output_path: Path, epochs: List[int], losses: List[float]) -> None:
    """Save training metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'epochs': epochs,
            'loss': losses
        }, f, indent=2)


def save_evaluation_metrics(
    output_path: Path,
    test_metrics: Optional[Dict[str, float]] = None,
    validation_metrics: Optional[Dict[str, float]] = None
) -> None:
    """Save evaluation metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if test_metrics:
        data['test'] = test_metrics
    if validation_metrics:
        data['validation'] = validation_metrics
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_training_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load training metrics from JSON file."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_evaluation_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics from JSON file."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)
