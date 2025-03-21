import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 40})  # Increase base font size

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(os.path.dirname(script_dir), "GPT4_MINI.xlsx")

# Read only the first 6 columns (A to F)
df = pd.read_excel(excel_path, usecols="A:F")
# Use the first row as column names
df.columns = df.iloc[0]
# Remove the first row and reset index
df = df.iloc[1:].reset_index(drop=True)

# Convert numeric columns
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
df['std'] = pd.to_numeric(df['std'], errors='coerce')
df['min'] = pd.to_numeric(df['min'], errors='coerce')
df['max'] = pd.to_numeric(df['max'], errors='coerce')

def plot_task_performance(df):
    """Create bar plots with error bars for each task."""
    # Define tasks and their nice names for plotting
    tasks = {
        'eval_cog_merge_data_v1': 'Cognitive Merge',
        'eval_complex_reasoning': 'Complex Reasoning',
        'eval_single_feature_judge': 'Single Feature Judge',
        'eval_support_params_v1': 'Support Parameters',
        'eval_tunnel_knowledge': 'Tunnel Knowledge'
    }
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(25, 32))
    axes = axes.flatten()
    
    # Define model names and their corresponding row indices in each task group
    models = {
        'llava-1.5-7b-hf-sft-5ep': 0,
        'llava-1.5-7b-hf': 1,
        'GPT4v': 2,
        'qwen2_vl_lora_sft_15ep': 3,
        'qwen2-vl-7b-r1': 4,
        'llava-1.5-13b-hf-sft-5ep': 5,
        'llava-1.5-13b-hf': 6,
        'qwen2_vl_full_sft_2ep': 7,
        'qwen2-vl-5ep': 8
    }
    # Plot each task
    for idx, (task_name, nice_name) in enumerate(tasks.items()):
        ax = axes[idx]
        
        # Get data for this task
        task_data = df[df['Task'] == task_name]
        
        # Collect data for each model
        means = []
        stds = []
        model_names = []
        
        for model_name, model_idx in models.items():
            if model_idx < len(task_data):
                means.append(task_data.iloc[model_idx]['mean'])
                stds.append(task_data.iloc[model_idx]['std'])
                model_names.append(model_name)
        
        x = range(len(means))
        
        # Plot bars with error bars
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        
        # Add value labels on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{means[i]:.2f}',
                   ha='center', va='bottom', fontsize=20)
        
        # Customize the plot
        ax.set_title(nice_name, fontsize=30, pad=20)
        ax.set_ylabel('Score', fontsize=30)
        ax.set_xlabel('Models', fontsize=30)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks with model names
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=30)
        
        # Add legend to the first plot
        # if idx == 0:
        #     ax.legend(['Mean Score with Std Dev'], fontsize=30, loc='lower right')
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Plot overall performance in the last subplot
    ax = axes[-1]
    
    # Calculate average performance for each model across all tasks
    overall_means = []
    overall_stds = []
    model_names = []
    
    for model_name, model_idx in models.items():
        model_means = []
        model_stds = []
        
        for task_name in tasks.keys():
            task_data = df[df['Task'] == task_name]
            if model_idx < len(task_data):
                model_means.append(task_data.iloc[model_idx]['mean'])
                model_stds.append(task_data.iloc[model_idx]['std'])
        
        if model_means:
            overall_means.append(np.mean(model_means))
            overall_stds.append(np.mean(model_stds))
            model_names.append(model_name)
    
    x = range(len(overall_means))
    
    # Plot overall performance as bars
    bars = ax.bar(x, overall_means, yerr=overall_stds, capsize=5, alpha=0.8, color='purple')
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{overall_means[i]:.2f}',
               ha='center', va='bottom', fontsize=20)
    
    # Customize the overall plot
    ax.set_title('Overall Performance (Average Across Tasks)', fontsize=30, pad=20)
    ax.set_ylabel('Average Score', fontsize=30)
    ax.set_xlabel('Models', fontsize=30)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=30)
    # ax.legend(['Mean Score with Std Dev'], fontsize=30, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(script_dir, 'task_performance_bar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Print DataFrame info for debugging
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head(20))
    
    # Create visualization
    plot_task_performance(df)
    print(f"\nVisualization has been created and saved in: {script_dir}")
