import os
import re
import numpy as np
import matplotlib.pyplot as plt
import json

# Directory containing the log files
log_dir = '/scratch/a.bip5/BraTS/training_logs'
output_dir = os.path.join(log_dir, 'loss_plots')
processed_files_log = os.path.join(output_dir, 'processed_files.json')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to load processed files log
def load_processed_files(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    else:
        return []

# Function to save processed files log
def save_processed_files(log_file, processed_files):
    with open(log_file, 'w') as f:
        json.dump(processed_files, f)

# Function to parse a log file and extract epoch average loss and mean dice
def parse_log_file(file_path):
    epoch_losses = []
    epoch_dices = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Extract average loss per epoch
            avg_loss_match = re.search(r'epoch \d+ average loss: ([\d\.]+)', line)
            if avg_loss_match:
                avg_loss = float(avg_loss_match.group(1))
                epoch_losses.append(1 - avg_loss)  # Calculating average dice as 1 - loss

            # Extract current mean dice
            mean_dice_match = re.search(r'current mean dice: ([\d\.]+)', line)
            if mean_dice_match:
                mean_dice = float(mean_dice_match.group(1))
                epoch_dices.append(mean_dice)
    return epoch_losses, epoch_dices

# Function to plot the dice scores against epoch number
def plot_dice_scores(file_name, epoch_losses, epoch_dices):
    # Truncate longer array to match the length of the shorter one
    min_length = min(len(epoch_losses), len(epoch_dices))
    epoch_losses = epoch_losses[:min_length]
    epoch_dices = epoch_dices[:min_length]

    epochs = np.arange(1, min_length + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, label='Training Dice', marker='o')
    plt.plot(epochs, epoch_dices, label='Validation Dice', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title(f'Dice Scores per Epoch for {file_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f'{file_name}_dice_scores.png')
    plt.savefig(plot_path)
    plt.close()

# Main script
if __name__ == "__main__":
    processed_files = load_processed_files(processed_files_log)
    
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path) and filename not in processed_files:
            epoch_losses, epoch_dices = parse_log_file(file_path)
            if epoch_losses and epoch_dices:
                plot_dice_scores(filename, epoch_losses, epoch_dices)
                processed_files.append(filename)
    
    save_processed_files(processed_files_log, processed_files)
