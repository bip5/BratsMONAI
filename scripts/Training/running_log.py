import csv
from datetime import datetime


log_path='/scratch/a.bip5/BraTS 2021/running_log.csv'
def log_run_details(details, model_names,best_metrics=None,csv_file_path=log_path):
    # Adding the timestamp and model names to the details
    details["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Define the field names
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            existing_data = list(reader)
    except FileNotFoundError:
        fieldnames = None
    
    if fieldnames:
        new_columns = details.keys() - set(fieldnames)
        fieldnames.extend(new_columns)
    else:
        fieldnames = list(details.keys())
        
    # Adding dynamic fields for models    
    for i in range(len(model_names)):
        fieldnames.append(f"model{i+1}")
    
    for i, model_name in enumerate(model_names):
        details[f"model{i+1}"] = model_name
        
    if best_metrics:  
        for i,best_metric in enumerate(best_metrics):
            details[f"bestmetric{i+1}"] = best_metric
    
    # Rewrite the file with new columns (if any)
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if existing_data:
            writer.writerows(existing_data)
     

    # Writing to csv
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Writing the headers if it's a new file
        file.seek(0, 2)  # Move the cursor to the end of the file
        if file.tell() == 0:  # Check if the file is empty
            writer.writeheader()
        
        writer.writerow(details)


