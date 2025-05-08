import csv
import os

# Specify input and output file paths
input_path = 'initial_prompt_test_responses.csv'  # File in current directory
output_path = 'initial_prompt_test_responses_with_scores.csv'  # Output to current directory

# Create a mapping dictionary
opinion_to_score = {
    'very negative': 0,
    'negative': 0.25,
    'neutral': 0.5,
    'positive': 0.75,
    'very positive': 1
}

# Read and process the CSV file
with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    
    # Get all fieldnames and add 'score'
    fieldnames = reader.fieldnames + ['score']
    
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        opinion = row.get('opinion', '')
        if isinstance(opinion, str) and opinion.lower() in opinion_to_score:
            row['score'] = opinion_to_score[opinion.lower()]
        else:
            row['score'] = 'N/A'
        
        writer.writerow(row)

print(f"File processed and saved to {output_path}")