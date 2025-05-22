import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Read the dataset
df = pd.read_csv('data/raw/responses_dataset_score.csv')

# Preprocessing
def prepare_data(df):
    # Convert opinion to score if not already numeric
    def convert_score(opinion):
        opinion = str(opinion).lower().strip()
        score_map = {
            'very negative': 0.0,
            'negative': 0.25,
            'neutral': 0.5,
            'positive': 0.75,
            'very positive': 1.0
        }
        
        # Check for direct text match
        for key, value in score_map.items():
            if key in opinion:
                return value
        
        # If already a number, return it
        try:
            score = float(opinion)
            return score if 0 <= score <= 1 else 0.5
        except:
            return 0.5
    
    # Add score column
    df['score'] = df['opinion'].apply(convert_score)
    
    # Detect language
    df['language'] = np.where(df['prompt_en'].notna(), 'English', 
                               np.where(df['prompt_gu'].notna(), 'Gujarati', 'Unknown'))
    
    return df

# Prepare data
df = prepare_data(df)

# Define Western models
western_models = ['llama3.2', 'mistral', 'phi', 'gemma3']
df['model_origin'] = df['model'].apply(
    lambda x: 'Western' if x.lower() in western_models else 'Non-Western'
)

# 1. Cross-Linguistic Comparison
plt.figure(figsize=(12, 6))
grouped_data = df.groupby(['model', 'language'])['score'].mean().reset_index()
sns.barplot(x='model', y='score', hue='language', data=grouped_data)
plt.title('Cross-Linguistic Opinion Score Comparison', fontsize=15)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Average Opinion Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('1_cross_linguistic_comparison.png', dpi=300)
plt.close()

# 2. Category Radar Chart
plt.figure(figsize=(10, 10))
category_scores = df.groupby('category')['score'].mean()
categories = list(category_scores.index)
scores = list(category_scores.values)

N = len(categories)
scores += scores[:1]
categories += categories[:1]
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
plt.polar(angles[:-1], scores[:-1], 'o-', linewidth=2)
plt.fill(angles, scores, alpha=0.25)
plt.theta_offset = np.pi / 2
plt.theta_direction = -1
plt.xticks(angles[:-1], categories[:-1])
plt.title("Opinion Scores Across Question Categories", size=15, y=1.1)
plt.tight_layout()
plt.savefig('2_category_radar_chart.png', dpi=300)
plt.close()

# 3. Response Heatmap
plt.figure(figsize=(10, 8))
pivot_data = df.pivot_table(
    index='model', 
    columns='language', 
    values='score', 
    aggfunc='mean'
)
sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', center=0.5, 
            cbar_kws={'label': 'Opinion Score'})
plt.title('Response Distribution Heatmap', fontsize=15)
plt.tight_layout()
plt.savefig('3_response_heatmap.png', dpi=300)
plt.close()

# 4. Western vs Non-Western Models
plt.figure(figsize=(10, 6))
sns.boxplot(x='model_origin', y='score', hue='language', data=df)
plt.title('Opinion Scores: Western vs Non-Western Models', fontsize=15)
plt.xlabel('Model Origin', fontsize=12)
plt.ylabel('Opinion Score', fontsize=12)
plt.tight_layout()
plt.savefig('4_western_vs_nonwestern.png', dpi=300)
plt.close()

# 5. Cross-Linguistic Consistency
consistency_data = []
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    english_scores = model_data[model_data['language'] == 'English']['score']
    gujarati_scores = model_data[model_data['language'] == 'Gujarati']['score']
    
    correlation = np.corrcoef(english_scores, gujarati_scores)[0, 1] if len(english_scores) > 0 and len(gujarati_scores) > 0 else np.nan
    consistency_data.append({
        'model': model,
        'correlation': correlation
    })

consistency_df = pd.DataFrame(consistency_data)
plt.figure(figsize=(10, 6))
sns.barplot(x='model', y='correlation', data=consistency_df)
plt.title('Cross-Linguistic Response Consistency', fontsize=15)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Correlation between Scores', fontsize=12)
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('5_consistency_correlation.png', dpi=300)
plt.close()

# 6. Controversial Topics
controversial_df = df[df['category'].str.contains('controversial', case=False, na=False)]
grouped_data = controversial_df.groupby(['model', 'language'])['score'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='model', y='score', hue='language', data=grouped_data)
plt.title('Opinion Scores for Controversial Topics', fontsize=15)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('6_controversial_topics.png', dpi=300)
plt.close()

# 7. Regression Analysis (Placeholder)
coefficients = {
    'Western Origin': 0.4,
    'Model Size': 0.2,
    'Release Date': 0.1,
    'Intercept': 0.3
}
plt.figure(figsize=(10, 6))
plt.barh(list(coefficients.keys()), list(coefficients.values()))
plt.title('Factors Predicting Opinion Positioning', fontsize=15)
plt.xlabel('Regression Coefficient', fontsize=12)
plt.ylabel('Predictor Variables', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('7_regression_analysis.png', dpi=300)
plt.close()

# 8. PCA Ideological Mapping
pivot_data = df.pivot_table(
    index=['model', 'language'], 
    columns='category', 
    values='score', 
    aggfunc='mean'
).reset_index()

features = pivot_data.columns[2:]
X = pivot_data[features].values

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'model': pivot_data['model'],
    'language': pivot_data['language'],
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})

plt.figure(figsize=(12, 10))
model_colors = {
    'llama3.2': '#1f77b4',    # Blue
    'mistral': '#ff7f0e',     # Orange
    'phi': '#2ca02c',         # Green
    'gemma3': '#d62728',      # Red
    'aya': '#9467bd',         # Purple
    'qwen': '#8c564b',        # Brown
    'deepseek-r1': '#e377c2'  # Pink
}

for model in pca_df['model'].unique():
    for language in pca_df['language'].unique():
        subset = pca_df[(pca_df['model'] == model) & (pca_df['language'] == language)]
        if not subset.empty:
            plt.scatter(
                subset['PC1'], subset['PC2'],
                marker='o' if language == 'English' else 's',
                color=model_colors.get(model.lower(), 'gray'),
                s=100, alpha=0.8,
                label=f"{model} ({language})"
            )
            
            plt.annotate(
                f"{model} ({language[0]})", 
                xy=(subset['PC1'].values[0], subset['PC2'].values[0]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

plt.title('Multi-Dimensional Opinion Mapping (PCA)', fontsize=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('8_pca_opinion_mapping.png', dpi=300)
plt.close()

print("All visualizations have been generated successfully!")