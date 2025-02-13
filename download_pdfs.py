import arxiv
import os
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import csv
from tqdm import tqdm

# ======================
# Configuration
# ======================
BASE_DIR = "./arxiv_analysis"
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
TEXT_DIR = os.path.join(BASE_DIR, "texts")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
EDA_DIR = os.path.join(BASE_DIR, "eda_plots")
CURRENT_YEAR = datetime.now().year

# Category distribution (last 5 years)
CATEGORIES = {
    'cs.LG': 800,       # Machine Learning
    'math.AP': 400,     # Applied Mathematics
    'stat.ML': 600,     # Statistical ML
    'econ.EM': 300,     # Econometrics
    'q-bio.QM': 200,    # Quantitative Biology
    'cs.CV': 500        # Computer Vision
}

# ======================
# Setup Functions
# ======================
def create_directories():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)

def init_metadata():
    if not os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'arxiv_id', 'title', 'authors', 'published_date', 'category',
                'pdf_path', 'text_path', 'word_count', 'abstract', 'version'
            ])

# ======================
# Core Functions
# ======================
def download_papers():
    client = arxiv.Client(
        page_size=200,
        delay_seconds=3.5,
        num_retries=5
    )

    with open(METADATA_PATH, 'a', newline='', encoding='utf-8') as meta_file:
        writer = csv.writer(meta_file)
        
        for category, target in CATEGORIES.items():
            print(f"\n🔍 Processing {category} ({target} papers)")
            
            search = arxiv.Search(
                query=f"cat:{category} AND submittedDate:[{CURRENT_YEAR-5}0101 TO {CURRENT_YEAR}1231]",
                max_results=int(target * 1.3),
                sort_by=arxiv.SortCriterion.LastUpdatedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            count = 0
            for result in tqdm(client.results(search), total=target, desc=category):
                if count >= target:
                    break

                paper_id = result.get_short_id()
                pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
                text_path = os.path.join(TEXT_DIR, f"{paper_id}.txt")

                try:
                    # Skip existing entries
                    if os.path.exists(pdf_path) and os.path.exists(text_path):
                        continue

                    # Download PDF
                    result.download_pdf(dirpath=PDF_DIR, filename=f"{paper_id}.pdf")
                    
                    # Convert to text
                    doc = fitz.open(pdf_path)
                    text = "\n".join([page.get_text() for page in doc])
                    word_count = len(text.split())
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                    # Store metadata
                    writer.writerow([
                        paper_id,
                        result.title,
                        '|'.join([a.name for a in result.authors]),
                        result.published.strftime('%Y-%m-%d'),
                        category,
                        pdf_path,
                        text_path,
                        word_count,
                        result.summary,
                        result.entry_id.split('v')[-1]
                    ])

                    count += 1
                    time.sleep(1.2)  # Rate limiting

                except Exception as e:
                    print(f"\n⚠️ Error processing {paper_id}: {str(e)}")
                    continue

# ======================
# EDA Functions
# ======================
def perform_eda():
    df = pd.read_csv(METADATA_PATH)
    
    print("\n📊 Basic Statistics:")
    print(df.describe(include='all'))
    
    # Temporal Analysis
    df['year'] = pd.to_datetime(df['published_date']).dt.year
    plt.figure(figsize=(12, 6))
    sns.countplot(x='year', hue='category', data=df)
    plt.title('Paper Distribution by Year and Category')
    plt.savefig(os.path.join(EDA_DIR, 'yearly_distribution.png'))
    
    # Category Distribution
    plt.figure(figsize=(10, 6))
    df['category'].value_counts().plot(kind='bar')
    plt.title('Paper Count by Category')
    plt.savefig(os.path.join(EDA_DIR, 'category_distribution.png'))
    
    # Word Count Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y='word_count', data=df)
    plt.title('Word Count Distribution by Category')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(EDA_DIR, 'word_count_distribution.png'))
    
    # Author Analysis
    df['num_authors'] = df['authors'].apply(lambda x: len(x.split('|')))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_authors'], bins=20)
    plt.title('Distribution of Number of Authors per Paper')
    plt.savefig(os.path.join(EDA_DIR, 'author_distribution.png'))
    
    # Save EDA report
    with open(os.path.join(EDA_DIR, 'eda_report.txt'), 'w') as f:
        f.write(f"Total Papers: {len(df)}\n")
        f.write(f"Time Range: {df['year'].min()} - {df['year'].max()}\n")
        f.write("\nCategory Counts:\n")
        f.write(str(df['category'].value_counts()))
        f.write("\n\nWord Count Stats:\n")
        f.write(str(df['word_count'].describe()))

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    create_directories()
    init_metadata()
    
    print("🚀 Starting arXiv paper download and processing...")
    download_papers()
    
    print("\n🔬 Performing Exploratory Data Analysis...")
    perform_eda()
    
    print(f"\n✅ All done! Results saved to {BASE_DIR}")
    print(f"📈 EDA plots available in {EDA_DIR}")
    print(f"📄 Metadata file: {METADATA_PATH}")