import os
import re
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import arxiv
from datetime import datetime
import textstat

# Configuration
PDF_DIR = "./all_pdfs"
TEXT_DIR = "./processed_papers"
METADATA_PATH = "./paper_metadata.csv"
EDA_DIR = "./analysis_results"
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

def extract_arxiv_id(filename):
    """Extract arXiv ID from PDF filename using regex"""
    match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?\.pdf$', filename)
    return match.group(1) if match else None

def preserve_equations(text):
    """Identify and replace equations with placeholders"""
    equation_patterns = [
        (r'\\begin{equation}(.*?)\\end{equation}', 'DISPLAY', re.DOTALL),
        (r'\\\[(.*?)\\\]', 'DISPLAY', re.DOTALL),
        (r'\$(.*?)\$', 'INLINE', re.DOTALL)
    ]
    
    equations = []
    counter = 1
    
    for pattern, eq_type, flags in equation_patterns:
        for match in re.finditer(pattern, text, flags):
            equation = match.group(1).strip()
            placeholder = f"[EQ_{eq_type}_{counter}]"
            text = text.replace(match.group(0), placeholder)
            equations.append({
                'placeholder': placeholder,
                'equation': equation,
                'type': eq_type,
                'length': len(equation.split())
            })
            counter += 1
    
    return text, equations

def extract_pdf_text(pdf_path):
    """Extract text from PDF with layout preservation"""
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page in doc:
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
        # Clean common artifacts
        text = re.sub(r'\s*\n\s*\d+\s*\n\s*', '\n', text)  # Page numbers
        text = re.sub(r'-\n(\w+)', r'\1', text)  # Hyphenated words
        full_text.append(text)
    
    return '\n\n'.join(full_text)

def get_paper_metadata(arxiv_id):
    """Fetch metadata from arXiv API"""
    try:
        result = next(arxiv.Search(id_list=[arxiv_id]).results())
        return {
            'title': result.title,
            'authors': '; '.join([a.name for a in result.authors]),
            'published_date': result.published.date(),
            'primary_category': result.primary_category,
            'categories': ', '.join(result.categories),
            'doi': result.doi or '',
            'journal_ref': result.journal_ref or 'arXiv'
        }
    except:
        return None

def calculate_readability(text):
    """Calculate readability metrics and statistics"""
    # Remove equations for accurate metrics
    clean_text = re.sub(r'\[EQ_\w+_\d+\]', '', text)
    
    # Advanced sentence splitting
    sentences = re.split(r'(?<!\b\w\w\.)(?<=[.!?])\s+(?=[A-Z])', clean_text)
    sentences = [s.strip() for s in sentences if 10 < len(s) < 500]
    
    words = re.findall(r'\b[\w-]+\b', clean_text)
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words)/len(sentences) if sentences else 0,
        'flesch_reading_ease': textstat.flesch_reading_ease(clean_text),
        'gunning_fog': textstat.gunning_fog(clean_text),
        'smog_index': textstat.smog_index(clean_text)
    }

def process_pdfs():
    """Main processing pipeline"""
    metadata = []
    
    for filename in tqdm(os.listdir(PDF_DIR), desc="Processing PDFs"):
        if not filename.endswith('.pdf'):
            continue
        
        try:
            paper_id = os.path.splitext(filename)[0]
            pdf_path = os.path.join(PDF_DIR, filename)
            
            # Extract and process text
            raw_text = extract_pdf_text(pdf_path)
            processed_text, equations = preserve_equations(raw_text)
            
            # Save processed text
            text_path = os.path.join(TEXT_DIR, f"{paper_id}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            # Save equations metadata
            eq_path = os.path.join(TEXT_DIR, f"{paper_id}_equations.csv")
            if equations:
                pd.DataFrame(equations).to_csv(eq_path, index=False)
            
            # Get metadata
            arxiv_id = extract_arxiv_id(filename)
            meta = get_paper_metadata(arxiv_id) if arxiv_id else None
            
            # Calculate metrics
            metrics = calculate_readability(processed_text)
            
            # Build metadata record
            metadata.append({
                'paper_id': paper_id,
                'arxiv_id': arxiv_id,
                'text_path': text_path,
                'equation_path': eq_path if equations else None,
                'domain': meta['primary_category'].split('.')[0] if meta else 'Unknown',
                'year': meta['published_date'].year if meta else datetime.now().year,
                **metrics,
                **(meta or {})
            })
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_PATH, index=False)
    return df

def generate_analysis(df):
    """Generate analytical visualizations"""
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Temporal Distribution
    plt.subplot(1, 3, 1)
    sns.countplot(x='year', hue='domain', data=df, palette='viridis')
    plt.title('Paper Distribution by Year and Domain')
    plt.xticks(rotation=45)
    
    # Plot 2: Readability Comparison
    plt.subplot(1, 3, 2)
    sns.boxplot(x='domain', y='flesch_reading_ease', data=df, palette='mako')
    plt.title('Readability Scores by Domain')
    plt.ylabel('Flesch Reading Ease')
    plt.xticks(rotation=45)
    
    # Plot 3: Complexity Analysis
    plt.subplot(1, 3, 3)
    sns.scatterplot(
        x='word_count', 
        y='gunning_fog', 
        hue='domain', 
        size='avg_sentence_length',
        sizes=(20, 200),
        data=df,
        palette='rocket'
    )
    plt.title('Complexity Analysis')
    plt.xlabel('Word Count')
    plt.ylabel('Gunning Fog Index')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'full_analysis.png'))
    plt.close()

if __name__ == "__main__":
    print("Starting PDF processing pipeline...")
    metadata_df = process_pdfs()
    
    print("\nGenerating analytical visualizations...")
    generate_analysis(metadata_df)
    
    print(f"""
    {'='*40}
    Processing Complete!
    Processed Papers: {len(metadata_df)}
    Text Files: {TEXT_DIR}
    Metadata CSV: {METADATA_PATH}
    Analysis Plots: {EDA_DIR}
    {'='*40}
    """)