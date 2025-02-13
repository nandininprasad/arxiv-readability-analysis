import arxiv
import os

# Create a directory to save PDFs
download_dir = "./statistics_papers"
os.makedirs(download_dir, exist_ok=True)

# Initialize the arXiv client with rate-limiting
client = arxiv.Client(
    page_size=100,
    delay_seconds=3,
    num_retries=5
)

# Configure the search query
search = arxiv.Search(
    query="cat:stat.*",
    max_results=500,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

success_count = 0
error_count = 0

try:
    for result in client.results(search):
        paper_id = result.entry_id.split("/")[-1].split("v")[0]
        pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")
        
        if os.path.exists(pdf_path):
            print(f"Skipped (exists): {paper_id}.pdf")
            continue
        
        try:
            # Check if PDF URL is available before downloading
            if not result.pdf_url:
                print(f"Skipped (no PDF): {paper_id}")
                error_count += 1
                continue
            
            result.download_pdf(dirpath=download_dir, filename=f"{paper_id}.pdf")
            success_count += 1
            print(f"Downloaded ({success_count}): {paper_id}.pdf")
        
        except Exception as e:
            print(f"Error downloading {paper_id}: {str(e)}")
            error_count += 1

except Exception as e:
    print(f"Fatal error during search: {str(e)}")

print(f"\nDownloaded {success_count} papers | Errors: {error_count}")
print(f"PDFs saved to: {os.path.abspath(download_dir)}")