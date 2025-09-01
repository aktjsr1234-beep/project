# URL Review Decoder (Advanced) - Streamlit App

This is an advanced single-file Streamlit application that:
- Scrapes reviews (Amazon, Flipkart, generic fallback)
- Runs sentiment analysis (DistilBERT)
- Generates an abstractive summary (BART)
- Produces a word cloud and visualizations
- Offers CSV/JSON/TXT downloads

## Deploy (Streamlit Cloud / Hugging Face Spaces / Lovable)

### Streamlit Cloud (recommended)
1. Create a public GitHub repo and push these files (app.py, requirements.txt, Dockerfile optional).
2. Go to https://streamlit.io/cloud, click "New app", connect the repo, choose `app.py` and deploy.
3. First run will download model weights (may take few minutes).

### Hugging Face Spaces (Streamlit)
1. Create a new Space (Streamlit) and push repo. Use same requirements.txt. Note: HF Spaces has storage/bandwidth limits.

### Docker (for other hosts / Lovable)
1. Build: `docker build -t url-review-decoder .`
2. Run: `docker run -p 8501:8501 url-review-decoder`

## Notes
- Some websites block scraping. If fetching fails, try a different product page or reduce `max_reviews`.
- Models download on first run — ensure sufficient disk and time.
- Use responsibly and check website Terms of Service.
