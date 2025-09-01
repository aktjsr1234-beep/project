# Advanced URL Review Decoder - Streamlit App
import re, time, json, math
from typing import List, Dict, Optional
import requests, pandas as pd, streamlit as st
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from urllib.parse import urlparse

st.set_page_config(page_title="URL Review Decoder (Advanced)", layout="wide")
st.title("🛒 URL Review Decoder — Advanced (Sentiment, Summary, WordCloud)")
st.markdown("Enter a product URL and get reviews summary, sentiment %, word cloud, and downloadable data.")

# ---------------- Utilities ----------------
def get_ua():
    try:
        return UserAgent().random
    except Exception:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

HEADERS = lambda: {"User-Agent": get_ua(), "Accept-Language": "en-US,en;q=0.9"}

def clean_text(s: Optional[str]) -> str:
    if not s: return ""
    return " ".join(s.replace("\xa0", " ").split())

def polite_sleep():
    time.sleep(0.5)

# --------------- Scrapers ------------------
class AmazonScraper:
    def _asin(self, url: str) -> str:
        m = re.search(r"/dp/([A-Z0-9]{10})", url)
        if m: return m.group(1)
        m = re.search(r"/product-reviews/([A-Z0-9]{10})", url)
        if m: return m.group(1)
        return ""
    def _reviews_url(self, asin: str, page: int) -> str:
        return f"https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}"
    def _parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for div in soup.select("div[data-hook='review']"):
            body = div.select_one("span[data-hook='review-body']")
            text = clean_text(body.get_text(" ")) if body else ""
            rating_el = div.select_one("i[data-hook='review-star-rating'] span, i[data-hook='cmps-review-star-rating'] span")
            rating = None
            if rating_el:
                m = re.search(r"([0-9.]+)\s+out of 5", rating_el.get_text())
                if m:
                    try: rating = float(m.group(1))
                    except: rating = None
            if text: out.append({"text": text, "rating": rating, "source": "amazon"})
        return out
    def fetch(self, url: str, max_reviews: int=200):
        asin = self._asin(url)
        if not asin:
            r = requests.get(url, headers=HEADERS(), timeout=20)
            if r.ok:
                m = re.search(r'"asin"\s*:\s*"([A-Z0-9]{10})"', r.text)
                if m: asin = m.group(1)
        if not asin: return []
        out=[]; page=1
        while len(out)<max_reviews and page<=25:
            try:
                r = requests.get(self._reviews_url(asin,page), headers=HEADERS(), timeout=20)
                if r.status_code!=200: break
                batch = self._parse(r.text)
                if not batch: break
                out.extend(batch); page+=1; polite_sleep()
            except Exception: break
        return out[:max_reviews]

class FlipkartScraper:
    def _url_with_page(self, url: str, page: int) -> str:
        if "page=" in url: return re.sub(r"page=\\d+", f"page={page}", url)
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}page={page}"
    def _parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml"); out=[]
        for blk in soup.select("div._27M-vq, div._1AtVbE"):
            txt_el = blk.select_one("div.t-ZTKy, div._6K-7Co, div._2-N8zT")
            if not txt_el: continue
            text = clean_text(txt_el.get_text(" "))
            if len(text.split())<4: continue
            rating_el = blk.select_one("div._3LWZlK"); rating=None
            if rating_el:
                try: rating=float(clean_text(rating_el.get_text()))
                except: rating=None
            out.append({"text":text,"rating":rating,"source":"flipkart"})
        return out
    def fetch(self, url: str, max_reviews: int=200):
        out=[]; page=1
        while len(out)<max_reviews and page<=25:
            try:
                r = requests.get(self._url_with_page(url,page), headers=HEADERS(), timeout=20)
                if r.status_code!=200: break
                batch = self._parse(r.text)
                if not batch: break
                out.extend(batch); page+=1; polite_sleep()
            except Exception: break
        return out[:max_reviews]

class GenericScraper:
    def _looks_like_review(self, t: str) -> bool:
        if not t: return False
        t_l = t.lower()
        if any(k in t_l for k in ["review","pros","cons","rating","stars","verified","bought"]): return True
        if "★" in t_l: return True
        return len(t.split())>=8
    def fetch(self, url: str, max_reviews: int=200):
        out=[]
        try:
            r = requests.get(url, headers=HEADERS(), timeout=20)
            if r.status_code!=200: return []
            soup = BeautifulSoup(r.text, "lxml")
            for el in soup.select("p,li,div"):
                text = clean_text(el.get_text(" "))
                if self._looks_like_review(text):
                    out.append({"text":text,"rating":None,"source":"generic"})
                if len(out)>=max_reviews: break
        except Exception: return []
        return out[:max_reviews]

SCRAPERS = [(re.compile(r"amazon\\."), AmazonScraper), (re.compile(r"flipkart\\."), FlipkartScraper)]

def choose_scraper(url: str):
    for rx, cls in SCRAPERS:
        if rx.search(url): return cls()
    return GenericScraper()

# --------------- Models (cached) ------------------
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        s_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        s_pipe = pipeline("sentiment-analysis")
    try:
        sum_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception:
        sum_pipe = pipeline("summarization")
    return s_pipe, sum_pipe

sentiment_pipe, summarizer_pipe = load_models()

# --------------- NLP helpers ------------------
def label_reviews(reviews: List[Dict], batch_size: int=16):
    if not reviews: return [], {"total":0,"positive":0,"negative":0,"positive_pct":0.0,"negative_pct":0.0}
    texts = [r["text"] for r in reviews]
    preds = sentiment_pipe(texts, batch_size=batch_size, truncation=True)
    labeled=[]; pos=neg=0
    for r,p in zip(reviews,preds):
        lbl = p.get("label","").upper(); score=float(p.get("score",0.0))
        if "NEG" in lbl: final="NEGATIVE"; neg+=1
        else: final="POSITIVE"; pos+=1
        rr = dict(r); rr["label"]=final; rr["score"]=score; labeled.append(rr)
    total=len(labeled)
    metrics={"total":total,"positive":pos,"negative":neg,"positive_pct":round(100*pos/total,2) if total else 0.0,"negative_pct":round(100*neg/total,2) if total else 0.0}
    return labeled, metrics

def summarize_reviews(texts: List[str], max_chars: int=4000):
    if not texts: return "No reviews to summarize."
    blob = " ".join(texts)[:max_chars]
    try:
        out = summarizer_pipe(blob, max_length=180, min_length=50, do_sample=False)
        if isinstance(out, list) and len(out): return out[0].get("summary_text","").strip()
        return str(out)
    except Exception as e:
        return f"Summarization failed: {e}"

def make_wordcloud(texts: List[str], max_words:int=100):
    joined = " ".join(texts)
    if not joined.strip(): return None
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words)
    wc.generate(joined)
    return wc

# --------------- UI ---------------------------
with st.sidebar:
    st.header("Run options")
    url = st.text_input("Product URL", placeholder="https://www.amazon.in/dp/XXXXXXXXXX")
    max_reviews = st.slider("Max reviews", 20, 1000, 300, step=20)
    sample_count = st.slider("Sample reviews to show", 1, 20, 5)
    run = st.button("Analyze")

if not url:
    st.info("Enter a product URL in the sidebar to begin.")
    st.stop()

if run:
    st.info("Fetching reviews (this may take time on first run while models download).")
    scraper = choose_scraper(url)
    st.write(f"Using scraper: **{scraper.__class__.__name__}**")
    reviews = scraper.fetch(url, max_reviews=max_reviews)
    if not reviews:
        st.error("No reviews found or site blocked the request. Try another URL or lower max_reviews.")
        st.stop()
    st.success(f"Fetched {len(reviews)} reviews. Running sentiment...")
    labeled, metrics = label_reviews(reviews)
    st.metric("Total", metrics["total"])
    c1, c2 = st.columns(2)
    c1.metric("Positive", f\"{metrics['positive']} ({metrics['positive_pct']}%)\")
    c2.metric("Negative", f\"{metrics['negative']} ({metrics['negative_pct']}%)\")
    # pie
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie([metrics['positive'], metrics['negative']], labels=[f\"Positive ({metrics['positive_pct']}%)\", f\"Negative ({metrics['negative_pct']}%)\"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    # summary
    st.subheader("Summary")
    summary = summarize_reviews([r['text'] for r in labeled])
    st.write(summary)
    # wordcloud
    st.subheader("Word Cloud (top words)")
    wc = make_wordcloud([r['text'] for r in labeled], max_words=120)
    if wc:
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)
    else:
        st.write("Not enough text for word cloud.")
    # sample table and downloads
    st.subheader("Sample reviews")
    df = pd.DataFrame(labeled[:sample_count])
    st.dataframe(df[['text','rating','label','score','source']])
    # downloads
    csv = pd.DataFrame(labeled).to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="reviews.csv", mime="text/csv")
    st.download_button("Download summary", summary.encode('utf-8'), file_name="summary.txt", mime="text/plain")
    st.download_button("Download metrics", json.dumps(metrics, indent=2).encode('utf-8'), file_name='metrics.json', mime='application/json')