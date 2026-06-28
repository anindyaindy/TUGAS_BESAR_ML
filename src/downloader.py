import os
import csv
import urllib.request
import urllib.error
import concurrent.futures
import time
from io import BytesIO
from PIL import Image
import argparse

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed in the environment
    def tqdm(iterable, *args, **kwargs):
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        count = 0
        for item in iterable:
            yield item
            count += 1
            if total and count % max(1, total // 20) == 0:
                print(f"Progress: {count}/{total} ({(count/total)*100:.1f}%)")

def download_image(url, save_path, timeout=10, retries=3):
    """
    Downloads an image from url, validates its integrity using PIL,
    converts it to RGB JPEG, and saves it.
    Retries up to 3 times with exponential backoff on failure.
    """
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Comprehensive browser headers to bypass server blocks (like HTTP 403)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            
            # Load image with PIL to verify integrity
            img = Image.open(BytesIO(data))
            img = img.convert('RGB')
            img.save(save_path, 'JPEG', quality=92)
            return True
        except Exception:
            # Exponential backoff retry delay
            time.sleep(0.5 * (attempt + 1))
            
    return False

def main():
    parser = argparse.ArgumentParser(description="Parallel Face Image Downloader")
    parser.add_argument('--workers', type=int, default=8, help="Number of parallel download workers (default: 8)")
    parser.add_argument('--timeout', type=int, default=10, help="Timeout in seconds per download (default: 10)")
    parser.add_argument('--retries', type=int, default=3, help="Number of retries on download failure (default: 3)")
    args = parser.parse_args()

    csv_path = os.path.join('dataset', 'FINAL_DATASET.csv')
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Read items
    items = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                'url': row['image_url'],
                'split': row['dataset_split'],
                'label': row['label'],
                'id': row['image_id']
            })

    print(f"Total images to download: {len(items)}")
    
    # Worker function for parallel execution
    def worker(item):
        save_path = os.path.join('dataset', item['split'], item['label'], f"{item['id']}.jpg")
        if os.path.exists(save_path):
            return True # Skip if already exists
        return download_image(item['url'], save_path, timeout=args.timeout, retries=args.retries)

    print(f"Starting downloader with {args.workers} concurrent workers (timeout={args.timeout}s)...")
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Use tqdm to monitor progress
        results = list(tqdm(executor.map(worker, items), total=len(items), desc="Downloading dataset"))
        success_count = sum(1 for r in results if r)

    print(f"Download complete. Successfully downloaded {success_count}/{len(items)} images.")

if __name__ == '__main__':
    main()
