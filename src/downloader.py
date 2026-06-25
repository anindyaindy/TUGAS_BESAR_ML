import os
import csv
import urllib.request
import urllib.error
import concurrent.futures
from io import BytesIO
from PIL import Image

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

def download_image(url, save_path):
    """
    Downloads an image from url, validates its integrity using PIL,
    converts it to RGB JPEG, and saves it.
    Retries up to 3 times on connection errors.
    """
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            )
            # Timeout set to 10 seconds
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
            
            # Load image with PIL to verify integrity
            img = Image.open(BytesIO(data))
            img = img.convert('RGB')
            img.save(save_path, 'JPEG', quality=92)
            return True
        except (urllib.error.URLError, Exception) as e:
            # Silent fallback or log on final failure
            if attempt == 2:
                # print(f"Failed to download {url}: {e}")
                pass
    return False

def main():
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
        return download_image(item['url'], save_path)

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Use tqdm to monitor progress
        results = list(tqdm(executor.map(worker, items), total=len(items), desc="Downloading dataset"))
        success_count = sum(1 for r in results if r)

    print(f"Successfully downloaded {success_count}/{len(items)} images.")

if __name__ == '__main__':
    main()
