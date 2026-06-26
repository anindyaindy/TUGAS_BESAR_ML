import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_class_distribution(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar Chart
    sns.countplot(x='label', data=df, hue='label', palette='Set2', ax=axes[0], legend=False)
    axes[0].set_title('Distribusi Jumlah Kelas (REAL vs FAKE)')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Jumlah Gambar')
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    # Pie Chart
    class_counts = df['label'].value_counts()
    axes[1].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette('Set2', 2))
    axes[1].set_title('Persentase Distribusi Kelas')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()

def plot_demographics(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Gender vs Label
    sns.countplot(x='gender', hue='label', data=df, palette='Set2', ax=axes[0])
    axes[0].set_title('Distribusi Gender vs Label (REAL / FAKE)')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Jumlah')
    axes[0].legend(title='Label')
    
    # Age Group vs Label
    age_order = sorted(df['age_group'].dropna().unique())
    sns.countplot(x='age_group', hue='label', data=df, order=age_order, palette='Set2', ax=axes[1])
    axes[1].set_title('Distribusi Kelompok Usia vs Label')
    axes[1].set_xlabel('Kelompok Usia')
    axes[1].set_ylabel('Jumlah')
    axes[1].legend(title='Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demographics_analysis.png'), dpi=150)
    plt.close()

def plot_quality_resolution(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Image Quality vs Label
    sns.countplot(x='image_quality', hue='label', data=df, palette='Set2', ax=axes[0])
    axes[0].set_title('Kualitas Gambar vs Label')
    axes[0].set_xlabel('Kualitas Gambar')
    axes[0].set_ylabel('Jumlah')
    axes[0].legend(title='Label')
    
    # Resolution vs Label
    res_counts = df['resolution'].value_counts()
    # Pick top 5 resolutions for readability
    top_res = res_counts.index[:5]
    df_top_res = df[df['resolution'].isin(top_res)]
    sns.countplot(x='resolution', hue='label', data=df_top_res, order=top_res, palette='Set2', ax=axes[1])
    axes[1].set_title('Resolusi Gambar Terbanyak vs Label')
    axes[1].set_xlabel('Resolusi')
    axes[1].set_ylabel('Jumlah')
    axes[1].legend(title='Label')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_resolution_analysis.png'), dpi=150)
    plt.close()

def plot_difficulty(df, output_dir):
    plt.figure(figsize=(10, 6))
    diff_order = ['Easy', 'Medium', 'Hard']
    existing_diffs = [d for d in diff_order if d in df['detection_difficulty'].unique()]
    sns.countplot(x='detection_difficulty', hue='label', data=df, order=existing_diffs, palette='Set2')
    plt.title('Tingkat Kesulitan Deteksi vs Label')
    plt.xlabel('Tingkat Kesulitan')
    plt.ylabel('Jumlah')
    plt.legend(title='Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'difficulty_analysis.png'), dpi=150)
    plt.close()

def main():
    csv_path = os.path.join('dataset', 'FINAL_DATASET.csv')
    output_dir = 'visualisasi_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print("Membaca data metadata...")
    df = pd.read_csv(csv_path)
    
    print("Menginisialisasi konfigurasi style...")
    setup_style()
    
    print("1. Membuat visualisasi distribusi kelas...")
    plot_class_distribution(df, output_dir)
    
    print("2. Membuat visualisasi analisis demografis...")
    plot_demographics(df, output_dir)
    
    print("3. Membuat visualisasi kualitas & resolusi...")
    plot_quality_resolution(df, output_dir)
    
    print("4. Membuat visualisasi tingkat kesulitan deteksi...")
    plot_difficulty(df, output_dir)
    
    print(f"Semua grafik EDA berhasil dibuat dan disimpan di folder '{output_dir}/'.")

if __name__ == '__main__':
    main()
