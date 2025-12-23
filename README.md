# qrate

Image curator. Index, analyze, score, and export your best shots.

## Install

```bash
pip install -e ".[cv]"
```

## Quick Start

```bash
# Index a directory (extracts previews, computes quality scores)
qrate index /path/to/photos

# See what you've got
qrate status /path/to/photos

# Score images for exhibition quality
qrate score /path/to/photos --top 20 --verbose

# Detect bursts and select best shots
qrate cull /path/to/photos

# Export top images
qrate export /path/to/photos --out picks.txt --top 50
```

## Commands

### `qrate index <directory>`

Index RAW files: extract EXIF, generate previews, compute quality metrics.

```bash
qrate index /Volumes/NIKON/DCIM
```

Creates:
- `.qrate.db` — SQLite database with metadata and scores
- `.qrate_previews/` — JPEG previews (auto-rotated via EXIF)

### `qrate status <directory>`

Show index statistics.

```bash
qrate status /Volumes/NIKON/DCIM
```

### `qrate score <directory>`

Rank images by exhibition-worthiness using multi-pass scoring.

```bash
# Top 10 with scores
qrate score /path/to/photos

# Top 50 with detailed breakdown
qrate score /path/to/photos --top 50 --verbose
```

**Scoring passes:**
| Category | Metrics |
|----------|---------|
| Technical | sharpness, subject_sharpness, exposure, noise, dynamic_range |
| Composition | thirds, balance, simplicity, obstruction, subject_clarity |
| Color | harmony, saturation_balance, contrast |
| Collection | uniqueness |

### `qrate cull <directory>`

Detect burst sequences and mark best-of-burst.

```bash
qrate cull /path/to/photos --burst-threshold 2.0
```

### `qrate export <directory>`

Export selected images in various formats.

```bash
# Export paths to text file
qrate export /path/to/photos --out picks.txt --top 100

# Copy files to directory
qrate export /path/to/photos --out /dest/folder --format copy --top 50

# Generate XMP sidecars (for Lightroom/Capture One)
qrate export /path/to/photos --out . --format xmp --top 100 --rating 5 --label red

# Create gallery with JPGs and scores (great for sharing/review)
qrate export /path/to/photos --out best/ --format gallery --top 20
```

**Options:**
- `--format list|copy|xmp|gallery` — output format
- `--top N` — export top N images
- `--min-sharpness N` — filter by sharpness
- `--include-dupes` — include duplicates
- `--all-burst` — include all burst members (not just best)

### `qrate select <input_dir>` (legacy)

Simple selection by modification time.

```bash
qrate select /Volumes/NIKON/DCIM --out export.txt --n 100 --ext .NEF
```

## Workflow Example

```bash
# 1. Index your card
qrate index /Volumes/NIKON/DCIM

# 2. See the scores
qrate score /Volumes/NIKON/DCIM --top 20 -v

# 3. Detect and cull bursts
qrate cull /Volumes/NIKON/DCIM

# 4. Export best shots with Lightroom ratings
qrate export /Volumes/NIKON/DCIM --out . --format xmp --top 50 --rating 5

# 5. Or just get a list to copy
qrate export /Volumes/NIKON/DCIM --out keepers.txt --top 100
cat keepers.txt | xargs -I {} cp {} /destination/
```

## Output Formats

### Text list (`--format list`)
```
/path/to/DSC_0001.NEF
/path/to/DSC_0002.NEF
```

### XMP sidecar (`--format xmp`)
Creates `DSC_0001.xmp` next to each selected RAW with star rating and color label.

### Gallery (`--format gallery`)
Creates a shareable folder with RAW files, JPG previews, and detailed scores:
```
best/
├── raw/
│   ├── 001_DSC_0001.NEF
│   ├── 002_DSC_0002.NEF
│   └── ...
├── jpg/
│   ├── 001_DSC_0001.jpg
│   ├── 002_DSC_0002.jpg
│   └── ...
└── scores.txt
```

`scores.txt` contains rankings with score breakdowns:
```
Rank  Score  File
------------------------------------------------------------
  1    71.7  _DSC0720.NEF
      Technical:   sharp=1859 subj=1.00 exp=0.75
      Composition: thirds=0.62 obstruct=1.00 clarity=1.00
      Color:       harmony=0.50 sat=1.00 contrast=0.77
```

## Supported RAW Formats

NEF, CR2, CR3, ARW, DNG, RAF, ORF, RW2, PEF

## Dependencies

- `rawpy` — RAW file processing
- `Pillow` — image handling
- `imagehash` — perceptual hashing
- `numpy`, `scipy` — numerical analysis
- `opencv-python-headless` (optional) — advanced CV features
