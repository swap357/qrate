# qrate

Select RAW files and export paths to a text file.

## Install

```bash
pip install -e .
```

## Usage

```bash
qrate <input_dir> --out <export.txt> [--n N] [--ext .NEF]
```

**Defaults:**
- `--n 200` — select newest 200 files
- `--ext .NEF` — file extension to match

## Example

```bash
qrate /Volumes/NIKON/DCIM --out export.txt --n 100
```

## Output format

```
# qrate v0
# input: /Volumes/NIKON/DCIM
# rule: newest_by_mtime
# requested: 100
# selected: 87
# generated: 2025-12-22T21:30:00Z

/Volumes/NIKON/DCIM/100ND6/DSC_0843.NEF
/Volumes/NIKON/DCIM/100ND6/DSC_0844.NEF
...
```

## Downstream usage

```bash
# count selected
wc -l export.txt

# copy files
rsync -av --files-from=export.txt / /destination/

# open for review
xargs -a export.txt open
```
