# Linefier
A program used to replicate images using a single smooth line.

## Installation
After cloning the repository you will need to install both opencv and numpy. You can install them with `pip3 install -r requirements.txt`  

## Usage
`python3 linefier.py path/to/image [arguments]`  
Use `python3 linefier.py --help` for information about available arguments

## Examples
 |Command|Original image|Generated image|
 |-|-|-|
 |`python3 linefier.py`<br />`mona.png`|![Mona lisa](docs/mona-resized.png)|![Mona lisa](docs/mona.png)|
 |`python3 linefier.py`<br />`mask.png`<br />`--tries 500`<br />`--opacity 0.25`|![Mask](docs/mask-resized.png)|![Mask](docs/mask.png)
 |`python3 linefier.py`<br />`starry.png`<br /> `--tries 500`<br /> `--opacity 0.1`<br /> `--line_color 1`<br /> `--background_color 0`|![Starry night](docs/starry-resized.png)|![Starry night](docs/starry.png)|
 |`python3 linefier.py`<br />`rick.png`<br />`--tries 1000`<br />`--opacity 0.5`<br />`--curves 2000`|![Mask](docs/rick-resized.png)|![Mask](docs/rick.png)|
