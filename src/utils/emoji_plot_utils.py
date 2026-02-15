import requests
from io import BytesIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def add_emoji_image(ax, x, y, url, zoom=0.05):
    """
    Add emoji image from URL at given (x,y) location in plot.
    Works cross-platform (no font dependency).
    """

    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))

        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)

        ax.add_artist(ab)

    except Exception:
        pass