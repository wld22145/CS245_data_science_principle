from skimage import io,transform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import time

def main():
    start_time = time.time()
    img = io.imread("antelope_10001.jpg")
    # resize the image
    img = transform.resize(img, (224,224))
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
    finish_time = time.time()
    print("processing time consumption", finish_time - start_time)

    print("image label")
    print(img_lbl[:10])
    print("region")
    print(regions[:10])

    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        x, y, w, h = r['rect']
        if h == 0 or w == 0:
            continue
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.savefig("selective_search_demo.jpg")
    plt.show()


main()