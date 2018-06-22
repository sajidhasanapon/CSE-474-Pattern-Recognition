import cv2
import numpy as np

search_image    = "sea.bmp"
template_image  = "ref.bmp"

search   = cv2.imread(search_image, cv2.IMREAD_GRAYSCALE).astype(float)
template = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE).astype(float)
# imread returns a numpy array (ndarray)

search_height, search_width  = search.shape
template_height, template_width = template.shape

dst = cv2.pyrDown(search)


def hierarchical(search, template, level):
    search_height, search_width = search.shape
    template_height, template_width = template.shape

    # use exhaustive search at the highest level
    if level == 0:
        # using cv2.TM_SQDIFF
        D_min = float('inf')
        for n in range(0, search_height - template_height + 1):
            for m in range(0, search_width - template_width + 1):
                block = search[n:n + template_height, m:m + template_width]
                d = block - template
                D = np.sum(d * d)
                if D < D_min:
                    D_min = D
                    best = n, m
        return 2*best[0], 2*best[1]

    down_best = hierarchical(cv2.pyrDown(search), cv2.pyrDown(template), level-1)
    points = []
    for y in [0, 1, -1]:
        for x in [0, 1, -1]:
            point = (down_best[0] + y, down_best[1] + x)
            points.append(point)

    D_min = float('inf')
    best = points[0]
    for point in points:
        n, m = point[0],  point[1]
        if n+template_height >= search_height or m+template_width >= search_width:
            continue
        if n < 0 or m < 0:
            continue
        block = search[n:n + template_height, m:m + template_width]
        # using cv2.TM_SQDIFF
        d = block - template
        D = np.sum(d * d)
        if D < D_min:
            D_min = D
            best = n, m

    return 2*best[0], 2*best[1]

top_left = hierarchical(search, template, 5)

res = top_left[1]//2, top_left[0]//2
print("Result: ", res)
