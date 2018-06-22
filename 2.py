import cv2
import numpy
from math import log2, ceil

search_image    = "sea.bmp"
template_image  = "ref.bmp"

search   = cv2.imread(search_image, cv2.IMREAD_GRAYSCALE).astype(float)
template = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE).astype(float)
# imread returns a numpy array (ndarray)

search_height, search_width  = search.shape
template_height, template_width = template.shape

center = (search_height/2, search_width/2)
d_height = 2**(ceil(log2(search_height/2.0)) - 2)
d_width  = 2**(ceil(log2(search_width/2.0)) - 2)

while d_height >= 1 and d_width >= 1:
    points = []
    for y in [0, 1, -1]:
        for x in [0, 1, -1]:
            point = (center[0] + y*d_height, center[1] + x*d_width)
            points.append(point)

    D_min = float('inf')
    best = points[0]
    for point in points:
        n, m = point[0] - template_height/2, point[1] - template_width/2
        n, m = ceil(n), ceil(m)
        if n+template_height >= search_height or m+template_width >= search_width:
            continue
        if n < 0 or m < 0:
            continue
        block = search[n:n + template_height, m:m + template_width]
        # using cv2.TM_SQDIFF
        d = block - template
        D = numpy.sum(d * d)
        if D < D_min:
            D_min = D
            center = point
            best = n, m

    d_height /= 2
    d_width /= 2

top_left = best[1], best[0]
print("Top-Left :", top_left, " : Origin at the top_left corner of the image")
bottom_right = (top_left[0]+template_width, top_left[1]+template_height)

original = cv2.imread(search_image, 1)
cv2.rectangle(original, top_left, bottom_right, 0, 2)
cv2.imshow('matching', original)
cv2.waitKey(0)
