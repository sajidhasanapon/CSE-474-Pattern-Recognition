import cv2
import numpy as np
from math import log2, ceil
from time import time

def exhaustive(test, template):
    test_height, test_width = test.shape
    template_height, template_width = template.shape

    # using cv2.TM_SQDIFF
    D_min = float('inf')
    best = 0, 0
    for n in range(0, test_height - template_height + 1):
        for m in range(0, test_width - template_width + 1):
            block = test[n:n+template_height, m:m+template_width]
            d = block - template
            D = np.sum(d * d)
            if D < D_min:
                D_min = D
                best = n, m
    return best

def two_d_log(test, template):
    test_height, test_width = test.shape
    template_height, template_width = template.shape

    center = (test_height / 2, test_width / 2)
    d_height = 2 ** (ceil(log2(test_height / 2.0)) - 2)
    d_width = 2 ** (ceil(log2(test_width / 2.0)) - 2)

    while d_height >= 1 and d_width >= 1:
        points = []
        for y in [0, 1, -1]:
            for x in [0, 1, -1]:
                point = (center[0] + y * d_height, center[1] + x * d_width)
                points.append(point)

        D_min = float('inf')
        best = points[0]
        for point in points:
            n, m = point[0] - template_height / 2, point[1] - template_width / 2
            n, m = ceil(n), ceil(m)
            if n + template_height >= test_height or m + template_width >= test_width:
                continue
            if n < 0 or m < 0:
                continue
            block = test[n:n + template_height, m:m + template_width]
            # using cv2.TM_SQDIFF
            d = block - template
            D = np.sum(d * d)
            if D < D_min:
                D_min = D
                center = point
                best = n, m

        d_height /= 2
        d_width /= 2

    return best

def hierarchical(search, template, level):
    search_height, search_width = search.shape
    template_height, template_width = template.shape

    # use exhaustive search at the highest level
    if level <= 1:
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


def main():

    test_image_path         = input("Test image path : ")
    reference_image_path    = input("Reference image path : ")

    test        = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    template    = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    # imread returns a numpy array (ndarray)

    original = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

    t_start = time()
    ret = cv2.matchTemplate(test.astype(np.uint8), template.astype(np.uint8), cv2.TM_SQDIFF)
    t_end = time()
    t = t_end - t_start
    top_left = cv2.minMaxLoc(ret)[2]
    bottom_right = top_left[0] + template.shape[1], top_left[1] + template.shape[0]
    print("Benchmark : \t", top_left, "in %6d millisecond" % (t * 1000))
    benchmark = original.copy()
    cv2.rectangle(benchmark, top_left, bottom_right, 0, 2)

    t_start = time()
    ret = exhaustive(test, template)
    t_end = time()
    t = t_end - t_start
    top_left = ret[1], ret[0]
    bottom_right = top_left[0]+template.shape[1], top_left[1]+template.shape[0]
    print("Exhaustive : \t", top_left, "in %6d millisecond" %(t*1000))
    img1 = original.copy()
    cv2.rectangle(img1, top_left, bottom_right, 0, 2)

    t_start = time()
    ret = two_d_log(test, template)
    t_end = time()
    t = t_end - t_start
    top_left = ret[1], ret[0]
    bottom_right = top_left[0] + template.shape[1], top_left[1] + template.shape[0]
    print("2D log : \t", top_left, "in %6d millisecond" %(t*1000))
    img2 = original.copy()
    cv2.rectangle(img2, top_left, bottom_right, 0, 2)

    t_start = time()
    ret = hierarchical(test, template, level=2)
    t_end = time()
    t = t_end - t_start
    top_left = ret[1]//2, ret[0]//2
    bottom_right = top_left[0] + template.shape[1], top_left[1] + template.shape[0]
    print("Hierarchical : \t", top_left, "in %6d millisecond" %(t*1000))
    img3 = original.copy()
    cv2.rectangle(img3, top_left, bottom_right, 0, 2)

    display = np.hstack((benchmark, img1, img2, img3))

    cv2.imshow("Benchmark, Exhaustive, 2D log, Hierarchical", display)
    cv2.imshow("Reference", cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
