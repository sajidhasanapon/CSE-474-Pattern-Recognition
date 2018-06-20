import cv2
import numpy

search_image    = "test.jpg"
template_image  = "reference.jpg"

search   = cv2.imread(search_image, cv2.IMREAD_GRAYSCALE)
template = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE)
# imread returns a numpy array (ndarray)

search_height, search_width  = search.shape
template_height, template_width = template.shape

# using cv2.TM_SQDIFF
D_min = float('inf')
for m in range(0, search_width - template_width + 1):
    for n in range(0, search_height - template_height + 1):
        block = search[n:n+template_height, m:m+template_width]
        d = block - template
        D = numpy.sum(d * d)
        if D < D_min:
            D_min = D
            best = m, n

print(best)
