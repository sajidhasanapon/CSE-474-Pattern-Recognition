import cv2

search_image    = "sea.bmp"        # the larger image (test)
template_image  = "ref.bmp"   # the template (reference)

search      = cv2.imread(search_image, cv2.IMREAD_GRAYSCALE)
template    = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE)
# img is now a numpy array (ndarray)
print("Search shape : ", search.shape)
print("Template shape : ", template.shape)

search_height, search_width  = search.shape
template_height, template_width = template.shape

res = cv2.matchTemplate(search, template, cv2.TM_SQDIFF)
# check other methods in matchTemplate

top_left = cv2.minMaxLoc(res)[2]
print("Top-Left :", top_left, " : Origin at the top_left corner of the image")
bottom_right = (top_left[0]+template_width, top_left[1]+template_height)

original = cv2.imread(search_image, 1)
cv2.rectangle(original, top_left, bottom_right, 0, 2)
cv2.imshow('matching', original)
cv2.waitKey(0)