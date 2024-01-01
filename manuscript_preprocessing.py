import cv2
import numpy as np    
import pytesseract
import matplotlib.pyplot as plt
import maxflow

# load the input image
img = cv2.imread('binarization.jpeg')

# convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
ret,thresh = cv2.threshold(gray,100,255,0)


# Create the graph.
g = maxflow.Graph[int]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(thresh.shape)
# Add non-terminal edges with the same capacity.
g.add_grid_edges(nodeids,100)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
g.add_grid_tedges(nodeids, thresh, 255-thresh)

# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img_denoised = np.logical_not(sgm).astype(np.uint8) * 255

cv2.imwrite("binary.jpg",thresh)
# display Binary Image

cv2.imwrite("Denoisedimage.jpg",img_denoised)
cv2.imshow("Denoisedimage.jpg",img_denoised)

import cv2
import numpy as np

# Load the input image
image = cv2.imread('Denoisedimage.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: The image could not be loaded.")
else:
    # Thresholding to separate letters
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of individual letters
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the input image for the result
    preprocessed_image = image.copy()

    for contour in contours:
        # Extract each letter's bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the individual letter from the binary image
        letter = binary_image[y:y + h, x:x + w]

        # Apply skew correction and slant removal to the letter
        # Skew correction (Hough Line Transform)
        edges = cv2.Canny(letter, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is not None:
            rho, theta = lines[0][0]
            angle = np.degrees(theta)
            if angle < -45:
                angle += 90
            rotated_letter = cv2.warpAffine(letter, cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1), (w, h))

            # Slant removal (Gaussian filter)
            rotated_letter = cv2.GaussianBlur(rotated_letter, (5, 5), 0)

            # Replace the preprocessed letter back into the result image
            preprocessed_image[y:y + h, x:x + w] = rotated_letter

    # Save the preprocessed image with skew correction and slant removal applied to each letter
    cv2.imshow('preprocessed_image.png',preprocessed_image)
    cv2.imwrite('preprocessed_image.png', preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





img=cv2.imread('preprocessed_image.png')
if img is None:
    print("image can't be loaded")
else:
    #Convert to gray scale
    def convert_to_grayscale(img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
    #noise removal
    def blur(img,param):
        img=cv2.medianBlur(img,param)
        return img
    #thresholding
    def threshold(img):
        img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
        return img

    #fetch image shape

    h,w,c =img.shape

    #Obtain boxes 
    boxes=pytesseract.image_to_boxes(img)

    #For loop to draw rectangles on detected boxes
    for b in boxes.splitlines():
        b=b.split(' ')
        segmentedimg=cv2.rectangle(img,(int(b[1]),h-int(b[2])), (int(b[3]), h-int(b[4])) ,(0,255,0),2)
    #Display image
    cv2.imshow('segmented_image.jpg',segmentedimg)
    cv2.imwrite("segmented_image.jpg",segmentedimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

