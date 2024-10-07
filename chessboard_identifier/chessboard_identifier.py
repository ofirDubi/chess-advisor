import cv2
import imutils 
import numpy as np
import matplotlib.pyplot as plt

class Board(object):
    '''
    possible values for the board:
    None: empty square
    'PW': white pawn
    'PB': black pawn
    'NW': white knight
    'NB': black knight
    'BW': white bishop
    'BB': black bishop
    'RW': white rook
    'RB': black rook
    'QW': white queen
    'QB': black queen
    'KW': white king
    'KB': black king
    '''
    def __init__(self):
        self.board = [[None]*8 for _ in range(8)]



def detect_square_shit(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if ar >= 0.95 and ar <= 1.05:
            return True
    return False

def draw_on_image(mask_img, original_image, color):
    # for every pixel which is not zero in the mask_image, draw a point in the original image - 
    original_image[np.nonzero(mask_img)] = color
    return original_image

def display_subimages(s, img1, img2):
    i,j,l = s
    sub1 = img1[i:i+l+1, j:j+l+1]
    sub2 = img2[i:i+l+1, j:j+l+1]
    plt.subplot(121)
    plt.imshow(sub1)
    plt.subplot(122)
    plt.imshow(sub2, cmap='gray')
    plt.show()

def has_point_nearby(img, i,j,r):
    #TODO - this is highly inefficient, i can do the pad one time instead of many
    # pad_img = np.pad(img, pad_width=r, mode='constant', constant_values=0)
    # i += r
    # j += r
    # non_zeros = np.argwhere(pad_img[i-r:i+r+1, j-r:j+r+1])
    # if len(non_zeros) == 0:
    #     return None
    # sorted(non_zeros, key= lambda p: np.linalg.norm([r,r] - p))
    
    # return non_zeros[0]

    # return the closest point to i,j within a radius r
    curr_point, distance = None, img.shape[0] + img.shape[1]
    for x in range(max(i-r, 0), min(i+r+1, img.shape[0])):
        for y in range(max(j-r, 0), min(j+r+1, img.shape[1])):
            if img[x, y] and np.linalg.norm(np.array((i,j)) - np.array((x,y))) < distance:
                curr_point = (x,y)
                distance = np.linalg.norm(np.array((i,j)) - np.array((x,y)))

    return curr_point, distance


def detect_squares(img, original_image):
    '''
    I have several options to do it - 
    1. Hit or Miss morph - i can search for a pattern of - 
    1 1 1 1
    1 0 0 0
    1 0 0 0
    1 0 0 0

    2. I can use sobel gradients results to calculate the degree of each pixel (or group of pixels), and find ones which go 90)

    3. canny edge detector - 
    https://www.youtube.com/watch?v=sRFM5IEqR2w&ab_channel=Computerphile

    Essentially takes the output of a sobel operator and does some cleaning up.
    a. thins the edges
    We do this by finding local maximum of the edge in a mask, eliminating what is not the local maximum. this way we only keep the
    edge pixels on the "direction" of the edge
    b. does a 2-level thresholding (hysteresis) -
        i. Set two thresholds - high and low.
        ii. If the pixel is above the high threshold, it will be automaticvally included
        iii. If the pixel is below the low threshold, it will be automatically excluded
        iv. If the pixel is between the two thresholds, it will be included only if it is connected to a pixel above the high threshold
    
    4 after canny, do a hit and miss search for edges. ù
    '''
    
    upper_left_kernel = np.array([[1, 1, 1], [1, 0, 0], [1, 0, -1]])
    upper_right_kernel = np.array([[1, 1, 1], [0, 0, 1], [-1, 0, 1]])

    lower_right_kernel =   np.array([[-1, 0, 1], [0, 0, 1], [1, 1, 1]])
    lower_left_kernel = np.array([[1, 0, -1], [1, 0, 0], [1, 1, 1]])

    upper_left_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, upper_left_kernel)
    upper_right_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, upper_right_kernel)
    lower_left_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, lower_left_kernel)
    lower_right_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, lower_right_kernel)

    original_image = draw_on_image(upper_left_img, original_image, (0, 255, 0)) # green
    original_image = draw_on_image(upper_right_img, original_image, (255, 255, 0)) # yellow
    original_image = draw_on_image(lower_left_img, original_image, (255, 0, 0)) # red
    original_image = draw_on_image(lower_right_img, original_image, (0, 0, 255)) # blue
    
    # display_subimages((200, 270, 80), original_image, lower_right_img)
    
    # plt.subplot(121)
    # plt.imshow(original_image)
    # plt.subplot(122)
    # plt.imshow(lower_right_img, cmap='gray')
    # # plt.axis("off")
    # plt.show()


    # cv2.imshow("upper_right_img", upper_right_img)
    # cv2.waitKey(0)
    # squares_rows = {i : [] for i in range(1, img.shape[0] - 1)}
    # squares_cols = {j : [] for j in range(1, img.shape[1] - 1)}
    # for i in range(1, img.shape[0] - 1):
    #     for j in range(1, img.shape[1] - 1):
    #         if upper_right_img[i, j]:
    #             squares_rows[i].append(j)
    #             print(f"found a square at {i,j} value is {img[i,j]}")
    # i want to find some good way to find the squares in the image 

    # let's think about an algorithm to find squares from the edges.
    # a basic method would be to iterate all upper left edges, and search from there for the other 3 edges, each time in increasing length.
    # That means that i iterate on every upper right edge for every possible length of the image.
    # get non-zero indices of the image
    squares_found = []
    square_potential_corners = np.argwhere(upper_left_img)
    # square_length = 4 # just a guess
    
    # i now have a problem that i get multiple squares because i search in a radius. 

    squares_initial_length = 20
    for square_length in range(squares_initial_length , min(img.shape[0], img.shape[1])//6):
        for potential_corner in square_potential_corners:
            #TODO: there is an optimization i can do here, to maybe skip points if i've already found matching squares for them. 
            i, j = potential_corner
            if i == 271 and j == 273 and square_length == 60:
                print("found it")
            if i + square_length >= img.shape[0] or j + square_length >= img.shape[1]:
                continue
            
            closest_r, d_ur = has_point_nearby(upper_right_img, i,j+square_length,3)
            if closest_r is None:
                continue
            closest_l_l, d_ll = has_point_nearby(lower_left_img, i+square_length, j, 3)
            if closest_l_l is None:
                continue
            closest_l_r, d_lr = has_point_nearby(lower_right_img, i+square_length, j+square_length, 3)  
            if closest_l_r is None:
                continue
            # if i got this then there is no point in searching square length + 1 for this point
            print(f"found a square at {i,j} value is {img[i,j]}")
            if i == 271 and j == 273 and square_length == 60 :
                 print(f"found it - {i}")
            # draw_square(original_image, ((i, j), closest_l_r)

            # plt.subplot(122)
            squares_found.append(((i, j), closest_l_r))
            # squares_found.append(((i, j), square_length + int(max(d_ur, d_ll, d_lr))))
            
    squares_found = list(set(squares_found))        
    return squares_found

def draw_square(img, s):
    ul,lr = s
    out_img = cv2.rectangle(img, (ul[1], ul[0]), (lr[1], lr[0]), (0, 255, 0), 2)
    return out_img


# def draw_square(img, s):
#     (i,j),l = s
#     out_img = cv2.rectangle(img, (j, i), (j+l, i+l), (0, 255, 0), 2)
#     return out_img


def squares_to_crosses(img):
    '''
    search for squares with morphological operations and turn them into crosses

    '''

    kernel = np.ones((3,3),np.uint8)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) * 255
    img_hms = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
    # cv2.imshow("img_hms", img_hms)
    # cv2.waitKey(0)
    print(kernel_cross)
    
    # # Create a copy of the input image for the output
    output_image = img.copy()
    # plt.hist(output_image.ravel(),256,[0,256]); plt.show()
    # vals = output_image.mean(axis=2).flatten()
    # # plot histogram with 255 bins
    # b, bins, patches = plt.hist(vals, 255)
    # plt.xlim([0,255])
    # plt.show()
    # TODO: make this faster with numpy
    # padded_matches = np.pad(img_hms, pad_width=1, mode='constant', constant_values=0)
    # output_image[1:-1, 1:-1] = np.where(
    # padded_matches[1:-1, 1:-1],  # Condition where square matches are found
    # kernel_cross[1, 1] * np.ones_like(output_image[1:-1, 1:-1]),  # Center of cross kernel (1s)
    # output_image[1:-1, 1:-1]  # Otherwise, keep original image
    # )
    # output_image[1:-1, :-2] = np.where(padded_matches[1:-1, 1:-1], kernel_cross[1, 0], output_image[1:-1, :-2])  # Left
    # output_image[1:-1, 2:]  = np.where(padded_matches[1:-1, 1:-1], kernel_cross[1, 2], output_image[1:-1, 2:])   # Right
    # output_image[:-2, 1:-1] = np.where(padded_matches[1:-1, 1:-1], kernel_cross[0, 1], output_image[:-2, 1:-1])  # Top
    # output_image[2:, 1:-1]  = np.where(padded_matches[1:-1, 1:-1], kernel_cross[2, 1], output_image[2:, 1:-1])   # Bottom
    plt.subplot(121)
    plt.imshow(output_image, cmap='gray')
    plt.show()

    for i in range(1, output_image.shape[0] - 1):
        for j in range(1, output_image.shape[1] - 1):
            if img_hms[i, j]:
                pre_value = output_image[i-1:i+2, j-1:j+2]
                # print(f"found a square at {i,j} value before replacing is {pre_value}")
                output_image[i-1:i+2, j-1:j+2] = kernel_cross
                post_value = output_image[i,j]
                # print(f"found a square at {i,j} value before replacing is {pre_value}, after replacing - {post_value}")
                # print(f"found a square at {i,j} value in output image is {output_image[i,j]}")   
                if post_value == 0:
                    print(f"found a square at {i,j} value after replacing is {output_image[i,j]}")   

    # for i in image_hms:
    #     for j in i:
    #         if j == 1:
    #             print("found a square")
    plt.subplot(121)
    plt.imshow(img_hms, cmap='gray')
    plt.subplot(122)
    plt.imshow(output_image, cmap='gray')
    plt.show()
    # cv2.imshow("img_hms", img_hms)
    # cv2.imshow("img", img)
    # cv2.imshow("output_image", output_image)
    # cv2.waitKey(0)
    # TODO: do the same thing for squares without top/bottom or left/right

    return output_image


# returns an FEN string of the chess board
def identify_board(image):
    # identify squares. We will search for squares within squares
    # For this we will convert the image into black and white - to make it easier to identify borders.
    # resized = imutils.resize(image, width=300)
    # ratio = image.shape[0] / float(resized.shape[0])
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ratio = 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # use a gaussian kernel to blur the image 

    # run a laplacian kernel on the image to find contours 
    l_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    img_sharp_laplac = cv2.filter2D(gray, -1, l_kernel)

    # run sobel operators to find edges in the image. 
    sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    img_sharp_sobel_x = cv2.filter2D(gray, -1, sobel_x_kernel)
    img_sharp_sobel_y = cv2.filter2D(gray, -1, sobel_y_kernel)

    t_lower = 60  # Lower Threshold 
    t_upper = 200  # Upper threshold 
    aperture_size = 3  # Aperture size 
    img_canny = cv2.Canny(gray, t_lower, t_upper,  
                 apertureSize=aperture_size, L2gradient =True) 
    
    # img_canny_l1 = cv2.Canny(gray, t_lower, t_upper,  
    #              apertureSize=aperture_size, L2gradient =False)
    # thresh = cv2.threshold(img_sharp_laplac, 10, 255, cv2.THRESH_BINARY)[1]

    # Use a close operator to connect any edges of the chess boards that might have been disconnected.
    # kernel_normal = np.ones((3,3),np.uint8)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # img_canny_closed_n = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel_normal)
    plt.subplot(121)
    plt.imshow(img_canny, cmap='gray')
    plt.show()

    img_canny_closed =cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel_cross)
    kernel_e = np.ones((1,1),np.uint8)
    # img_canny_erossed =cv2.erode(img_canny_closed, kernel_e, iterations = 1)

    # cv2.imshow("image", image)
    # # cv2.imshow("thresh", thresh)
    # cv2.imshow("img_sharp_laplac", img_sharp_laplac)
    # cv2.imshow("img_sharp_sobel", img_sharp_sobel_x+img_sharp_sobel_y)
    # cv2.imshow("img_sharp_sobel_x", img_sharp_sobel_x)
    # # cv2.imshow("img_sharp_sobel_y", img_sharp_sobel_y)
    
    #TODO: i have an issue here where the cannay edge detector someties leaves holes in the intersections of chess board squares.
    # I tried to fix this by applying closeing on the image, but because the intersections sometimes contain a pixel in a diognal direction,
    # The closing operator creates a square at the intersection instead of simply connection the edges.
    # this can probably be solved in two ways - 
    # 1. implement the canny myself, but instead of regular sobel i will use only pixels where the degree of the edge is straight line (in either x or y)
    # 2. do some morph which will turn squares into edges, this shouldn't be very hard. 
    # plt.subplot(121)
    # plt.imshow(img_canny_closed, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(img_canny, cmap='gray')
    # plt.axis("off")
    # plt.show()

    # cv2.imshow("img_canny", img_canny)
    # cv2.imshow("img_canny_closed_n", img_canny_closed_n)
    cv2.imshow("img_canny_closed", img_canny_closed)
    # cv2.imshow("img_canny_closed_e", img_canny_erossed)
    # cv2.imshow("img_canny_2", img_canny_l2)
    img_canny_crossed = squares_to_crosses(img_canny_closed)
    squares = detect_squares(img_canny_crossed, original_image=image)
    print("squares found: ", squares)   
    for s in squares:
        s_image = draw_square(image, s)
        print("drawing square - ", s)
    cv2.imshow("s_image", s_image)
    cv2.waitKey(0)

    # detect_squares(img_canny_closed)
    # i will take img_canny_l2, and i will perform opening and dielating on it 1 time to connect edges.
    

    cnts = cv2.findContours(img_canny_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    squares = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
            
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        if detect_square_shit(c):
            squares.append(c)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int") 
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, "s", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
            # show the output image
           
            print("showing contur!!")
            cv2.imshow("Image with contours ", image)
            cv2.waitKey(0) 

def main():
    image = cv2.imread('chessboard_identifier/res/board_webpage.png')
    identify_board(image)
if __name__ == "__main__":
    main()


