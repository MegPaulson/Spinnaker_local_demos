import cv2
import numpy as np

def vollath5(f, w, h, size):
    sum = 0
    for i in range(h-1):
        for j in range(w):
            sum += f[coordToIndex(i, j, w)] * f[coordToIndex(i+1, j, w)]
    
    mean = determineMean(f, w, h)
    sum -= h * w * pow(mean, 2)
    #sum += pow(size,2) * pow(mean, 2)
    #sum /= size
    
    return sum

def coordToIndex(row, col, w):
    return row*w + col

def determineMean( f, w, h ):
    aggregate = 0
    for i in range(h):
        for j in range(w):
            aggregate += f[i * w + j]
    
    mean = aggregate / (w * h)
    
    return mean

# # read the image as grayscale
# img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# # get the width and height of the image
# h, w = img.shape

# generate new images of a white square on a black background with progressively larger gaussian blur
size = 50 # modify the size of the square in pixels
img = np.zeros((100, 100), dtype=np.uint8)
print(img.shape)
#img[50-size//2:50+size//2, 50-size//2:50+size//2] = 255

# img = cv2.GaussianBlur(img, (5, 5), 8)

# orig = vollath5(img.flatten().astype(np.float64), 100, 100, size)
# print(orig, orig*16)

# # upsample the generated image to twice the size
# img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
# print(vollath5(img.flatten().astype(np.float64), 400, 400, size))

original_img = img.copy()
vals = []
# generate a white square in the center of the image, of progressively larger size
for i in range(size):
    img = original_img.copy()
    img[50-i:51+i, 50-i:51+i] = 255
    print(vollath5(img.flatten().astype(np.float64), 100, 100, i))
    vals.append(vollath5(img.flatten().astype(np.float64), 100, 100, i))
    # save the image to output folder
    cv2.imwrite(f'C:/Acquisition_Data/vidsample/{i}.png', img)






# show the image
# cv2.imshow('image', img)
# cv2.waitKey(0)

# original_img = img.copy()
# vals = []
# for i in range(1, 10):
#     img = original_img.copy()
#     img = cv2.GaussianBlur(img, (5, 5), i)
#     print(vollath5(img.flatten().astype(np.float64), 100, 100, size))
#     vals.append(vollath5(img.flatten().astype(np.float64), 100, 100, size))

# #plot the results as discrete points
# import matplotlib.pyplot as plt
# plt.scatter(range(size//2), vals)
# plt.show()

# # show the log of the values
# plt.scatter(range(size//2), vals)
# plt.xscale('log')
# #plt.yscale('log')
# plt.show()


# import matplotlib.pyplot as plt
# plt.scatter(range(1, 10), vals)
# plt.show()

# # plot the y axis on a log scale
# plt.scatter(range(1, 10), vals)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()







