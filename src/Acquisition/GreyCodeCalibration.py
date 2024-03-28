import cv2
import numpy as np
#import image_generate
import os
from PIL import Image, ImageDraw, ImageFont
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List
import operator
import time


def decode(source):
    x_graycodes, y_graycodes = np.zeros((5,5), dtype=int), np.zeros((5,5), dtype=int)
    x_addresses, y_addresses = np.zeros((5,5), dtype=int), np.zeros((5,5), dtype=int)

    def alphanumeric_sort(filename):
        parts = re.split('([0-9]+)', filename)
        # Convert numerical parts to integers for sorting
        parts[1::2] = map(int, parts[1::2])
        return parts

    print(x_graycodes)
    if os.path.isdir(source):
        file_list = sorted(os.listdir(source), key=alphanumeric_sort)
        for filename in file_list:
            #x_graycodes, y_graycodes = np.zeros((5,5), dtype=int), np.zeros((5,5), dtype=int)
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Split the filename to extract information
                parts = filename.split("_")
                
                # Extract information from the filename for sorting
                #cam_number = int(parts[1])
                pattern_type = parts[1]
                pattern_number = parts[2].split(".")[0]  # Remove the file extension

                print("filename:", filename)
                if pattern_type == 'x':
                   x_graycodes = toGraycode(os.path.join(source, filename), x_graycodes, int(pattern_number))
                   print("x pattern number:", pattern_number)
                   #print("x addresses", '\n', x_addresses)
                   print("x graycodes", '\n', x_graycodes)
                elif pattern_type == 'y':
                    y_graycodes = toGraycode(os.path.join(source, filename), y_graycodes, int(pattern_number))
                    print("y pattern number:", pattern_number)

    x_addresses = toBinary(x_graycodes, x_addresses)
    y_addresses = toBinary(y_graycodes, y_addresses)
    # print("x addresses", '\n', x_addresses)
    # print("y addresses", '\n', y_addresses)
    return x_addresses, y_addresses

def binarize_pixel(pixel):
    if pixel > 100:
        pixel = 1
    else:
        pixel = 0
    return pixel

def toAddress(graycodes, addresses):
    for i in range(len(graycodes)-1):
        for j in range(len(graycodes[i])-1):
            addresses[i][j] += graycodes[i][j] 
    return addresses

def toGraycode(path, graycodes, bit_int):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("image:", image)
    print("bit int:", bit_int)

    for i in range(len(image)):
        for j in range(len(image[0])):
            pixel = image[i][j]
            binary_pix =  binarize_pixel(pixel)
            graycodes[i][j] ^= (binary_pix << bit_int)
    return graycodes

def toBinary(graycodes, addresses):
    for i in range(len(graycodes)):
        for j in range(len(graycodes[0])):
            gray_code = graycodes[i][j]
            binary_code = graycodes[i][j]
            while gray_code >> 1:
                gray_code >>= 1
                binary_code ^= gray_code
            addresses[i][j] = binary_code

    return addresses

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image
    _, binary_im = cv2.threshold(image, 100, 1, cv2.THRESH_BINARY)
    
    # save for viewing
    save_path = os.path.join(r"C:\Acquisition_Data\thresholded_imgs", os.path.splitext(os.path.basename(path))[0] + ".png")
    cv2.imwrite(save_path, binary_im * 255)

    pixel_values = binary_im.tolist()
    return pixel_values

#Sort test images by the camera number, encoded dimension (x/y), then pattern order- 0/lsb to 10/msb
def alphanumeric_sort(filename):
    parts = re.split('([0-9]+)', filename)
    # Convert numerical parts to integers for sorting
    parts[1::2] = map(int, parts[1::2])
    return parts

# Unused function- for testing
def process_im(source):
    pixel_values = []
    if os.path.isdir(source):
        file_list = sorted(os.listdir(source), key=alphanumeric_sort)
        for filename in file_list:
            #print(filename)
            pixel_values.append(read_image(os.path.join(source, filename)))
    elif isinstance(source, list):
        for path in source:
            pixel_values.append(read_image(path))
    else:
        pixel_values.append(read_image(source))
    
    # Retrieve original image dimensions to rebuild coordinates into correct matrix shape later
    rows,columns = np.array(pixel_values).shape[1], np.array(pixel_values).shape[2]

    return rows, columns, pixel_values

# Creates an instance of ImageData class for each unique camera ip, and populates with image data
def create_ImageData_instance(source):
    # Directory containing the image files
    folder_path = source

    # Dictionary to store instances of ImageData for each camera
    image_data_instances = {}

    if os.path.isdir(source):
        file_list = sorted(os.listdir(source), key=alphanumeric_sort)
        for filename in file_list:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Split the filename to extract information
                parts = filename.split("_")
                
                # Extract information from the filename for sorting
                cam_number = int(parts[1])
                pattern_type = parts[3]
                pattern_number = parts[4].split(".")[0]  # Remove the file extension
                
                # Create an instance of ImageData for the current camera if it doesn't exist
                if cam_number not in image_data_instances:
                    image_data_instances[cam_number] = ImageData(cam_number)
                
                # Read image data
                img_data = read_image(os.path.join(folder_path, filename))
                
                # Add image data to the instance
                image_data_instances[cam_number].add_pattern(pattern_type, img_data)
                
                # Set rows and columns for the current camera instance
                rows, columns = len(img_data), len(img_data[0])
                image_data_instances[cam_number].rows = rows
                image_data_instances[cam_number].columns = columns

    # Return dictionary of instances
    return image_data_instances #list(image_data_instances.values())
    
# Transpose matrix of pixel values to get gray codes
def to_graycode(pixel_values):
    gray_codes = []
    pixel_values = np.array(pixel_values)
    transpose3d = pixel_values.transpose()

    for sub_array in transpose3d:
        for element in sub_array:
            gray_codes.append(element)

    return gray_codes

# Convert gray code to binary- perform XOR starting with msb and next item
def to_binary(gray_code): 
    gray_code = gray_code[::-1]
    bin = [gray_code[0]]

    for i in range(1, len(gray_code)):
         bin.append(gray_code[i] ^ bin[i-1])
    
    return bin

# Convert binary code to decimal
def to_decimal(binary_code):
    decimal_val = 0
    for i, bit in enumerate(reversed(binary_code)): # Want smallest bit first- enumerate(reversed(binary_code))
        decimal_val += (bit << i)

    return decimal_val

# putting it all together- convert gray code to decimal (coordinate) values
def to_coordinates(gray_codes):
    coordinates = []
    for code in gray_codes:
        decimal_value = to_decimal(to_binary(code))
        coordinates.append(decimal_value)
    coordinates = np.array(coordinates)
    
    return coordinates

# Returns matrix of x-y coordinate pairs
# ie. a matrix of size cameraSensorWidth * cameraSensorHeight, where each element represents the coordinate of a pixel from the projected image
def coordinate_map(x_coords, y_coords, imagewidth, imageheight):
    matrix = []
    # Put coordinate list back into matrix form based on input image size
    # Transpose x and y coordinate matrices to get proper orientation relative to camera coordinates
    if imageheight <= 0 or imagewidth <= 0:
         raise ValueError("invalid matrix size")
    
    matrixX = np.transpose(x_coords.reshape((imagewidth,imageheight)))
    matrixY = np.transpose(y_coords.reshape((imagewidth,imageheight)))
    
    # Collect x and y pairs into single matrix
    for i in range(imageheight): # number of rows
        row = []
        for j in range(imagewidth): # number of columns
            row.append((matrixX[i][j], matrixY[i][j]))
        matrix.append(row)
        
    return matrix

def get_center_element(matrix):

    # projecting at pixel that maps to center element of camera sensor
    rows = len(matrix)
    cols = len(matrix[0])
    print(rows,cols)
    # print("Center element retrieved: ",rows,cols)

    center_row = rows // 2
    center_col = cols // 2
    
    return matrix[center_row][center_col]

def calculate_center_pixel(pixel_coordinates):
    sum_x = sum(coord[0] for coord in pixel_coordinates)
    sum_y = sum(coord[1] for coord in pixel_coordinates)

    # Calculate the average coordinates
    avg_x = sum_x // len(pixel_coordinates)
    avg_y = sum_y // len(pixel_coordinates)

    average_coordinate = (avg_x, avg_y)
    print(average_coordinate)
    return average_coordinate
    
def visualize_matrix(coordinate_matrix, horizontal_res, vertical_res):

    image = np.zeros((len(coordinate_matrix), len(coordinate_matrix[0]), 3), dtype = np.uint8)
    # Access each coordinate pair
    for i, row in enumerate(coordinate_matrix):
        for j, pair in enumerate(row):
            # Red channel- divide x coordinate value by horizontal resolution of projected image
            red = (pair[0]/(horizontal_res-1))*255
            # green channel- divide y coordinate value by vertical resolution of projected image
            green = (pair[1]/(vertical_res-1))*255
            blue = 0
            image[i,j] = [blue, green, red]

    cv2.imwrite(r"C:\Acquisition_Data\crosshairs\vis.png", image)

# unused function- for testing
def crop_to_window(source, destination, crop_width, crop_height, center_x, center_y):
    if not os.path.exists(destination):
        os.makedirs(destination)

    for filename in os.listdir(source):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  
            with Image.open(os.path.join(source, filename)) as img:
                left = max(0, center_x - crop_width // 2)
                top = max(0, center_y - crop_height // 2)
                right = min(img.width, center_x + crop_width // 2 + crop_width % 2)
                bottom = min(img.height, center_y + crop_height // 2 + crop_height % 2)

                cropped_img = img.crop((left, top, right, bottom))

                cropped_img.save(os.path.join(destination, filename))

def generate_single_pixel(image_dimensions, output_directory, center_elements):
    filename = 'crosshairs.png'
    width, height = image_dimensions

    print("generating image:", width, height)

    # Create a new black image
    img = Image.new('RGB', (width, height), color='black')

    # Draw white pixels at each center element
    draw = ImageDraw.Draw(img)
    for pair in center_elements:
        x, y = pair
        print(pair)

        if x >= width or y >= height:
            print("Center coordinate outside image boundaries, skipping")
            continue

        draw.point((x, y), fill='white')

    img.save(os.path.join(output_directory, filename))
    
def generate_crosshair_pattern(image_dimensions, output_directory, center_elements):
    
    filename= 'crosshairs.png'
    width, height = image_dimensions

    print("generating image:", width , height)

    # Create a new black image
    img = Image.new('RGB', (width, height), color='black')

    # Draw white crosshair centered at the center element of each matrix
    draw = ImageDraw.Draw(img)
    for pair in center_elements:
        x, y = pair
        print(pair)

        if x > width:
            print("x center coordinate greater than image width, vertical line failed")
        if y > height:
            print("y center coordinate greater than image height, horizontal line failed")
        # line2 = draw.line([(x, 100), (x, height-100)], fill='white', width=1)  # Vertical line
        # line1 = draw.line([(100, y), (width-100, y)], fill='white', width=1)  # Horizontal line
        
        line2 = draw.line([(x, y+100), (x, y-100)], fill='white', width=1)  # Vertical line
        line1 = draw.line([(x+100, y), (x-100, y)], fill='white', width=1)  # Horizontal line

        # if not line1 or not line2:
        #     raise ValueError("One or more lines could not be drawn")


    img.save(os.path.join(output_directory, filename))

# Takes a coordinate matrix, calculates the coordinates of the projector pixel in the center of the camera sensor (used for reprojection),
# and the distance between projector pixels in mm -> these two values are the end goal of calibration
def process_matrix(coordinate_matrix):
    # Want to measure center-to-center distance between pixels in order to establish global scale for focus metric...
    # each "blob" that represents a projector pixel should have the same x/y coordinates within camera matrix
    
    # binarize the matrix based on "blobs" of unique projector pixel coordinates, and locate each blob center
    valid_proj_coordinates, centers = binarize_matrix(coordinate_matrix)

    center = calculate_center_pixel(valid_proj_coordinates)
    distance = calculate_average_distance(centers)
    
    return center, distance

# Isolate blobs from coordinate matrix
def binarize_matrix(matrix):

    # look up table for color visualization of blobs
    color_table = {
                0: (0, 255, 0),
                1: (255, 127, 0),
                2: (255, 255, 255),
                3: (255, 0, 0),
                4: (0, 255, 0),
                5: (0, 0, 255),
                6: (255, 255, 0),
                7: (0, 255, 255),
                8: (255, 0, 255),
                9: (128, 128, 128),
                10: (128, 0, 0),
                11: (128, 128, 0),
                12: (0, 0, 128),
                13: (128, 0, 128),
                14: (0, 128, 128),
                15: (192, 192, 192),
                16: (255, 192, 203),
                17: (165, 42, 42),
                18: (64, 224, 208),
                19: (255, 215, 0),
                }

    # Find all unique projector pixel coordinate pairs in the matrix- pairs are sorted in ascending order in y, then x (sorting only necessary for visualization)
    unique_values = sorted(set(tuple(pair) for row in matrix for pair in row), key=lambda x: (x[1], x[0]))
    temp_mat = np.zeros_like(matrix, dtype=int)
    # Reshape list into numpy array for easier handling
    matrix = np.reshape(matrix, temp_mat.shape)

    # initialize binary matrix with 1 less dimension than coordinate matrix (0th and 1st axis instead of 0th,1st,2nd) and 3 channels
    binary_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

    i = 1
    avg_camera_coordinates = []
    indices = None
    min_pix_size = float('inf')
    max_pix_size = 0
    valid_projector_pixel_coordinates = []
    for val in unique_values:
        
        
        # Find all indices in matrix where the unique coordinate value occurs
        # we are only searching the 2nd axis- if the matrix element matches the unique coordinate value, np.all returns true
        # np.where returns lists of the 0th and 1st dimension indices where we find a match

        indices = np.where(np.all(matrix == val, axis=(2))) 
        # if either index list (x or y) contains zero, or an index that matches the size of the matrix in that dimension, 
        # then there is an edge coordinate in the blob represented by coordinate "val"
        # indices[0] and matrix.shape[0] correspond to y, indices[1] and matrix.shape[1] correspond to x
        edge_coordinate = np.any((indices[1] == 0) | (indices[1] == matrix.shape[1]-1)) | np.any((indices[0] == 0) | (indices[0] == matrix.shape[0]-1))

        # if the projector pixel doesn't touch the edge of the camera window, its size is above the threshold, and it has a valid coordinate- collect its average (center) coordinate in camera pixels
        # (0,0) indicates an invalid coordinate- a pixel is not in this location since the camera never saw white here. 
        # Image features with this coordinate are the screen door/gap between pixels, and dust on the lens

        size_threshold = 30000 # 19000

        if val == (0,0):
            print("invalid coordinate:", val)
            
        if not edge_coordinate and len(indices[0]) > size_threshold and val != (0,0): #not edge_coordinate and
            #print("unique coordinate:", val)
            valid_projector_pixel_coordinates.append(val)

            if len(indices[0]) < min_pix_size:
                min_pix_size = len(indices[0])
            if len(indices[0]) > max_pix_size:
                max_pix_size = len(indices[0])

            avg_camera_coordinates.append((int(np.mean(indices[1])), int(np.mean(indices[0]))))
        
 
        # Populate all elements of the binary matrix that have a match for a given coordinate value "val"
        # Creates a checkerboard pattern where each square represents a projected pixel
        if val != (0,0):
            binary_matrix[indices] = [ color_table[i%19][2], color_table[i%19][1], color_table[i%19][0]]
        i+=1

    print("minimum size:", min_pix_size)
    print("maximum size:", max_pix_size)
        
    binary_matrix_uint8 = binary_matrix.astype('uint8')

    image = Image.fromarray(binary_matrix_uint8)
    draw = ImageDraw.Draw(image)

    # draw center coordinates on each blob for color visualization
    for i,coord in enumerate(avg_camera_coordinates):#coord in avg_camera_coordinates:
        x, y = coord
        radius = 3 
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 0, 0))
        label = f"{valid_projector_pixel_coordinates[i]}"
        font = ImageFont.truetype("arial.ttf", size=20) 
        draw.text((coord[0] + 5, coord[1]), label, fill=(0, 0, 0), font=font)
    image.save('binary_image6.png')
    
    return valid_projector_pixel_coordinates, avg_camera_coordinates


from scipy.spatial import distance
# Find the projector pixel pitch/distance between pixels in mm
def calculate_average_distance(pixel_centers):
    print("# pixel centers:", len(pixel_centers))
    distances = []
    for i in range(len(pixel_centers)):
        current_center = pixel_centers[i]
        min_dist = float('inf')
        for adjacent_center in pixel_centers:
            # Only proceed if the centroids are not the same.. 
            if current_center != adjacent_center:
                dist = distance.euclidean(current_center, adjacent_center)
                #print(centroid1, centroid2)
                # collect only the minimum distance between two centroids- this will be the distance between adjacent pixel centers
                if dist < min_dist:
                    min_dist = dist
                    #print(min_dist)
        distances.append(min_dist)
    # Find the average of all minimum distances- gives a value in camera pixels
    avg = int(sum(distances) / len(distances))

    print("size of proj pixel in units of camera pixels:", avg)
    # get distance of projector pixel in mm using real camera sensor pixel pitch- 0.00345mm
    pixel_distance = avg*0.00345

    return pixel_distance

# unused function- for testing
def upsample_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust the extensions as needed
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # Upsample the image
            img_upsampled = cv2.resize(img, ((img.shape[1]*3), (img.shape[0]*3)))

            # Write the upsampled image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img_upsampled)

            print(f"{filename} upsampled and saved to {output_path}")

@dataclass
# Data class for storing camera number, image data
class ImageData:
    cam_number: int
    patterns: Dict[str, List[str]] = field(default_factory=dict)
    rows: int = 0  # Default value set to None instead?
    columns: int = 0  
    center_pixel: tuple = None
    pixel_width: int = 0

    def add_pattern(self, pattern_type: str, image_data: str):
        if pattern_type not in self.patterns:
            self.patterns[pattern_type] = []
        self.patterns[pattern_type].append(image_data)

    def num_patterns(self) -> int:
        return len(self.patterns)

    def num_elements_in_pattern(self, pattern_type: str) -> int:
        if pattern_type in self.patterns:
            return len(self.patterns[pattern_type])
        else:
            return 0
    
    # Decodes gray code patterns to produce a coordinate matrix, and uses this matrix to determine the projector pixel corresponding to the center of the camera sensor,
    # and the average distance between projector pixels in mm
    def calibrate(self):
        x_pix_values = self.patterns.get('x', [])
        y_pix_values = self.patterns.get('y', [])
        
        x_graycodes = to_graycode(x_pix_values)
        y_graycodes = to_graycode(y_pix_values)

        x_coordinates = to_coordinates(x_graycodes)
        y_coordinates = to_coordinates(y_graycodes)

        matrix = coordinate_map(x_coordinates, y_coordinates, self.columns, self.rows)

        # Set center element
        #self.center_element = get_center_element(matrix)

        center, distance = process_matrix(matrix)

        self.center_pixel = center
        self.pixel_width = distance

        return matrix
    
# def gray_to_binary(gray):
#     num_bits = gray.bit_length()
#     # Shift the MSB to the rightmost position
#     msb = gray >> (num_bits - 1)
#     binary = msb
#     print("msb:", msb)
#     for i in range (num_bits - 1):
#         gray >>= i
#         print(gray)
#         #gray >>= 1
#         binary ^= (gray & 1)
#         print(binary, (gray & 1))
#         #binary <<= 1
#     return binary 


if __name__ == "__main__":
    def gray_to_binary(gray_code):
        binary_code = gray_code
        while gray_code >> 1:
            gray_code >>= 1
            binary_code ^= gray_code
        return binary_code

    # Example usage:
    gray_code = 6
    binary_number = gray_to_binary(gray_code)
    print(f"Gray code: {gray_code}, Binary number: {binary_number}, Binary number in binary: {format(binary_number, 'b')}")
    
    crop_to_window(r"c:\Acquisition_Data\Gray_code_patterns", r"C:\Acquisition_Data\cropped_testImages", 5, 5, 500, 500)

    #upsample_images(r"c:\Acquisition_Data\Multiple_camera_images", r"C:\Acquisition_Data\upsampled_testImages")
    start_time = time.time()
    x, y, = decode(r"C:\Acquisition_Data\cropped_testImages")
    print("time elapsed:", time.time() - start_time)
    print("addresses",'\n', x, '\n\n', y)

    if 0: # for testing
        cameras = create_ImageData_instance(r"C:\Acquisition_Data\cropped_testImages")
        # print("PIXEL VALS IMAGEDATA CLASS", cameras[2].patterns['x'])
        # print("dimensions imagedata class:", cameras[2].rows, cameras[2].columns,  end = '\n\n')
        #matrix = cameras[2].create_coordinate_matrix()
        #print("IMAGEDATA MATRIX",matrix)
        # print(cameras[2].center_element)
        # print(cameras.items())

        center_elements = []
        matrix = None
        for cam_number, item in cameras.items():
            matrix = item.calibrate()
            center_elements.append(item.center_pixel)
            print(cam_number, item.center_pixel)
        # binary_matrix = binarize_matrix(matrix)
        # print(binary_matrix)
        # w, b = calculate_centroid(binary_matrix)
        # print(w, b)
        # #print(calculate_distances(w))
        # print(calculate_average_distance(w, b))

        # matrix, centers = binarize_matrix(matrix)
        # print("avg dist (mm):", calculate_average_distance(centers))
        #print(calculate_average_pixelwidth(matrix))

        # dimensions = (1920, 1080)
        # generate_crosshair_pattern(dimensions, "C:\Acquisition_Data\crosshairs", center_elements)

    if 0: # for testing
        filelist = 'C:\Acquisition_Data\Gray_code_testImages'
        file_list = sorted(os.listdir(filelist), key=alphanumeric_sort)
        for i,file in enumerate(file_list):
            #print("file", i, file_list[i])
            
            shutil.copy(os.path.join(filelist, file_list[i]), r"C:\Acquisition_Data\Multiple_camera_images")
            renamed = os.rename(os.path.join(r"C:\Acquisition_Data\Multiple_camera_images",file_list[i]), os.path.join(r"C:\Acquisition_Data\Multiple_camera_images", "Cam_2_" + file_list[i]))
            print(renamed)



    while False: # for testing

        rows, columns, x_pix_values = process_im(r"C:\Acquisition_Data\Gray_code_testImages")
        print("dimensions process im:", rows, columns)
        
        y_pix_values = []
        for i in range(len(x_pix_values)): 
            #first 11 images (horizontal position)
            if i>10: 
                #last 11 images (vertical position)
                y_pix_values.append(x_pix_values.pop())
        y_pix_values = y_pix_values[::-1]

        print("PIXEL VALS PROCESS_IM", x_pix_values, end ='\n\n')
        
        # x_pix_values = image_data[1].patterns['x']
        # y_pix_values = image_data[1].patterns['y']

        x_graycodes = to_graycode(x_pix_values)
        print("GRAY CODES PROCESS_IM",x_graycodes, end ='\n\n')
        y_graycodes = to_graycode(y_pix_values)

        x_coordinates = to_coordinates(x_graycodes)
        #print(x_coordinates)
        y_coordinates = to_coordinates(y_graycodes)

        matrix = coordinate_map(x_coordinates, y_coordinates, columns, rows)

        print((matrix), end ='\n\n')
        print(np.array(matrix), end ='\n\n')
        print(get_center_element(matrix))
        

        