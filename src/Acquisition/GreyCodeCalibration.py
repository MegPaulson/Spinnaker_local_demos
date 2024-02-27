import cv2
import numpy as np
#import image_generate
import os
from PIL import Image, ImageDraw
import re
import shutil
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import Dict, List


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image
    _, binary_im = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    
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

    # center_row = rows // 2
    # center_col = cols // 2
    
    # return matrix[center_row][center_col]
    total_x = 0
    total_y = 0
    for row in matrix:
        for coord in row:
            # print(coord)
            x, y = coord
            total_x += x
            total_y += y

    # Calculate the average x and y values
    average_x = total_x // (cols*rows)
    average_y = total_y // (rows*rows)

    # print("totalx", total_x, "avg x", average_x)
    # print("totaly", total_y, "avg y", average_y)

    return (average_x, average_y)

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

def calculate_average_pixelwidth():
    # Want to measure center-to-center distance between pixels in order to establish global scale for metric...
    # each "blob" that represents a projector pixel should have the same x and y coordinates within camera matrix
    # binarize based on unique coordinates to distinguish blobs
    # Find centroid of each whole blob, then distance in camera pixels between them
    pass

def find_longest_sequence(matrix):
    m = len(matrix)
    n = len(matrix[0])
    max_length = 0  # Storing the longest sequence of identical coordinate pairs

    # Check horizontally and vertically
    for i in range(m):
        for j in range(n):
            current_length = 1
            for k in range(j + 1, n):
                if matrix[i][k] == matrix[i][j]:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    break

            current_length = 1
            for k in range(i + 1, m):
                if matrix[k][j] == matrix[i][j]:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    break

    return max_length

# Isolate blobs from coordinate matrix
def binarize_matrix(matrix):
    #print(matrix)
    unique_values = sorted(set(tuple(pair) for row in matrix for pair in row), key=lambda x: (x[0], x[1]))
    temp_mat = np.zeros_like(matrix, dtype=int)
    matrix = np.reshape(matrix, temp_mat.shape)
    binary_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=int)
    
    #print(matrix)
    print("unique:",unique_values)
    # print(binary_matrix)
    # print(binary_matrix.shape)
    indices = None

    i = 1
    for val in unique_values:
        # Find indices where the value occurs
        print(val)
        indices = np.where(np.all(matrix == val, axis=(2)))
        binary_matrix[indices] = i%2
        i+=1
    return binary_matrix

def locate_centroid(binary_matrix):
    im = cv2.imread(r"C:\Acquisition_Data\upsampled_testImages\Cam_2_pattern_x_3.png")
    wcentroids = []
    bcentroids = []
    binary_matrix = np.array(binary_matrix, dtype=np.uint8)
    #contours,hierarchy = cv2.findContours(binary_matrix, 1, 2)

    bcontours, _ = cv2.findContours((binary_matrix == 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    wcontours, _ = cv2.findContours((binary_matrix == 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for wcnt in wcontours:
        rect = cv2.minAreaRect(wcnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(im,[box],0,(0,0,255),1)
    for bcnt in bcontours:
        rect = cv2.minAreaRect(bcnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(im,[box],0,(0,255,0),1)
    cv2.imwrite(r"C:\Acquisition_Data\upsampled_testImages\Cam_2_pattern_x_3CNT.png", im)
    cv2.imshow('img',im)
    cv2.waitKey(0)  

    for c in wcontours:
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'] + 1e-5)
        cy = int(M['m01']/M['m00'] + 1e-5)
        wcentroids.append((cx,cy))
    for c in bcontours:
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'] + 1e-5)
        cy = int(M['m01']/M['m00'] + 1e-5)
        bcentroids.append((cx,cy))
    return [wcentroids, bcentroids]

def calculate_distances(centroids):
    distances = []
    #centerpoints = centerpoints[:-3]
    for i in range(len(centroids) - 1):
        # Calculate the Euclidean distance between centers of adjacent centroids
        distance = np.sqrt((centroids[i][0] - centroids[i+1][0]) ** 2 + (centroids[i][1] - centroids[i+1][1]) ** 2)
        distances.append(distance)
    return (distances)

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
    center_element: tuple = None

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
    
    def create_coordinate_matrix(self):
        x_pix_values = self.patterns.get('x', [])
        y_pix_values = self.patterns.get('y', [])
        
        x_graycodes = to_graycode(x_pix_values)
        y_graycodes = to_graycode(y_pix_values)

        x_coordinates = to_coordinates(x_graycodes)
        y_coordinates = to_coordinates(y_graycodes)

        matrix = coordinate_map(x_coordinates, y_coordinates, self.columns, self.rows)

        # Set center element
        self.center_element = get_center_element(matrix)

        return matrix
    


if __name__ == "__main__":
    #crop_to_window(r"C:\Acquisition_Data\Gray_code_patterns", r"C:\Acquisition_Data\Gray_code_testImages", 5, 5, 300, 300)

    #upsample_images(r"C:\Acquisition_Data\Multiple_camera_images", r"C:\Acquisition_Data\upsampled_testImages")

    if 1: # for testing
        cameras = create_ImageData_instance(r"C:\Acquisition_Data\upsampled_testImages")
        # print("PIXEL VALS IMAGEDATA CLASS", cameras[2].patterns['x'])
        # print("dimensions imagedata class:", cameras[2].rows, cameras[2].columns,  end = '\n\n')
        #matrix = cameras[2].create_coordinate_matrix()
        #print("IMAGEDATA MATRIX",matrix)
        # print(cameras[2].center_element)
        # print(cameras.items())

        center_elements = []
        matrix = None
        for cam_number, item in cameras.items():
            matrix = item.create_coordinate_matrix()
            center_elements.append(item.center_element)

        binary_matrix = binarize_matrix(matrix)
        print(binary_matrix)
        w, b = locate_centroid(binary_matrix)
        print(w, b)
        print(calculate_distances(w))

        #print(find_longest_sequence(matrix))
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
        

        