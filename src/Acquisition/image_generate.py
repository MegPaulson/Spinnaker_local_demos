from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.metrics as met
from skimage.metrics import peak_signal_noise_ratio

def generate_image(width, height, bar_thickness):

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Define the colors for black and white bars
    colors = ["black", "white"]

    # Draw vertical bars with equal spacing
    for x in range(0, width, bar_thickness):
        draw.rectangle([x, 0, x + bar_thickness // 2, height], fill=colors[x % 2])
    
    # Save the generated image
    image.save("black_white_bars.png")
    return np.array(image)

def calculate_intensity_difference(image, bar_width):

    excluded_pixels = bar_width // 2
    sampled_pixels = image[:, excluded_pixels:-excluded_pixels]

    # Calculate the difference between maximum and minimum intensity values
    return np.max(sampled_pixels) - np.min(sampled_pixels)

def gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    # Read the image using OpenCV
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x)

    # Save the blurred image (optional)
    #cv2.imwrite("blurred_image.jpg", blurred_image)

    return blurred_image

def calculate_frequency_bins_above_threshold(image, threshold=0.1):
    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = image

    # Compute the 2D Fourier transform
    f_transform = np.fft.fft2(gray_image)

    # Shift zero frequency components to the center
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Normalize to range [0, 1]
    magnitude_spectrum /= np.max(magnitude_spectrum)

    # Count the number of frequency bins above the threshold
    bins_above_threshold = np.sum(magnitude_spectrum > threshold)

    return bins_above_threshold

if __name__ == "__main__":
    # Set the image dimensions and bar thickness
    image_width = 100
    image_height = 100
    bar_thickness = 6

    # Generate the image
    #generate_image(image_width, image_height, bar_thickness)

    current_bar_thickness = bar_thickness
    generate_image(image_width, image_height, bar_thickness)
    #generate_image(image_width, image_height, bar_thickness)

    image = cv2.imread("black_white_bars.png", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Initial Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    intensity_diff = calculate_intensity_difference(image, current_bar_thickness)

    print(intensity_diff)

    # gauss = gaussian_blur(image)
    # cv2.imshow("image", gauss)
    # cv2.waitKey(0)

    #intensity_diff, pixel_row_values = calculate_intensity_difference(gauss)


    #print(intensity_diff)

    current_image = image.copy()
    iterations = 0
    mtf = []
    width_increase_iterations = []
    freq = [] #cycles per window

    while False:
        while current_bar_thickness <= 20:

            # Apply Gaussian blur
            current_image = gaussian_blur(current_image)

            # Calculate intensity difference
            current_bar_width = current_image.shape[1] // 4
            difference = calculate_intensity_difference(current_image, current_bar_width)

            mtf.append(difference/255)
            #print("mtf:", difference/255)

            # Display the blurred image
            cv2.imshow("Blurred Image", current_image)
            cv2.waitKey(100)  # Adjust the delay as needed

            freq.append(image_width/(current_bar_thickness*2 ))

            # Break the loop if intensity difference is 0
            if difference/255 <= (0.5):
                print("mtf <", 0.1*current_bar_thickness, " reached")
                # Increase bar thickness by 2
                current_bar_thickness += 2

                # Save the iteration where the bar thickness gets increased
                width_increase_iterations.append(iterations)
                
                # Generate a new initial image with the increased bar thickness
                current_image = generate_image(image_width, image_height, current_bar_thickness)
                iterations += 1
            else:
                iterations += 1
                #print(iterations)

        print(f"Total iterations: {iterations}")

        plt.plot(range(iterations), mtf, marker='o', label='MTF')

        # Mark each iteration where a width increase occurred with a vertical dotted red line
        for inc in width_increase_iterations:
            plt.axvline(x=inc, color='red', linestyle='--', linewidth=0.8)

        plt.xlabel('Iterations')
        plt.ylabel('MTF (Difference/255)')
        plt.title('Modulation Transfer Function (MTF) over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.plot(freq, mtf, marker='o', label='MTF')
        plt.xlabel('freq')
        plt.ylabel('MTF (Difference/255)')
        plt.title('Modulation Transfer Function (MTF) over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Set the final image parameters
        final_bar_width = current_bar_thickness  # Keeping the same bar width for the final image

        # Generate the final image with the specified bar width
        final_image = generate_image(current_image.shape[1], image_height, final_bar_width)

        cv2.imshow("Final Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Calculate and plot the intensity profile of the center pixel row for the final image
        final_center_row = final_image[final_image.shape[0] // 2, :, 0]  # Assuming 3-channel image, taking the blue channel
        plt.plot(final_center_row)
        plt.xlabel('Pixel Position')
        plt.ylabel('Intensity')
        plt.title('Intensity Profile of Center Pixel Row (Final Image)')
        plt.legend()
        plt.grid(True)
        #plt.show()


        cv2.waitKey(0)
        cv2.destroyAllWindows()

 
    psnr_values = []

    current_image = image.copy()

    while False:
        while True:
            # Apply Gaussian blur
            
            blurred_image = gaussian_blur(image)

            mse = met.mean_squared_error(current_image, blurred_image)
            # Calculate PSNR
            if mse == 0:
                psnr = float('inf')
                break
            else:
                psnr = peak_signal_noise_ratio(current_image, image)
            print(psnr)
            print("mse:", mse)

            # Append PSNR to the list
            psnr_values.append(psnr)

            # Display the blurred image
            cv2.imshow("Blurred Image", blurred_image)
            cv2.waitKey(100)  # Adjust the delay as needed

            # Break the loop if PSNR drops significantly (you can set a threshold)
            if psnr < 7:  # Adjust the threshold as needed
                break

            iterations += 1
            image = blurred_image

        print(f"Total iterations: {iterations}")

        # Plot PSNR over iterations
        plt.plot(range(iterations + 1), psnr_values, marker='o', label='PSNR')
        plt.xlabel('Iterations')
        plt.ylabel('PSNR (dB)')
        plt.title('Progressive Blur Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    bins_above_threshold_values = []
    while True:
        # Apply Gaussian blur
        blurred_image = gaussian_blur(current_image)

        # Calculate the number of frequency bins above the threshold
        bins_above_threshold = calculate_frequency_bins_above_threshold(blurred_image)

        # Append the result to the list
        bins_above_threshold_values.append(bins_above_threshold)

        # Display the blurred image
        cv2.imshow("Blurred Image", blurred_image)
        cv2.waitKey(100)  # Adjust the delay as needed

        # Break the loop if a certain condition is met
        # (e.g., a specific number of iterations or a threshold in the metric)
        if bins_above_threshold <= 2:
            break

        iterations += 1
        current_image = blurred_image

    print(f"Total iterations: {iterations}")

    # Plot the number of frequency bins above the threshold over iterations
    plt.plot(range(iterations + 1), bins_above_threshold_values, marker='o', label='Frequency Bins')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Bins Above Threshold')
    plt.title('Progressive Blur Over Iterations (Frequency Domain)')
    plt.legend()
    plt.grid(True)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()