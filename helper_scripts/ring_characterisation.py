from skimage.draw import line
from scipy.interpolate import interp1d
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import image 
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from skimage.draw import circle_perimeter
from skimage.draw import circle_perimeter
from scipy.optimize import curve_fit

def draw_circle_contour_on_image(image, center, radius, color='255'):
    """
    Draw the contour of a circle on oa grayscale image at the given center with the specified radius.
    The circle contour will be drawn in the specified color (default is white).
    """
    rr, cc = circle_perimeter(center[0], center[1], radius, shape=image.shape)
    image_with_circle_contour = np.copy(image)
    image_with_circle_contour[rr, cc] = color  # Set circle contour pixels to the specified color
    return image_with_circle_contour

def normalize_and_average_profiles(profiles, desired_length):
    """
    Normalize the length of all profiles to a common length and calculate the average.
    """
    normalized_profiles = []

    for profile in profiles:
        interp_func = interp1d(np.linspace(0, 1, len(profile)), profile, kind='linear')
        new_profile = interp_func(np.linspace(0, 1, desired_length))
        normalized_profiles.append(new_profile)

    return np.mean(np.array(normalized_profiles), axis=0)

def draw_full_extent_lines_on_image(image, center, angles):
    """
    Draw radial lines on the image for all angles, extending to the full extent of the image.
    """
    image_with_lines = np.copy(image)

    for angle in angles:
        # Calculate the maximum radius for each angle
        dx = max(center[1], image.shape[1] - center[1])
        dy = max(center[0], image.shape[0] - center[0])
        max_radius = int(np.sqrt(dx**2 + dy**2))

        # Calculate the end coordinates of the line
        x_end = int(center[1] + max_radius * np.cos(angle))
        y_end = int(center[0] + max_radius * np.sin(angle))

        # Ensure the coordinates are within the image boundaries
        x_end = np.clip(x_end, 0, image.shape[1] - 1)
        y_end = np.clip(y_end, 0, image.shape[0] - 1)

        # Draw the line
        rr, cc = line(center[0], center[1], y_end, x_end)
        image_with_lines[rr, cc] = 255  # Draw line in white

    return image_with_lines

def calculate_average_radial_profile(image, center, angles, max_radius, desired_length):
    """
    Calculate the average radial profile for a given set of angles.
    """
    profiles = []

    for angle in angles:
        if -np.pi / 2 <= angle <= np.pi / 2:
            max_radius = min(center[1] / np.cos(angle), (image.shape[1] - center[1]) / np.cos(angle))
        else:
            max_radius = min(center[0] / np.sin(angle), (image.shape[0] - center[0]) / np.sin(angle))

        x_end = int(center[1] + max_radius * np.cos(angle))
        y_end = int(center[0] + max_radius * np.sin(angle))

        x_end = max(0, min(image.shape[1] - 1, x_end))
        y_end = max(0, min(image.shape[0] - 1, y_end))

        line_coords = line(int(center[0]), int(center[1]), y_end, x_end)
        profile = image[line_coords]
        profiles.append(profile)

    return normalize_and_average_profiles(profiles, desired_length)

def smooth_average_profile(image_path1, image_path2=None, normalisationFluo=1):
    # Read and normalize first image
    imagePre1 = plt.imread(image_path1)
    integralImageFluo1 = imagePre1.sum()
    imageFluo1 = imagePre1 * (normalisationFluo / integralImageFluo1)
    image_array1 = np.array(imageFluo1)

    if image_path2:
        # Read and normalize second image if provided
        imagePre2 = plt.imread(image_path2)
        integralImageFluo2 = imagePre2.sum()
        imageFluo2 = imagePre2 * (normalisationFluo / integralImageFluo2)
        image_array2 = np.array(imageFluo2)
        
        # Merge the two normalized images by taking the maximum value at each position
        merged_image_array = np.maximum(image_array1, image_array2)
        merged_image_array = merged_image_array * (normalisationFluo / merged_image_array.sum())
    else:
        # If no second image is provided, proceed with the first image
        merged_image_array = image_array1

    # Continue with the rest of the function using the merged image array
    image_center = (merged_image_array.shape[0] // 2, merged_image_array.shape[1] // 2)
    max_radius = min(image_center[0], merged_image_array.shape[0] - image_center[0], 
                     image_center[1], merged_image_array.shape[1] - image_center[1])
    desired_length = merged_image_array.shape[1] // 2
    print(max_radius)
    full = 0
    left = 1
    nLines = 100
    restricted_angle_range = np.pi / 4 if not full else np.pi
    if left:
        symmetric_angles = np.concatenate((
            np.linspace(-restricted_angle_range, restricted_angle_range, nLines),
            np.linspace(-restricted_angle_range, restricted_angle_range, nLines)
        ))
    else:
        symmetric_angles = np.concatenate((
            np.linspace(np.pi-restricted_angle_range, np.pi+restricted_angle_range, nLines), 
            np.linspace(np.pi-restricted_angle_range, np.pi+restricted_angle_range, nLines)
        ))

    avg_symmetric_radial_profile = calculate_average_radial_profile(merged_image_array, image_center, symmetric_angles, max_radius, desired_length)
    smoothedAverage = gaussian_filter1d(avg_symmetric_radial_profile, sigma=10)

    #smoothedAverage = smoothedAverage * 255 / smoothedAverage.max()
    #smoothedAverage = smoothedAverage * (normalisationFluo / smoothedAverage.sum())

    show=0
    if show:
        # Example usage with the center of the image and a radius of 200 pixels
        image_with_circle = draw_circle_contour_on_image(imagePre1, (440, 660), 255, color='0')
        image_center=(440, 660)
        # Display the image with the circle
        plt.imshow(image_with_circle, cmap='gray')
        plt.title('Image with Red Circle')
        plt.axis('off')

        # Draw lines on the image for the full extent within the symmetric angle range
        image_with_symmetric_lines = draw_full_extent_lines_on_image(image_array1, image_center, symmetric_angles)

        # Display the image with symmetric extended angle radial lines
        #plt.imshow(image_with_symmetric_lines, cmap='gray')
        plt.axis('off')
        # Compute the averaged gradient of the intensity profile
        gradient = np.gradient(smoothedAverage)

        # Plot the averaged gradient
        plt.plot(gradient)
        plt.title('Averaged Gradient of the Intensity Profile')
        plt.xlabel('Distance from center')
        plt.ylabel('Gradient of intensity')
    
    return smoothedAverage

def gaussian(x, a, mu, sigma):
    return a * np.exp(- (x - mu)**2 / (2 * sigma**2))

def fit_gaussian_to_profile(profile, plot=False, label=''):
    x = np.arange(len(profile))
    initial_guess = [np.max(profile), np.argmax(profile), 50]
    popt, _ = curve_fit(gaussian, x, profile, p0=initial_guess)
    a, mu, sigma = popt
    fwhm = 2.355 * sigma

    if plot:
        fit_curve = gaussian(x, *popt)
        plt.plot(x, profile, label=f'{label} data')
        plt.plot(x, fit_curve, '--', label=f'{label} fit')
        plt.xlabel('Distance from center [pixels]')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title('Gaussian Fit')

    return mu, fwhm

    x = np.arange(len(profile))
    initial_guess = [np.max(profile), np.argmax(profile), 50, 0]  # alpha=0 for symmetric
    popt, _ = curve_fit(voigt_profile, x, profile, p0=initial_guess)
    a, mu, sigma, alpha = popt
    fwhm = 2.355 * sigma  # Approximate, true FWHM depends on alpha

    if plot:
        fit_curve = skewed_gaussian(x, *popt)
        plt.plot(x, profile, label=f'{label} data')
        plt.plot(x, fit_curve, '--', label=f'{label} skew fit')
        plt.xlabel('Distance from center [pixels]')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title('Skewed Gaussian Fit')

    return mu, fwhm, alpha

def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
            a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

def fit_double_gaussian_to_profile(profile, plot=False, label=''):
    x = np.arange(len(profile))
    
    # Initial guess: two peaks around the main peak
    max_val = np.max(profile)
    peak_idx = np.argmax(profile)
    initial_guess = [max_val/2, peak_idx - 20, 20,
                     max_val/2, peak_idx + 20, 20]

    # Fit
    popt, _ = curve_fit(double_gaussian, x, profile, p0=initial_guess)
    a1, mu1, sigma1, a2, mu2, sigma2 = popt
    fwhm1 = 2.355 * sigma1
    fwhm2 = 2.355 * sigma2

    if plot:
        fit_curve = double_gaussian(x, *popt)
        plt.plot(x, profile, label=f'{label} data')
        plt.plot(x, fit_curve, '--', label=f'{label} double fit')
        plt.xlabel('Distance from center [pixels]')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title('Double Gaussian Fit')

    return (mu1, fwhm1), (mu2, fwhm2)

def calculate_ring_area(inner_radius, outer_radius):
    """
    Calculate the area of a ring given its internal and external radii.
    The radii should be in the same units (e.g., micrometers).
    Returns the area in square units (e.g., square micrometers).
    """
    area = np.pi * (outer_radius**2 - inner_radius**2)
    return area

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))

# Set the desired font size and axis width (it needs Python >3.8.5)
matplotlib.rcParams['font.sans-serif'] = 'Palatino'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Palatino'
matplotlib.rcParams['mathtext.it'] = 'Palatino:italic'
matplotlib.rcParams['mathtext.bf'] = 'Palatino:bold'
mpl.rcParams['mathtext.fontset'] = 'stix' 
matplotlib.rcParams['pdf.fonttype'] = 42 #to have the text as editable in Illustrator
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 7
axis_width = 1
mpl.rcParams['path.simplify'] = False
colorPink=hex_to_rgb('#FD57E7')
colorOrange=hex_to_rgb('#F4D40C') 
colorGreen=hex_to_rgb('#7FF64D')
colorBlue=hex_to_rgb('#001bff')


dataFolder='C:/Users/arian/Downloads/'
#image_path1 = dataFolder + 'largeRing_FLUO3.tiff' #largeRing
image_path1 = dataFolder + 'illumination77.tiff'
#image_path1 = dataFolder + 'illumination7.tiff' #largeRing
image_path2 = dataFolder + 'illumination23.tiff'#smallRing
normalisationFluo_1 = 180
normalisationFluo_2 = 25
smoothed_average1 = smooth_average_profile(image_path1, normalisationFluo=normalisationFluo_1)
smoothed_average2 = smooth_average_profile(image_path2, normalisationFluo=normalisationFluo_2)

#merged = np.maximum(smoothed_average1,smoothed_average2)
fig, axs = plt.subplots(1, 1, figsize=(2.1,1.8), dpi=600)
#plt.figure(figsize=(4, 3), dpi=600)
# Plot the average radial profile
axs.plot(smoothed_average1* 1e06,color=colorBlue,label=r'$r_{ext} = 245\,\mu m$')
#axs.plot(smoothed_average1* 1e06, color = colorGreen,label=r'$r_{ext} = 420\,\mu m$')


mu1, fwhm1 = fit_gaussian_to_profile(smoothed_average1, plot=False)
mu2, fwhm2 = fit_gaussian_to_profile(smoothed_average2, plot=False)

print(f"Curve 1: Peak at {mu1:.2f}, FWHM = {fwhm1:.2f}")
print(f"Curve 2: Peak at {mu2:.2f}, FWHM = {fwhm2:.2f}")


area1 = calculate_ring_area (mu1-fwhm1,mu1+fwhm1)
print(area1/normalisationFluo_1)
area2 = calculate_ring_area (mu2-fwhm2,mu2+fwhm2)
print(area2/normalisationFluo_2)
#plt.plot(smoothed_average2* 1e06)
#plt.plot(merged)
#plt.axhline(y=7.4e-6, color='r', linestyle='--')
#plt.axhline(y=9.1e-6, color='g', linestyle='--')
#plt.title('Average Symmetric Radial Profile')
axs.set_xlabel(r"distance from center $\mathrm{[\mu m]}$")
axs.set_ylabel(r'energy density $\mathrm{(W / m^2)}$')
#axs.set_ylabel(r'energy density $(W / m^2)$')
#plt.legend(frameon=False, loc='upper right')

axs.set_yscale('log')
axs.set_ylim(0, 2000)
#########################
# energy density in µmol#
#########################
microMol=1
SIunits=0
if microMol:
    #merged = np.maximum(smoothed_average1,smoothed_average2)
    fig, axs = plt.subplots(1, 1, figsize=(2.27,2.5), dpi=300)
    #plt.figure(figsize=(4, 3), dpi=600)
    # Plot the average radial profile
    k = 0.2517
    convertedIntensity = smoothed_average2 * 1e06 * k
    axs.plot(convertedIntensity, color = 'k') #,label = 'old')
    #plt.plot(smoothed_average2)
    #plt.plot(merged)
    #plt.axhline(y=7.4e-6, color='r', linestyle='--')
    #plt.axhline(y=9.1e-6, color='g', linestyle='--')
    #plt.title('Average Symmetric Radial Profile')
    axs.set_xlabel(r"distance from center $\mathrm{[\mu m]}$")
    axs.set_ylabel(r'energy density $\mathrm{(W / m^2)}$')
    #plt.legend()
    axs.set_yscale('log')
elif SIunits:
    #merged = np.maximum(smoothed_average1,smoothed_average2)
    fig, axs = plt.subplots(1, 1, figsize=(2.27,2.5), dpi=300)
    #fig, axs = plt.subplots(1, 1, figsize=(1.58,1.58), dpi=300)
    #plt.figure(figsize=(4, 3), dpi=600)
    # Plot the average radial profile
    convertedIntensity = smoothed_average1 * 1e06
    axs.plot(convertedIntensity, color = 'k') #,label = 'old')
    #line = axs.axvline(586.46, c=colorPink, ls="--", ymax=1, lw=1)
    #line.set_dashes([8, 6])
    #line = axs.axvline(np.where(convertedIntensity==convertedIntensity.max()), c='k', ls="--", ymax=1, lw=1)
    #line.set_dashes([8, 6])
    #plt.plot(smoothed_average2)
    #plt.plot(merged)
    #plt.axhline(y=7.4e-6, color='r', linestyle='--')
    #plt.axhline(y=9.1e-6, color='g', linestyle='--')
    #plt.title('Average Symmetric Radial Profile')
    axs.set_xlabel(r"distance from center $\mathrm{[\mu m]}$")
    axs.set_ylabel(r'energy density $\mathrm{(W / m^2)}$')
    #plt.legend()
    axs.set_yscale('log')
    axs.set_ylim([0, 2000])

zeroIntensity=0
if zeroIntensity:
    index_zero = np.where(smoothed_average2 < 1e-6)[0]
    # Check if zero is found and print the first index
    if index_zero.size > 0:
        first_index_zero = index_zero[0]
        print(f"The first index where the value hits 0 is: {first_index_zero}")
    else:
        print("The value 0 is not found in the array.")

    if first_index_zero is not None:
        axs.text(0.5, 0.5, f'zero intensity: {int(first_index_zero/0.38)} $\mu m$', 
                transform=axs.transAxes, 
                fontsize=12, 
                verticalalignment='center', 
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
    plt.axvline(x=first_index_zero, color='k', linestyle='--', label=r'$\approx 1300 \mu m$')

tailShow=0
if tailShow:
    axs.set_xlim([400, 700])
    #axs.set_ylim([0, 0.00003])
    axs.set_ylim([0, 0.00001])

saving=0
if saving:
    plt.tight_layout()
    fig.savefig("..//figs//intensityProfileScheme_log.pdf", format='pdf',bbox_inches="tight", transparent=True)

txtWriting=0
if txtWriting:
    title = "intensityVsPixelDistance"

    with open("..//inputSimulations//SI_radialEnergyDensityMap_610µW.txt", "w") as f:
        # Write the title
        f.write(title + "\n")
        
        #merged_normalised=convertedIntensity
        merged_normalised=convertedIntensity
        # Write each value from merged_normalised on a new line
        for value in merged_normalised:
            f.write(str(value) + "\n")