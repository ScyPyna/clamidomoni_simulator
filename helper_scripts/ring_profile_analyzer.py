from skimage.draw import line
from scipy.interpolate import interp1d
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from skimage.draw import circle_perimeter
from scipy.optimize import curve_fit

# ── configurazione ───────────────────────────────────────────
DATA_FOLDER   = "/home/user/Scrivania/"
SMALL_RING    = DATA_FOLDER + "illumination25.tiff"
LARGE_RING    = DATA_FOLDER + "illumination24.tiff"
OUTPUT_FOLDER = "inputSimulations/"
PIX           = 1.041667   # µm/pixel

NORM_SMALL = 180
NORM_LARGE = 610
# ────────────────────────────────────────────────────────────

def draw_circle_contour_on_image(image, center, radius, color='255'):
    rr, cc = circle_perimeter(center[0], center[1], radius, shape=image.shape)
    image_with_circle_contour = np.copy(image)
    image_with_circle_contour[rr, cc] = color
    return image_with_circle_contour

def normalize_and_average_profiles(profiles, desired_length):
    normalized_profiles = []
    for profile in profiles:
        interp_func = interp1d(np.linspace(0, 1, len(profile)), profile, kind='linear')
        new_profile = interp_func(np.linspace(0, 1, desired_length))
        normalized_profiles.append(new_profile)
    return np.mean(np.array(normalized_profiles), axis=0)

def draw_full_extent_lines_on_image(image, center, angles):
    image_with_lines = np.copy(image)
    for angle in angles:
        dx = max(center[1], image.shape[1] - center[1])
        dy = max(center[0], image.shape[0] - center[0])
        max_radius = int(np.sqrt(dx**2 + dy**2))
        x_end = int(center[1] + max_radius * np.cos(angle))
        y_end = int(center[0] + max_radius * np.sin(angle))
        x_end = np.clip(x_end, 0, image.shape[1] - 1)
        y_end = np.clip(y_end, 0, image.shape[0] - 1)
        rr, cc = line(center[0], center[1], y_end, x_end)
        image_with_lines[rr, cc] = 255
    return image_with_lines

def calculate_average_radial_profile(image, center, angles, max_radius, desired_length):
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

def smooth_average_profile(image_path, normalisationFluo=1):
    imagePre = plt.imread(image_path)
    integralImageFluo = imagePre.sum()
    imageFluo = imagePre * (normalisationFluo / integralImageFluo)
    image_array = np.array(imageFluo)

    image_center = (image_array.shape[0] // 2, image_array.shape[1] // 2)
    max_radius = min(image_center[0], image_array.shape[0] - image_center[0],
                     image_center[1], image_array.shape[1] - image_center[1])
    desired_length = image_array.shape[1] // 2
    print(f"  Centro: {image_center}  |  raggio max: {max_radius}px  |  lunghezza profilo: {desired_length}")

    nLines = 100
    restricted_angle_range = np.pi / 4
    symmetric_angles = np.concatenate((
        np.linspace(-restricted_angle_range, restricted_angle_range, nLines),
        np.linspace(-restricted_angle_range, restricted_angle_range, nLines)
    ))

    avg_profile = calculate_average_radial_profile(image_array, image_center,
                                                    symmetric_angles, max_radius, desired_length)
    smoothed = gaussian_filter1d(avg_profile, sigma=10)
    return smoothed

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian_to_profile(profile):
    x = np.arange(len(profile))
    initial_guess = [np.max(profile), np.argmax(profile), 50]
    popt, _ = curve_fit(gaussian, x, profile, p0=initial_guess)
    a, mu, sigma = popt
    fwhm = 2.355 * sigma
    return mu, fwhm, popt

def calculate_ring_area(inner_radius, outer_radius):
    return np.pi * (outer_radius**2 - inner_radius**2)

def save_profile_txt(profile, filename, title="intensityVsPixelDistance"):
    with open(filename, "w") as f:
        f.write(title + "\n")
        for value in profile:
            f.write(str(value) + "\n")
    print(f"  Salvato: {filename}")

def analyze_ring(image_path, norm, label, output_suffix, pix=PIX):
    print(f"\n── {label} ──────────────────────────────────")
    smoothed = smooth_average_profile(image_path, normalisationFluo=norm)

    # fit gaussiana
    mu, fwhm, popt = fit_gaussian_to_profile(smoothed)
    mu_um   = mu   * pix
    fwhm_um = fwhm * pix
    print(f"  Picco:  {mu_um:.1f} µm  (pixel {mu:.1f})")
    print(f"  FWHM:   {fwhm_um:.1f} µm")
    print(f"  r_in:   {mu_um - fwhm_um/2:.1f} µm")
    print(f"  r_out:  {mu_um + fwhm_um/2:.1f} µm")
    area = calculate_ring_area(mu_um - fwhm_um/2, mu_um + fwhm_um/2)
    print(f"  Area anello: {area:.1f} µm²")

    # salva profilo txt per simulatore
    txt_path = OUTPUT_FOLDER + f"radialEnergyDensityMap_{output_suffix}_good.txt"
    save_profile_txt(smoothed, txt_path)

    return smoothed, mu, fwhm, popt

def plot_comparison(profiles, labels, colors, pix=PIX):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # pannello 1: profili in µm
    ax = axes[0]
    for (smoothed, mu, fwhm, popt), label, color in zip(profiles, labels, colors):
        x_px = np.arange(len(smoothed))
        x_um = x_px * pix
        ax.plot(x_um, smoothed * 1e6, color=color, label=label, lw=1.5)
        # fit gaussiana
        fit_curve = gaussian(x_px, *popt)
        ax.plot(x_um, fit_curve * 1e6, '--', color=color, alpha=0.6, lw=1)
        ax.axvline(mu * pix, color=color, ls=':', alpha=0.5)

    ax.set_xlabel("distanza dal centro (µm)")
    ax.set_ylabel("energy density (W/m²)")
    ax.set_title("Profili radiali — scala lineare")
    ax.legend()

    # pannello 2: scala log
    ax = axes[1]
    for (smoothed, mu, fwhm, popt), label, color in zip(profiles, labels, colors):
        x_um = np.arange(len(smoothed)) * pix
        ax.plot(x_um, smoothed * 1e6, color=color, label=label, lw=1.5)
    ax.set_xlabel("distanza dal centro (µm)")
    ax.set_ylabel("energy density (W/m²)")
    ax.set_title("Profili radiali — scala log")
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    out = "dataset/ring_profiles_experimental_2nd.png"
    plt.savefig(out, dpi=150)
    print(f"\nSalvato: {out}")
    plt.close()

# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # analizza entrambi gli anelli
    result_small = analyze_ring(
        image_path    = SMALL_RING,
        norm          = NORM_SMALL,
        label         = "Small ring (illumination25.tiff)",
        output_suffix = "exp_small_ring_ill25_180muW"
    )

    result_large = analyze_ring(
        image_path    = LARGE_RING,
        norm          = NORM_LARGE,
        label         = "Large ring (illumination24.tiff)",
        output_suffix = "exp_large_ring_ill24_610muW"
    )

    # plot comparativo
    plot_comparison(
        profiles = [result_small, result_large],
        labels   = ["Small ring 180µW", "Large ring 610µW"],
        colors   = ["steelblue", "darkorange"]
    )

    print("\nDone. File salvati in inputSimulations/ e dataset/")
