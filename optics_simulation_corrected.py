import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Camera / optical parameters
# -----------------------------------------------------------------------------
wavelength = 532e-9
pixel_pitch = 3.2e-6
k0 = 2.0 * np.pi / wavelength

n0 = 1.0

dz = 30e-6
U0 = 1.0

d1 = 0.4
f = 0.15
d2 = 1.0 / (1.0 / f - 1.0 / d1)

pupil_r = 25.4e-3

# actual sensor size
SENSOR_WIDTH_PX = 4112
SENSOR_HEIGHT_PX = 3004
SENSOR_PIXEL_PITCH = 3.2e-6

RGB_labels = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.int32)
N_labels = np.array([1.474, 1.333], dtype=np.float64)

# -----------------------------------------------------------------------------
# Refractive index map loading
# -----------------------------------------------------------------------------
def load_refractive_index_map(filename):
    rgb = np.asarray(Image.open(filename).convert("RGB"), dtype=np.int32)
    diff = rgb[:, :, None, :] - RGB_labels[None, None, :, :]
    color_distance = np.sum(diff ** 2, axis=-1)
    material_label = np.argmin(color_distance, axis=-1)
    n_map = N_labels[material_label]
    return n_map, material_label

# -----------------------------------------------------------------------------
# Lens and propagation operators
# -----------------------------------------------------------------------------
def lens_tf(E_lens_front, dx, dy, f, lens_r):
    height, width = E_lens_front.shape

    x = (np.arange(width) - width // 2) * dx
    y = (np.arange(height) - height // 2) * dy
    radius_squared = y[:, None] ** 2 + x[None, :] ** 2

    pupil = (radius_squared <= lens_r ** 2).astype(np.float64)
    lens_phase = np.exp(-1j * k0 / (2.0 * f) * radius_squared)
    E_lens_back = E_lens_front * pupil * lens_phase
    return E_lens_back, pupil, lens_phase


def propagate_fresnel_tf(E_in, dx, dy, z):
    if z == 0:
        return E_in.copy()

    height, width = E_in.shape
    fx = np.fft.fftfreq(width, d=dx)
    fy = np.fft.fftfreq(height, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    H_tf = np.exp(1j * k0 * z) * np.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2))

    E_spectrum = np.fft.fft2(E_in)
    E_spectrum_out = E_spectrum * H_tf
    E_out = np.fft.ifft2(E_spectrum_out)
    return E_out

# -----------------------------------------------------------------------------
# Full-field placement on sensor without cropping the object image
# -----------------------------------------------------------------------------
def crop_or_pad_center(E_field, target_h, target_w):
    """Place the full field at the center of a target canvas.
    If the field is larger than the target canvas, crop the center.
    If smaller, pad with zeros around the edges.
    """
    field_h, field_w = E_field.shape

    # if larger, crop central region
    if field_h > target_h or field_w > target_w:
        y_start = max(0, (field_h - target_h) // 2)
        x_start = max(0, (field_w - target_w) // 2)
        y_end = y_start + min(target_h, field_h)
        x_end = x_start + min(target_w, field_w)

        cropped = E_field[y_start:y_end, x_start:x_end]
        out_h, out_w = cropped.shape

        y0 = (target_h - out_h) // 2
        x0 = (target_w - out_w) // 2
        out = np.zeros((target_h, target_w), dtype=np.complex128)
        out[y0:y0 + out_h, x0:x0 + out_w] = cropped
        return out

    # if smaller, center on canvas
    y0 = (target_h - field_h) // 2
    x0 = (target_w - field_w) // 2
    out = np.zeros((target_h, target_w), dtype=np.complex128)
    out[y0:y0 + field_h, x0:x0 + field_w] = E_field
    return out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    n_map, material_label = load_refractive_index_map("micro 2D.tif")
    print("Input object size:", n_map.shape)

    U_in = np.full(n_map.shape, U0, dtype=np.complex128)

    # object phase
    phase = k0 * (n_map - n0) * dz
    U_object = U_in * np.exp(1j * phase)

    # propagate to lens
    U_lens_front = propagate_fresnel_tf(U_object, pixel_pitch, pixel_pitch, d1)

    # apply lens
    U_lens_back, pupil, lens_phase = lens_tf(U_lens_front, pixel_pitch, pixel_pitch, f, pupil_r)

    # propagate to sensor plane
    U_detector = propagate_fresnel_tf(U_lens_back, pixel_pitch, pixel_pitch, d2)

    # preserve full projected image on sensor by centering the field instead of direct crop
    U_on_sensor = crop_or_pad_center(U_detector, SENSOR_HEIGHT_PX, SENSOR_WIDTH_PX)

    I_detector = np.abs(U_on_sensor) ** 2
    max_val = I_detector.max()
    I_normalized = I_detector / max_val if max_val > 0 else np.zeros_like(I_detector)

    image = np.rint(255.0 * I_normalized).astype(np.uint8)
    print("Output image size:", image.shape)
    Image.fromarray(image).save("synthetic_image.png")
    print("Saved synthetic_image.png")


if __name__ == "__main__":
    main()
