import cv2
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, diags
def estimate_initial_illumination(image):
    # Estimate the initial illumination map by taking the maximum value across the RGB channels
    return np.max(image, axis=2)



# def refine_illumination_map(T_init, alpha=1.5, sigma=1):
def refine_illumination_map(T_init, alpha=2.1, sigma=1.5):
    # Refine the illumination map using structure-aware smoothing
    T = T_init.copy()
    rows, cols = T.shape
    T_flat = T.flatten()

    # Compute gradients
    dy = np.diff(T, axis=0, append=T[-1:, :])
    dx = np.diff(T, axis=1, append=T[:, -1:])

    # Compute weights
    W_h = np.exp(-np.abs(dx) / sigma)
    W_v = np.exp(-np.abs(dy) / sigma)

    # Flatten weights
    W_h_flat = W_h.flatten()
    W_v_flat = W_v.flatten()

    # Construct sparse diagonal weight matrices
    Wh = spdiags(W_h_flat, 0, rows * cols, rows * cols)
    Wv = spdiags(W_v_flat, 0, rows * cols, rows * cols)

    # Construct difference matrices
    e = np.ones(rows * cols)
    D_h = spdiags([-e, e], [0, 1], rows * cols, rows * cols)
    D_v = spdiags([-e, e], [0, cols], rows * cols, rows * cols)

    # Regularization term
    A = D_h.T @ Wh @ D_h + D_v.T @ Wv @ D_v
    A = A + alpha * diags(np.ones(rows * cols), 0)

    # Solve linear system
    b = alpha * T_flat
    T_refined = spsolve(A, b)
    return T_refined.reshape((rows, cols))
# def enhance_image(image, T_refined, gamma=0.2):
def enhance_image(image, T_refined, gamma=0.1):

    # Enhance the image using the refined illumination map
    T_refined = np.power(T_refined, gamma)
    enhanced_image = np.zeros_like(image)
    for i in range(3):
        enhanced_image[:, :, i] = image[:, :, i] / (T_refined + 1e-6)
    return np.clip(enhanced_image, 0, 1)

def lime_enhancement(image_path):
    # Read the input image
    # image = cv2.imread(image_path)
    image = image_path
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Estimate initial illumination map
    T_init = estimate_initial_illumination(image)

    # Refine illumination map
    T_refined = refine_illumination_map(T_init)

    # Enhance the image
    enhanced_image = enhance_image(image, T_refined)

    # Convert back to BGR for OpenCV
    enhanced_image_bgr = cv2.cvtColor((enhanced_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return enhanced_image_bgr
    # Save the enhanced image
    # output_path = image_path.replace('.jpg', '_enhanced.jpg')
    # output_path = './enhanced.png'
    # cv2.imwrite(output_path, enhanced_image_bgr)
    # print(f"Enhanced image saved to {output_path}")

# Example usage
# lime_enhancement('./input_original.png')