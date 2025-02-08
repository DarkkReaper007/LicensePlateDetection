import cv2
import numpy as np
import easyocr

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh

def perspective_correction(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    box = sorted(box, key=lambda x: (x[1], x[0]))
    top_left, top_right = sorted(box[:2], key=lambda x: x[0])
    bottom_left, bottom_right = sorted(box[2:], key=lambda x: x[0])
    width = int(max(
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_right - top_left)
    ))
    height = int(max(
        np.linalg.norm(top_right - bottom_right),
        np.linalg.norm(top_left - bottom_left)
    ))
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped



def find_license_plate(preprocessed_image, image):
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        if 1000 < area < 50000 and 2.0 < aspect_ratio < 6.0:
            print(f"Contour {i}: Area={area}, Aspect Ratio={aspect_ratio}")


            license_plate = perspective_correction(image, contour)
            return license_plate, image

    return None, image

def refine_license_plate(license_plate):
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, refined = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return refined

def main():
    image_path = "C://Users//Shasmeet//PycharmProjects//LicensePlateOnlyOpenCv//images.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load the image.")
        return

    reader = easyocr.Reader(['en'])
    preprocessed_image = preprocess_image(image)
    cv2.imwrite("preprocessed_image.jpg", preprocessed_image)
    license_plate, annotated_image = find_license_plate(preprocessed_image, image)

    if license_plate is not None:
        cv2.imwrite("license_plate.jpg", license_plate)

        refined_plate = refine_license_plate(license_plate)
        cv2.imwrite("refined_license_plate.jpg", refined_plate)

        text = reader.readtext(refined_plate, detail=0)
        print("Detected Text:", text)

        cv2.imshow("Annotated Image", annotated_image)
        cv2.imshow("License Plate", license_plate)
    else:
        print("No license plate detected.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
