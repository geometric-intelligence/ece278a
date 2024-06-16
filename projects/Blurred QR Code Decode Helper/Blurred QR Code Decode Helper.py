import qrcode
import os
import shutil
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import zxing

website_url = "https://www.google.com"
file_name = "QR_code.png"
file_path = os.path.join(os.getcwd(), file_name)

if os.path.exists(file_path):
    os.remove(file_path)

qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
qr.add_data(website_url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

img.save(file_path)
print(f"QR code saved as {file_name} successfully.")

qr_image = Image.open(file_path)

qr_image_cropped = ImageOps.crop(qr_image, border=40)

cropped_file_name = "Cropped_QR_code.png"
cropped_file_path = os.path.join(os.getcwd(), cropped_file_name)
qr_image_cropped.save(cropped_file_path)
print(f"Cropped QR code saved as {cropped_file_name} successfully.")

test_case_dir = os.path.join(os.getcwd(), "test_case")
if os.path.exists(test_case_dir):
    shutil.rmtree(test_case_dir)
os.makedirs(test_case_dir)
print(f"Test Case directory created successfully.")

reader = zxing.BarCodeReader()

blur_levels = range(11)
failed_blur_level = None
successful_blur_level = None
summary_table = []

for blur_level in blur_levels:
    blurred_image = qr_image_cropped.filter(ImageFilter.BoxBlur(blur_level)).convert('RGB')
    blurred_file_name = f"Blurred_QR_code_{blur_level}.png"
    blurred_file_path = os.path.join(test_case_dir, blurred_file_name)
    blurred_image.save(blurred_file_path)
    
    barcode = reader.decode(blurred_file_path)
    if barcode and barcode.parsed == website_url:
        successful_blur_level = blur_level
        summary_table.append((blur_level, 1, "BoxBlur"))
    else:
        if failed_blur_level is None:
            failed_blur_level = blur_level
        summary_table.append((blur_level, 0, "BoxBlur"))

if failed_blur_level is not None:
    print(f"Failed to decode at blur level {failed_blur_level}.")
    summary_table.append(("Failed Level", failed_blur_level, None))
    
    start_level = failed_blur_level
    end_level = min(failed_blur_level + 5, 11)
    
    for blur_level in range(start_level, end_level):
        attempts = 0
        while attempts < 10:
            blurred_image = qr_image_cropped.filter(ImageFilter.BoxBlur(blur_level)).convert('RGB')
            sharpened_image = blurred_image.filter(ImageFilter.UnsharpMask(radius=3 + attempts, percent=200, threshold=2))
            
            sharpened_file_name = f"Sharpened_QR_code_{blur_level}_attempt_{attempts}.png"
            sharpened_file_path = os.path.join(test_case_dir, sharpened_file_name)
            sharpened_image.save(sharpened_file_path)
            
            barcode = reader.decode(sharpened_file_path)
            if barcode and barcode.parsed == website_url:
                summary_table.append((blur_level, attempts+1, "UnsharpMask"))
                break
            else:
                attempts += 1

        attempts = 0
        while attempts < 10:
            contrast_enhancer = ImageEnhance.Contrast(blurred_image)
            contrast_factor = 3 + attempts * 1
            contrast_enhanced_image = contrast_enhancer.enhance(contrast_factor)

            sharpened_file_name = f"Contrast_QR_code_{blur_level}_attempt_{attempts}.png"
            sharpened_file_path = os.path.join(test_case_dir, sharpened_file_name)
            contrast_enhanced_image.save(sharpened_file_path)

            barcode = reader.decode(sharpened_file_path)
            if barcode and barcode.parsed == website_url:
                summary_table.append((blur_level, attempts+1, "ContrastEnhance"))
                break
            else:
                attempts += 1

print("Testing completed.")

print(summary_table)