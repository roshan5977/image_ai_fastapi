from fastapi import FastAPI, UploadFile, HTTPException

from fastapi.responses import JSONResponse

# from

import cv2

import numpy as np

from skimage.metrics import structural_similarity as compare_ssim

 

app = FastAPI()

 

def calculate_ssim(image1, image2):

 

 

    # Convert images to grayscale

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

 

 

    # Calculate SSIM

    ssim = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]

 

    return ssim

 

 

 

 

@app.post("/compare_images/")

async def compare_images(files: list[UploadFile]):

        print("compare_images1",files,len(files))

        if len(files) < 2:

            raise HTTPException(status_code=400, detail="You must upload at least 2 images for comparison.")

 

        images = []

 

        for file in files:

            # Read and convert uploaded images to numpy arrays

            image_data = np.frombuffer(file.file.read(), np.uint8)

            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            images.append(image)

 

        # Initialize variables to count similar and dissimilar images

        similar_count = 0

        dissimilar_count = 0

        arr=[]

        for i in range(len(images)):

            for j in range(len(images)):

                if i != j and i not in arr:

                    ssim = calculate_ssim(images[i], images[j])

                    threshold = 0.99999976  # Adjust this threshold as needed

                    print("For image",i,"Comparing with",j,"With SSIM value =",ssim,"********************************")

                    if ssim > threshold:

                        arr.append(j)

                        similar_count += 1

           

        dissimilar_count =len(images)-(similar_count+1)

        return {"similar_count": similar_count+1, "dissimilar_count": dissimilar_count}

@app.post("/upload/")
async def upload_images(images: list[UploadFile], single_image: UploadFile):
    images_arr = []

    single_image_data = np.frombuffer(single_image.file.read(), np.uint8)

    single_img = cv2.imdecode(single_image_data, cv2.IMREAD_COLOR)

    matching_images = []
    for image in images:

         image_data = np.frombuffer(image.file.read(), np.uint8)
         img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
         ssim = calculate_ssim(img, single_img)
         print(ssim)
         threshold = 0.99999976

         if ssim > threshold:

          print(image)

          matching_images.append(image)
    return {"matching_images": matching_images}

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

# @app.post("/compare_imagess/")

# async def compare_images(files: list[UploadFile]):

 

#         if len(files) < 2:

#             raise HTTPException(status_code=400, detail="You must upload at least 2 images for comparison.")

 

#         images = []

#         for file in files:

#             image_data = np.frombuffer(file.file.read(), np.uint8)

#             image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

#             images.append(image)

 

#         results = []

 

#         for i, image in enumerate(images):

#             is_similar = False

#             for j in range(i + 1, len(images)):  # Compare to remaining images only

#                 ssim = calculate_ssim(image, images[j])

#                 threshold = 0.9  # Adjust this threshold as needed

#                 if ssim >= threshold:

#                     is_similar = True

#                     break  # If similar to one, no need to compare further

#             results.append({"image_index": i, "is_similar": is_similar})

 

#         return results
