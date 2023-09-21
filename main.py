from fastapi import FastAPI, UploadFile, HTTPException

from fastapi.responses import JSONResponse

# from
import base64
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



def calculate_ssim1(image_1, image_2):
    print("IMAGE 1",image_1,"\nIMAGE 2", image_2, "-----", end="\n")
    image_1.file.seek(0)
    image_2.file.seek(0)
    image_data1 = np.frombuffer(image_1.file.read(), np.uint8)
    image1 = cv2.imdecode(image_data1, cv2.IMREAD_COLOR)
    image_1.file.seek(0)
    image_2.file.seek(0)
    image_data2 = np.frombuffer(image_2.file.read(), np.uint8)
    image2 = cv2.imdecode(image_data2, cv2.IMREAD_COLOR)
    del image_data1
    del image_data2

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    ssim = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
    return ssim




@app.post("/upload/")
async def compare_images(images: list[UploadFile]):
        if len(images) < 2:
            raise HTTPException(status_code=400, detail="You must upload at least 2 images for comparison.")
        similar_count = 0             
        arr=[]
        duplicate_images = []
        for i in range(len(images)):
            for j in range(i+1,len(images)):
                if i != j and i not in arr:
                    print(images[i],images[j],"+++++++++++++++++++++++++++++++++",i,j,end="\n")
                    ssim = calculate_ssim1(images[i],images[j])
                    threshold = 0.99999976  
                    print("For image",i,"Comparing with",j,"With SSIM value =",ssim,"********************************")
                    if ssim > threshold:
                        duplicate_images.append(images[j].filename)
                        arr.append(j)
                        similar_count += 1

        return {"similar_count": similar_count, "duplicate_images": duplicate_images}

@app.post("/compare/")
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

          print(image.filename.split(".")[1],"Image content type+++++++++++++++++++++++++++++++++++++++++")

          matching_images.append({
                "filename": image.filename,
                "image_base64": image_to_base64(img,"."+image.filename.split(".")[1])  # Convert the matching image to Base64
          })

    return {"matching_images": matching_images}

 

 
def image_to_base64(image, type):
    _, buffer = cv2.imencode(type, image)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64
 

 

 

 

 
