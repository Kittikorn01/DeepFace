from deepface import DeepFace
import pandas as pd
import os

# รันการค้นหา
results = DeepFace.find(img_path = "test_image2.jpg", 
                        db_path = "dataset/", 
                        model_name = "VGG-Face",
                        enforce_detection = False) # ใส่ไว้กัน Error กรณีภาพทดสอบหาหน้าไม่เจอ

# ตรวจสอบว่าเจอคนในฐานข้อมูลไหม
if len(results) > 0 and not results[0].empty:
    image_path = results[0].iloc[0]['identity']
    
    # วิธีแก้ที่ชัวร์ที่สุด: ใช้ os.path.dirname เพื่อถอยกลับไป 1 โฟลเดอร์ แล้วเอาชื่อโฟลเดอร์นั้น
    folder_path = os.path.dirname(image_path)
    person_name = os.path.basename(folder_path)
    
    print(f"คนในภาพคือ: {person_name}")
    print(f"ความแม่นยำ (Distance): {results[0].iloc[0]['distance']}")
else:
    print("ไม่พบใบหน้าที่ตรงกับในฐานข้อมูล")