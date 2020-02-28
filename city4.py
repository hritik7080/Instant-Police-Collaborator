import face_recognition
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from datetime import datetime
import warnings
import os
import sqlite3
import boto3
import requests
import io
from keras.models import load_model
from datetime import date
from PIL import Image
import cv2
from class_CNN import NeuralNetwork
from class_PlateDetection import PlateDetector

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

warnings.filterwarnings("ignore")
plateDetector = PlateDetector(type_of_plate='RECT_PLATE',
                              minPlateArea=4500,
                              maxPlateArea=30000)

access_key_id = 'USE YOUR ID'
secret_access_key = 'USE YOUR KEY'
client = boto3.client('rekognition', region_name='ap-south-1', aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_access_key)

# Initialize the Neural Network
myNetwork = NeuralNetwork(modelFile="model/binary_128_0.50_ver3.pb",
                          labelFile="model/binary_128_0.50_labels_ver2.txt")

conn = sqlite3.connect("sih.db")
model = load_model('./model/model.h5')
cordinates = {"latitude": '28.6504', "longitude": "77.2372", 'region': 'Pitam Pura', 'city': 'Delhi'}


def sus_loc(name, img):
    if name != 'Unknown':

        path = f"./database/suspect/{cordinates['city']}.{name}.{date.today()}"
        cur = conn.cursor()
        cur = cur.execute('Select * from sus_loc where name=?', [name])
        fetch = cur.fetchall()
        if fetch:

            curr = conn.cursor()
            curr = conn.execute("Select * from sus_loc where latitude=? and longitude=? and name=?",
                                [cordinates['latitude'], cordinates['longitude'], name])
            fetche = curr.fetchall()
            if not fetche:

                conn.execute("Insert into sus_loc values (?,?,?,?,?,?)",
                             [name, cordinates['latitude'], cordinates['longitude'], cordinates['region'],
                              cordinates['city'],
                              date.today()])
                conn.commit()
                if os.path.exists(path):
                    cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", img)
                else:
                    os.mkdir(path)
                    cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", img)



        else:
            conn.execute("Insert into sus_loc values (?,?,?,?,?,?)",
                         [name, cordinates['latitude'], cordinates['longitude'], cordinates['region'],
                          cordinates['city'], date.today()])
            conn.commit()
            if os.path.exists(path):
                cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", img)
            else:
                os.mkdir(path)
                cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", img)


def make_face_encodings(suspects):
    known_face_encodings = []
    known_face_names = []
    for i in suspects:
        image = face_recognition.load_image_file(f'./suspects/{i}')
        image_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(image_encoding)
        known_face_names.append(i)
    return known_face_encodings, known_face_names


def find_car(plates, img):
    possible_plates = plateDetector.find_possible_plates(img)
    if possible_plates is not None:
        for i, p in enumerate(possible_plates):
            chars_on_plate = plateDetector.char_on_plate[i]
            recognized_plate, _ = myNetwork.label_image_list(chars_on_plate, imageSizeOuput=128)
            if recognized_plate in plates:
                print(recognized_plate)
                cordinates = {"latitude": '28.6504', "longitude": "77.2372", 'region': 'Pitam Pura', 'city': 'Delhi'}
                path = f"./database/Plates/{cordinates['city']}.{recognized_plate}.{date.today()}"
                cur = conn.cursor()
                cur = cur.execute('Select * from car_loc where num=?', [recognized_plate])
                fetch = cur.fetchall()
                if fetch:

                    curr = conn.cursor()
                    curr = conn.execute("Select * from car_loc where latitude=? and longitude=? and num=?",
                                        [cordinates['latitude'], cordinates['longitude'], recognized_plate])
                    fetche = curr.fetchall()
                    if not fetche:

                        conn.execute("Insert into car_loc values (?,?,?,?,?,?)",
                                     [cordinates['latitude'], cordinates['longitude'], cordinates['region'],
                                      cordinates['city'], date.today(), recognized_plate])
                        conn.commit()
                        if os.path.exists(path):
                            cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", p)
                        else:
                            os.mkdir(path)
                            cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", p)


                else:
                    conn.execute("Insert into car_loc values (?,?,?,?,?,?)",
                                 [cordinates['latitude'], cordinates['longitude'], cordinates['region'],
                                  cordinates['city'], date.today(), recognized_plate])
                    conn.commit()
                    if os.path.exists(path):
                        cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", p)
                    else:
                        os.mkdir(path)
                        cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.png", p)


def detect_violence(frame, i):
    frame1=frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    pil_img = Image.fromarray(frame)  # convert opencv frame (with type()==numpy) into PIL Image
    stream = io.BytesIO()
    pil_img.save(stream, format='JPEG')  # convert PIL Image to Bytes
    bin_img = stream.getvalue()
    # print(type(i))
    if (i % 40 == 0):
        text = "No disturbing content"
    if (i % 5 == 0):
        response = client.detect_moderation_labels(
            Image={
                'Bytes': bin_img
            },
            MinConfidence=70,
        )
        responses = response['ModerationLabels']
        res = ['Violence', 'Physical Violence', 'Weapon Violence', 'Suggestive']
        for item in responses:
            # cv2.imwrite('videos/images/frame' + str(i) + ' ' + item['Name'] + '.jpg', frame)
            # print(item['Name'])
            text = item['Name']
            print(text)
            if text in res:

                now = datetime.now()
                current_time = now.strftime("%H:%M")
                path = f"./database/violence/{cordinates['city']}.{date.today()}"
                cur = conn.cursor()
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                cur = cur.execute('Select * from locations where date=? and time=?', [date.today(), current_time])
                fetch = cur.fetchall()
                if fetch:
                    # print("idahra aya")
                    now = datetime.now()
                    current_time = now.strftime("%H:%M")
                    curr = conn.cursor()
                    curr = conn.execute(
                        "Select * from locations where latitude=? and longitude=? and date=? and time=?",
                        [cordinates['latitude'], cordinates['longitude'], date.today(), current_time])
                    fetche = curr.fetchall()
                    if not fetche:
                        # print("andar")
                        now = datetime.now()
                        current_time = now.strftime("%H:%M")
                        conn.execute("Insert into locations values (?,?,?,?,?,?)",
                                     [cordinates['latitude'], cordinates['longitude'], cordinates['region'],
                                      cordinates['city'], date.today(), current_time])
                        conn.commit()
                        # print("agau")
                        if os.path.exists(path):
                            # print("bina aye")
                            cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.{current_time}.png", frame1)
                        else:
                            # print("banake")
                            os.mkdir(path)
                            cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.{current_time}.png", frame1)


                else:
                    print("iske")
                    now = datetime.now()
                    path = f"./database/violence/{cordinates['city']}.{date.today()}"

                    current_time = now.strftime("%H:%M")
                    conn.execute("Insert into locations values (?,?,?,?,?,?)",
                                 [cordinates['latitude'], cordinates['longitude'], cordinates['region'],
                                  cordinates['city'], date.today(), current_time])
                    conn.commit()
                    if os.path.exists(path):
                        print('mial')
                        # cv2.imwrite(f"{path}/{str(date.today())}.{cordinates['latitude']}.{current_time}.png", frame1)
                        # cv2.imshow('frame1',frame1)
                        # cv2.waitKey(0)
                        # cv2.imwrite(path+'/'+str(date.today())+'.'+cordinates['latitude']+'.'+current_time+'.png', frame1)
                        cv2.imwrite(f"{path}/{cordinates['latitude']}.{i}.png", frame1)
                    else:
                        print('bana')
                        os.mkdir(path)
                        # cv2.imwrite(f"{path}/{date.today()}.{cordinates['latitude']}.{current_time}.png"+'.png', frame1)
                        cv2.imwrite(f"{path}/{cordinates['latitude']}.{i}.png", frame1)


# video_capture = cv2.VideoCapture(r'E:\SIH\test_videos\test.MOV')
video_capture = cv2.VideoCapture('./4.mp4')

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
newlength = 0
known_face_encodings = []
known_face_names = []
new_plate_len = 0
jkl = 0
while True:
    suspects = list(os.walk(r'./suspects'))[0][2]
    num = len(suspects)

    # print(num, newlength)
    if num != newlength:
        newlength = num
        # if newlength != 0:

        known_face_encodings, known_face_names = make_face_encodings(suspects)

    ret, frame = video_capture.read()
    if ret == True:

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        if list(os.walk('./suspects'))[0][2] != []:

            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                sus_loc(name, frame[top:top + bottom, left:left + right])

        # Display the resulting image
        cur = conn.cursor()
        num_list = []
        cur = cur.execute("Select * from plates")
        l = cur.fetchall()
        for i in l:
            num_list.append(i[0])
        count_plates = len(num_list)

        if count_plates > 0:
            find_car(num_list, frame)

        # detect_violence(model, frame)
        detect_violence(frame, jkl)
        jkl += 1

        cv2.imshow('Video', frame)
        #
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

