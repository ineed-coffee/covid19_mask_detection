import numpy as np
from PIL import Image, ImageFile
import os

class FacialMask:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin', 'top_lip', 'nose_tip', 'bottom_lip')

    def __init__(self, img_path, flat_mask_path,curved_mask_path, show=False, model='hog'):
        self.img_path = img_path
        self.flat_mask_path = flat_mask_path
        self.curved_mask_path = curved_mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        import face_recognition

        files = os.listdir(self.img_path)
        images = [file for file in files if (file.endswith(".png")) or (file.endswith(".jpg"))]
        self.flat_mask_img = Image.open(self.flat_mask_path)
        self.curved_mask_img = Image.open(self.curved_mask_path)
        
        self.chin_path = os.path.join("/".join(self.img_path.split("/")[:-1]),"chin_mask")
        os.mkdir(self.chin_path)
        self.nose_path = os.path.join("/".join(self.img_path.split("/")[:-1]),"nose_mask")
        os.mkdir(self.nose_path)
        self.full_path = os.path.join("/".join(self.img_path.split("/")[:-1]),"full_mask")
        os.mkdir(self.full_path)
        types_=["full_mask","nose_mask","chin_mask"]

        
        for type_ in types_:
            cnt=0
            for image in images:
                face_image_np = face_recognition.load_image_file(os.path.join(self.img_path,image))
                face_locations = face_recognition.face_locations(face_image_np, model=self.model)
                face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
                self._face_img = Image.fromarray(face_image_np)
                
                found_face = False
                for face_landmark in face_landmarks:
                    skip = False # check whether facial features meet requirement
                    for facial_feature in self.KEY_FACIAL_FEATURES:
                        if facial_feature not in face_landmark:
                            skip = True
                            break
                    if skip:
                        continue

                    # mask face
                    found_face = True
                    flag=self._mask_face(face_landmark,type_)

                if found_face:
                    if self.show:
                        self._face_img.show()

                    # save
                    if flag:
                        self._save(image,type_)
                        cnt+=1
            print (f"Generated {cnt} images with type: {type_}")

    def _mask_face(self, face_landmark: dict,type_):
        nose_tip = face_landmark['nose_tip']
        nose_tip_point = nose_tip[len(nose_tip) // 2]
        nose_tip_v = np.array(nose_tip_point)
        
        top_lip = face_landmark['top_lip']
        top_lip_point = top_lip[len(top_lip) // 2]
        top_lip_v = np.array(top_lip_point)
        
        bottom_lip = face_landmark['bottom_lip']
        bottom_lip_point = bottom_lip[len(bottom_lip) // 4]
        bottom_lip_v = np.array(top_lip_point)
        
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]


        # split mask and resize
        if type_!="chin_mask":
            self._mask_img = self.flat_mask_img
        else:
            self._mask_img = self.curved_mask_img
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        
        if type_=="full_mask":
            new_height = int(np.linalg.norm(nose_v - chin_bottom_v))
        elif type_=="nose_mask":    
            new_height = int(np.linalg.norm(nose_tip_v - chin_bottom_v))
        else:
            new_height = int(np.linalg.norm(bottom_lip_v - chin_bottom_v))
            
        if new_height<=0:
            return False

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        if type_=="full_mask":
            mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        elif type_=="nose_mask":    
            mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_tip_point, chin_bottom_point)
        else:
            mask_left_width = self.get_distance_from_point_to_line(chin_left_point, bottom_lip_point, chin_bottom_point)
            
        if mask_left_width<=0:
            return False
        
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        if type_=="full_mask":
            mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        elif type_=="nose_mask":
            mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_tip_point, chin_bottom_point)
        else:
            mask_right_width = self.get_distance_from_point_to_line(chin_right_point, bottom_lip_point, chin_bottom_point)
            
        if mask_right_width<=0:
            return False
        
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        if type_=="full_mask":
            angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        elif type_=="nose_mask":
            angle = np.arctan2(chin_bottom_point[1] - nose_tip_point[1], chin_bottom_point[0] - nose_tip_point[0])
        else:
            angle = np.arctan2(chin_bottom_point[1] - bottom_lip_point[1], chin_bottom_point[0] - bottom_lip_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        if type_=="full_mask":
            center_x = (nose_point[0] + chin_bottom_point[0]) // 2
            center_y = (nose_point[1] + chin_bottom_point[1]) // 2
        elif type_=="nose_mask":
            center_x = (nose_tip_point[0] + chin_bottom_point[0]) // 2
            center_y = (nose_tip_point[1] + chin_bottom_point[1]) // 2
        else:
            center_x = (bottom_lip_point[0] + chin_bottom_point[0]) // 2
            center_y = (bottom_lip_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        return True

    def _save(self,filename,suffix):
        if suffix=="full_mask":
            save_path=self.full_path
        elif suffix=="nose_mask":
            save_path=self.nose_path
        else:
            save_path=self.chin_path
        head,tail = filename.split(".")
        new_path = os.path.join(save_path,head+"_"+suffix+"."+tail)
        self._face_img.save(new_path)
        #print(f'Save to {new_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)
