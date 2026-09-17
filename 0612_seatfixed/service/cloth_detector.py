import os
from ultralytics import YOLO
import torch


DEFAULT_DETECT_MODEL = os.getenv("DETECT_MODEL_PATH", "./models/cloth/detect.pt")
DEFAULT_LONG_MODEL = os.getenv("LONG_MODEL_PATH", "./models/cloth/recognition_long.pt")


class ClothDetector:


    def __init__(
        self,
        detect_model_path=DEFAULT_DETECT_MODEL,
        long_model_path=DEFAULT_LONG_MODEL
    ):


        device = "cuda" if torch.cuda.is_available() else "cpu"


        # ==========================
        # 第一阶段：三分类检测模型
        # ==========================
        self.detect_model = YOLO(
            detect_model_path
        )

        self.detect_model.to(device)



        # ==========================
        # 第二阶段：长袖细粒度分类模型
        # ==========================
        self.long_model = YOLO(
            long_model_path
        )

        self.long_model.to(device)



        self.device = device



        # 第一阶段类别
        # detect.pt训练标签
        self.stage1_names = {

            0: "短袖",

            1: "长袖",

            2: "背心"

        }



        # 第二阶段类别
        # recognition_long.pt训练标签
        self.long_names = {

            0: "西装外套",

            1: "薄夹克",

            2: "长款大衣",

            3: "羽绒服",

            4: "长袖针织毛衣",

            5: "长袖衬衫",

            6: "长袖",

            7: "连帽卫衣"

        }




    def classify_long(self, crop):

        """
        YOLO26-cls
        长袖区域细粒度分类
        """


        if crop is None:
            return None



        results = self.long_model(
            crop,
            verbose=False
        )


        if not results:
            return None



        result = results[0]



        # YOLO分类模型结果
        cls = int(
            result.probs.top1
        )


        conf = float(
            result.probs.top1conf
        )



        return {


            "label":
                self.long_names[cls],


            "confidence":
                conf

        }




    def detect(self, frame):


        h, w = frame.shape[:2]



        results = self.detect_model(
            frame,
            verbose=False
        )



        left_candidates = []

        right_candidates = []




        if results and results[0].boxes:



            for box in results[0].boxes:



                conf = float(
                    box.conf[0]
                )


                if conf < 0.5:
                    continue



                cls = int(
                    box.cls[0]
                )



                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )



                # 防止越界
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))



                crop = frame[
                    y1:y2,
                    x1:x2
                ]



                # ======================
                # 第一阶段结果
                # ======================

                label = self.stage1_names[cls]


                final_conf = conf




                # ======================
                # 长袖进入第二阶段
                # ======================

                if cls == 1:


                    long_result = self.classify_long(
                        crop
                    )



                    if long_result:


                        label = long_result["label"]



                        # 直接使用细粒度分类的置信度
                        final_conf = long_result["confidence"]




                item = {


                    "label": label,


                    "confidence": final_conf,


                    "bbox":[

                        float(x1),

                        float(y1),

                        float(x2),

                        float(y2)

                    ]

                }




                # ======================
                # 左右乘员划分
                # ======================

                center = (
                    x1+x2
                ) / 2




                if center > w/2:

                    right_candidates.append(
                        item
                    )


                else:

                    left_candidates.append(
                        item
                    )




        # ======================
        # 每侧最高置信度
        # ======================


        driver_result = (

            max(
                right_candidates,
                key=lambda x:x["confidence"]
            )

            if right_candidates

            else None

        )



        passenger_result = (

            max(
                left_candidates,
                key=lambda x:x["confidence"]
            )

            if left_candidates

            else None

        )




        return {


            "driver":
                driver_result,


            "passenger":
                passenger_result

        }