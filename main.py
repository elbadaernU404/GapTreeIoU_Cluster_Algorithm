# -*- coding: UTF-8 -*-
"""
间隙树交并比聚类算法
Gap Tree IoU Clustering Algorithm
GTIoUC
"""

import os
import ast
import cv2
import copy
import math
import base64
import requests
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageFilter

from gap_tree_algorithm.gap_tree import GapTree
from gap_tree_algorithm.preprocessing import linePreprocessing

PR = os.path.dirname(__file__)
font_file = os.path.join(PR, 'font/SourceHanSansCN-Medium.otf')
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = YOLO(os.path.join(PR, "models/hg_doclaynet_yolo11x_imgsz1120.pt"))


class StructureOCR:
    def __init__(self, blocks_data):
        self.blocks_data = blocks_data
        self.bboxes = linePreprocessing(self.blocks_data)

    @staticmethod
    def get_info(tb):  # 返回信息
        b = tb["box"]
        return (b[0][0], b[0][1], b[2][0], b[2][1]), tb["text"]

    @staticmethod
    def set_end(tb, end):  # 获取预测的块尾分隔符
        tb["end"] = end

    def structure_ocr(self):
        for line, tb in enumerate(self.blocks_data):
            tb["bbox"] = self.bboxes[line]

        gtree = GapTree(lambda tb: tb["bbox"])
        sorted_text_blocks = gtree.sort(self.blocks_data)  # 文本块排序
        return sorted_text_blocks


class VisualizeOCR:
    def __init__(self, im, layout_boxes, texts, figures):
        self.boxes = layout_boxes
        self.texts = texts
        self.figures = figures
        if isinstance(im, str):
            self.im = Image.open(im)
            self.im = np.ascontiguousarray(np.copy(im))
            self.im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else:
            self.im = np.ascontiguousarray(np.copy(im))
        self.im = Image.fromarray(self.im)
        self.im = self.im.convert('RGBA')
        self.size = (int(self.im.size[0]), int(self.im.size[1]))

    def split_text(self, width, sentence, font):
        # 按规定宽度分组
        max_line_height, total_lines = 0, 0
        allText = []
        for sen in sentence.split('\n'):
            paragraph_content, line_height, line_count = self.get_paragraph(sen, width, font)
            max_line_height = max(line_height, max_line_height)
            total_lines += line_count
            allText.append((paragraph_content, line_count))
        line_height = max_line_height
        total_height = total_lines * line_height
        return allText, total_height, line_height

    @staticmethod
    def get_paragraph(text, width, font):
        txt = Image.new('RGBA', (1920, 1080), (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)
        # 所有文字的段落
        paragraph_content = ""
        # 宽度总和
        sum_width = 0
        # 行数
        line_count = 0
        # 行高
        line_height = 0
        for char in text:
            _, _, w, h = draw.textbbox((0, 0), char, font=font)

            sum_width += w
            if sum_width > width:  # 超过预设宽度就修改段落 以及当前行数
                line_count += 1
                sum_width = 0
                paragraph_content += '\n'
            paragraph_content += char
            line_height = max(h, line_height)
        if not paragraph_content.endswith('\n'):
            paragraph_content += '\n'
        return paragraph_content, line_height, line_count

    def visualize_ocr(self):
        origin_image = copy.deepcopy(self.im)
        for figure in self.figures:
            cropped_image = self.im.crop((figure[0][0], figure[0][1], figure[2][0], figure[2][1]))
            # im_canvas.paste(cropped_image, (figure[0][0], figure[0][1]))
            self.im.paste(cropped_image, (figure[0][0], figure[0][1]))

        for i, text in enumerate(self.texts):
            if self.boxes is not None:
                box = self.boxes[i]
                x, y = box[0][0], box[0][1]
                width = int(box[1][0] - box[0][0])
                height = int(box[2][1] - box[1][1])
                width_p = int((box[1][0] - box[0][0]) * 0.97)
                if text == "":
                    continue

                blank_line_count = text.count('\n')
                blank_scale = blank_line_count * 2 if blank_line_count > 1 else blank_line_count

                """
                方程组，求字体最大像素
                x_num: x轴个数
                y_num: y轴个数
                text_scale: 文本像素

                (y_num - blank_scale) * x_num = len(text)
                x_num * text_scale = width_p
                y_num * text_scale = height
                ===>
                x_num = len(text) / (y_num - blank_scale)
                x_num = width_p / text_scale
                width_p / text_scale = len(text) / ((height / text_scale) - blank_scale)
                ===>
                len(text) * text_scale * text_scale + blank_scale * width_p * text_scale - width_p * height = 0
                """

                text_scale = int(quadratic(3 * len(text), 3 * blank_scale * width_p, 2 * -width_p * height))
                font = ImageFont.truetype(font_file, text_scale)
                draw = ImageDraw.Draw(self.im)
                paragraph_content, note_height, line_height = self.split_text(width_p, text, font)

                im_rectangle = self.im.crop((box[0][0], box[0][1], box[2][0], box[2][1]))
                im_rectangle = im_rectangle.filter(ImageFilter.GaussianBlur(radius=2000))
                self.im.paste(im_rectangle, (box[0][0], box[0][1]))

                for sen, line_count in paragraph_content:
                    draw.text((x, y), sen, fill=(0, 0, 0), font=font)
                    y += line_height * line_count

        # TODO: 是否需要拼接原始图片
        # im = image_join(origin_image, self.im, 'x')
        im = self.im
        im = im.convert('RGB')
        # 还原连续存储数组
        im = np.ascontiguousarray(np.copy(im))
        return im


def quadratic(a, b, c):
    n = b * b - 4 * a * c
    if n >= 0:
        x1 = (-b + math.sqrt(n)) / (2 * a)
        x2 = (-b - math.sqrt(n)) / (2 * a)
        return x1 if x1 > 0 else x2
    else:
        return False


def image_join(img1, img2, flag='y'):
    size1, size2 = img1.size, img2.size
    if flag == 'x':
        im = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
    else:
        im = Image.new("RGB", (size1[0], size2[1] + size1[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
    im.paste(img1, loc1)
    im.paste(img2, loc2)
    return im


def cv2_base64(im):
    base64_str = cv2.imencode('.jpg', im)[1].tobytes()
    base64_str = base64.b64encode(base64_str).decode('utf-8')
    return base64_str


def get_layout_img(image, api_url, lang):
    content_clses = ["Caption", "Footnote", "List-item", "Section-header", "Text", "Title"]
    figure_clses = ["Formula", "Picture", "Table"]
    det_res = model.predict(
        image,
        # iou=0.1,  # 检测框重叠置信度
        imgsz=1024,
        conf=0.5,
        device="cuda:1",
        save=False,
    )

    for res in det_res:
        boxes = res.boxes
        names = res.names
        clses = [names.get(i) for i in res.boxes.cls.tolist()]

        layout_results = [
            {'layout_cls': cls,
             'layout_box': [
                 [int(box[0]), int(box[1])],
                 [int(box[2]), int(box[1])],
                 [int(box[2]), int(box[3])],
                 [int(box[0]), int(box[3])]
             ]
             } for cls, box in zip(clses, boxes.xyxy.tolist()) if cls in content_clses]

        layout_figures = [
            {'layout_cls': cls,
             'layout_box': [
                 [int(box[0]), int(box[1])],
                 [int(box[2]), int(box[1])],
                 [int(box[2]), int(box[3])],
                 [int(box[0]), int(box[3])]
             ]
             } for cls, box in zip(clses, boxes.xyxy.tolist()) if cls in figure_clses]

        origin_image = image
        encoded = cv2_base64(origin_image)

        json_data = {
            "img_b64": encoded,
            "lang": lang
        }
        if ocr_api.startswith('http'):
            response = requests.post(api_url, json=json_data).json()
            ocr_result = response.get('data')
        else:
            with open('imgs/10.txt', 'r', encoding='utf-8') as f:
                ocr_result = ast.literal_eval(f.read())
        ocr_boxes = [line[0] for line in ocr_result[0]]
        ocr_txts = [line[1][0] for line in ocr_result[0]]
        ocr_scores = [line[1][1] for line in ocr_result[0]]

        json_data = []

        for i in range(len(ocr_result[0])):
            json_data.append({
                "box": [[int(i[0]), int(i[1])] for i in ocr_boxes[i]],
                "score": ocr_scores[i],
                "text": ocr_txts[i]
            })
        so = StructureOCR(json_data)
        blocks = so.structure_ocr()

        # OCR对所有文本块进行排序，每个文本块匹配最近板块id
        layout_blocks = []  # 聚合结果
        un_layout_blocks = []  # 缺失信息
        for line in blocks:
            iou_set = []
            for i, layout_result in enumerate(layout_results):
                ocr_box = line.get('box')
                lay_box = layout_result.get('layout_box')

                # IoU
                poly1 = Polygon(ocr_box)
                poly2 = Polygon(lay_box)
                intersection = poly1.intersection(poly2)
                iou_area = intersection.area if not intersection.is_empty else 0
                iou_set.append((i, iou_area, line.get('text')))

            iou_set = sorted(iou_set, key=lambda x: x[1])[-1]
            # 为板块添加在其内部的OCR文本块
            if iou_set[1] > 0:
                layout_blocks.append((iou_set[0], iou_set[2]))
            # 额外处理不在任何板块内的OCR文本块
            else:
                un_layout_blocks.append({"paragraph": line.get('text'), "layout_box": line.get('box')})

        # 将文本块排序结果按板块id聚合，聚合结果——板块id：段落
        grouped_layout_blocks = defaultdict(list)
        for line in layout_blocks:
            grouped_layout_blocks[line[0]].append(line[1])

        # 将板块检测框添加进聚合结果内
        layout_paragraphs = []
        layout_figures_list = [i.get('layout_box') for i in layout_figures]
        for i, layout_result in enumerate(layout_results):
            paragraph = grouped_layout_blocks.get(i)
            layout_paragraphs.append({"paragraph": "".join(paragraph), "layout_box": layout_result.get("layout_box")})
        layout_paragraphs.extend(un_layout_blocks)
        layout_paragraphs_list = [i.get('paragraph') for i in layout_paragraphs]
        layout_boxes_list = [i.get('layout_box') for i in layout_paragraphs]

        return layout_paragraphs, layout_boxes_list, layout_paragraphs_list, layout_figures_list


if __name__ == '__main__':
    # 如果存在ocr接口，添加ocr api地址。如果暂时没有ocr接口，可使用内置ocr文件，目前只做了10.png一份文档的。
    # ocr_api = 'http://192.168.9.242:7773/ai/surya/ocr'
    ocr_api = 'imgs/10.txt'
    ocr_lang = 'vi'
    img = cv2.imread('imgs/10.png')
    ocr_layout_paragraphs, \
        ocr_layout_boxes_list, \
        ocr_layout_paragraphs_list, \
        ocr_layout_figures_list = get_layout_img(img, ocr_api, ocr_lang)

    # 结果可视化
    vo = VisualizeOCR(img, ocr_layout_boxes_list, ocr_layout_paragraphs_list, ocr_layout_figures_list)
    translated_image = vo.visualize_ocr()
    cv2.imwrite('output/output.png', translated_image)
