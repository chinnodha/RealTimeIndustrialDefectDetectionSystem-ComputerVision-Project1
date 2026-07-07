import os
import xml.etree.ElementTree as ET

# Change these paths if needed
XML_DIR = r"C:\Users\ADMIN\Downloads\archive (7)\NEU-DET\train/annotations"
LABEL_DIR = "dataset/labels/train"

XML_DIR = r"C:\Users\ADMIN\Downloads\archive (7)\NEU-DET\validation/annotations"
LABEL_DIR = r"dataset/labels/validation"

os.makedirs(LABEL_DIR, exist_ok=True)

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

for xml_file in os.listdir(XML_DIR):

    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    txt_name = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(LABEL_DIR, txt_name)

    with open(txt_path, "w") as f:

        for obj in root.findall("object"):

            class_name = obj.find("name").text
            class_id = classes.index(class_name)

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            f.write(
                f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
            )

print("Conversion completed!")