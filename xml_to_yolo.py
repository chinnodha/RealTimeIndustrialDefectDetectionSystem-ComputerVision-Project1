import os
import xml.etree.ElementTree as ET

# =====================================
# Train and Validation Dataset Paths
# =====================================

DATASETS = [

    (
        r"C:\Users\ADMIN\Downloads\archive (7)\NEU-DET\train\annotations",
        r"dataset\labels\train"
    ),

    (
        r"C:\Users\ADMIN\Downloads\archive (7)\NEU-DET\validation\annotations",
        r"dataset\labels\validation"
    )

]

# =====================================
# Class Names
# =====================================

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# =====================================
# XML -> YOLO Conversion
# =====================================

for XML_DIR, LABEL_DIR in DATASETS:

    os.makedirs(LABEL_DIR, exist_ok=True)

    print(f"\nProcessing : {XML_DIR}")

    for xml_file in os.listdir(XML_DIR):

        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(XML_DIR, xml_file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        txt_name = xml_file.replace(".xml", ".txt")
        txt_path = os.path.join(LABEL_DIR, txt_name)

        with open(txt_path, "w") as f:

            for obj in root.findall("object"):

                class_name = obj.find("name").text

                # Check whether class exists
                if class_name not in classes:
                    print(f"Skipping unknown class : {class_name}")
                    continue

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
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

    print(f"Finished : {LABEL_DIR}")

print("\nXML to YOLO conversion completed successfully!")