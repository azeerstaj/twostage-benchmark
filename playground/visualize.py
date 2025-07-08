import numpy as np
import torch
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

COLORS = [
    "GoldenRod",
    "MediumTurquoise",
    "GreenYellow",
    "SteelBlue",
    "DarkSeaGreen",
    "SeaShell",
    "LightGrey",
    "IndianRed",
    "DarkKhaki",
    "LawnGreen",
    "WhiteSmoke",
    "Peru",
    "LightCoral",
    "FireBrick",
    "OldLace",
    "LightBlue",
    "SlateGray",
    "OliveDrab",
    "NavajoWhite",
    "PaleVioletRed",
    "SpringGreen",
    "AliceBlue",
    "Violet",
    "DeepSkyBlue",
    "Red",
    "MediumVioletRed",
    "PaleTurquoise",
    "Tomato",
    "Azure",
    "Yellow",
    "Cornsilk",
    "Aquamarine",
    "CadetBlue",
    "CornflowerBlue",
    "DodgerBlue",
    "Olive",
    "Orchid",
    "LemonChiffon",
    "Sienna",
    "OrangeRed",
    "Orange",
    "DarkSalmon",
    "Magenta",
    "Wheat",
    "Lime",
    "GhostWhite",
    "SlateBlue",
    "Aqua",
    "MediumAquaMarine",
    "LightSlateGrey",
    "MediumSeaGreen",
    "SandyBrown",
    "YellowGreen",
    "Plum",
    "FloralWhite",
    "LightPink",
    "Thistle",
    "DarkViolet",
    "Pink",
    "Crimson",
    "Chocolate",
    "DarkGrey",
    "Ivory",
    "PaleGreen",
    "DarkGoldenRod",
    "LavenderBlush",
    "SlateGrey",
    "DeepPink",
    "Gold",
    "Cyan",
    "LightSteelBlue",
    "MediumPurple",
    "ForestGreen",
    "DarkOrange",
    "Tan",
    "Salmon",
    "PaleGoldenRod",
    "LightGreen",
    "LightSlateGray",
    "HoneyDew",
    "Fuchsia",
    "LightSeaGreen",
    "DarkOrchid",
    "Green",
    "Chartreuse",
    "LimeGreen",
    "AntiqueWhite",
    "Beige",
    "Gainsboro",
    "Bisque",
    "SaddleBrown",
    "Silver",
    "Lavender",
    "Teal",
    "LightCyan",
    "PapayaWhip",
    "Purple",
    "Coral",
    "BurlyWood",
    "LightGray",
    "Snow",
    "MistyRose",
    "PowderBlue",
    "DarkCyan",
    "White",
    "Turquoise",
    "MediumSlateBlue",
    "PeachPuff",
    "Moccasin",
    "LightSalmon",
    "SkyBlue",
    "Khaki",
    "MediumSpringGreen",
    "BlueViolet",
    "MintCream",
    "Linen",
    "SeaGreen",
    "HotPink",
    "LightYellow",
    "BlanchedAlmond",
    "RoyalBlue",
    "RosyBrown",
    "MediumOrchid",
    "DarkTurquoise",
    "LightGoldenRodYellow",
    "LightSkyBlue",
]


def visualize_detections(image_path, output_path, detections, labels=[]):
    image = Image.open(image_path).convert(mode="RGB")
    draw = ImageDraw.Draw(image)
    line_width = 2
    font = ImageFont.load_default()
    for d in detections:
        # color = COLORS[d["class"] % len(COLORS)]
        color = COLORS[0]
        draw.line(
            [
                (d["xmin"], d["ymin"]),
                (d["xmin"], d["ymax"]),
                (d["xmax"], d["ymax"]),
                (d["xmax"], d["ymin"]),
                (d["xmin"], d["ymin"]),
            ],
            width=line_width,
            fill=color,
        )
        label = f"Class {d['class']}"
        if d["class"] < len(labels):
            label = f"{labels[d['class']]}"
        score = d["score"]
        text = f"{label}: {int(100*score)}%"
        if score < 0:
            text = label
        left, top, right, bottom = font.getbbox(text)
        text_width, text_height = right - left, bottom - top
        text_bottom = max(text_height, d["ymin"])
        text_left = d["xmin"]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (text_left, text_bottom - text_height - 2 * margin),
                (text_left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (text_left + margin, text_bottom - text_height - margin),
            text,
            fill="black",
            font=font,
        )
    if output_path is None:
        return image
    image.save(output_path)

def visualize_boxes(image_path, output_path, detections, labels=[], confidence_threshold=0.5):
    """
    Visualize detected bounding boxes on an image and save the result.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path where the output image will be saved
        detections (list): List of detection dictionaries containing 'boxes', 'labels', 'scores'
        labels (list): Optional list of class names corresponding to label indices
        confidence_threshold (float): Minimum confidence score to display a box
    """
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Process detections (assuming detections is a list with one dict)
    detection = detections[0] if isinstance(detections, list) else detections
    
    boxes = detection['boxes']
    class_labels = detection['labels']
    scores = detection['scores']
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    if torch.is_tensor(class_labels):
        class_labels = class_labels.detach().cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    
    # Color palette for different classes
    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080',
        '#008000', '#800000', '#808000', '#C0C0C0', '#FF69B4', '#DDA0DD',
        '#98FB98', '#F0E68C', '#DEB887', '#D2691E', '#FF7F50', '#6495ED',
        '#DC143C', '#00CED1', '#FF1493', '#00BFFF', '#696969', '#1E90FF'
    ]
    
    # Draw boxes
    for i, (box, label, score) in enumerate(zip(boxes, class_labels, scores)):
        # Skip boxes below confidence threshold
        if score < confidence_threshold:
            continue
            
        # Extract coordinates
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Choose color based on class label
        color = colors[int(label) % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Prepare label text
        if labels and int(label) < len(labels):
            class_name = labels[int(label)]
            text = f"{class_name}: {score:.2f}"
        else:
            text = f"Class {int(label)}: {score:.2f}"
        
        # Calculate text size and position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position text above the box, or inside if there's no space above
        text_x = x1
        text_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + 2
        
        # Draw text background
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                      fill=color, outline=color)
        
        # Draw text
        draw.text((text_x, text_y), text, fill='white', font=font)
    
    # Save the image
    image.save(output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Print summary
    total_detections = len(boxes)
    displayed_detections = sum(1 for score in scores if score >= confidence_threshold)
    print(f"Total detections: {total_detections}")
    print(f"Displayed detections (confidence >= {confidence_threshold}): {displayed_detections}")
