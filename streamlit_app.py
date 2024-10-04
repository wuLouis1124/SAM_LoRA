import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_erosion, binary_dilation
import time  # 新增导入time库

# 加载目标图片，并获取其尺寸
target_image_path = "000068.jpg"  # 目标图片的路径
target_image = Image.open(target_image_path)
target_width, target_height = target_image.size  # 获取目标图片的宽度和高度

# 定义缩放函数，将图片直接缩放至目标大小（不保留原比例）
def resize_image_to_target(image, target_width, target_height):
    """
    将输入图片直接缩放至目标图片的宽度和高度，不保留原宽高比。
    :param image: 输入的PIL图片对象
    :param target_width: 目标图片的宽度
    :param target_height: 目标图片的高度
    :return: 缩放后的Tensor对象，格式为 (B, C, H, W)
    """
    # 直接缩放图片到目标宽度和高度
    resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return resized_image

# 加载SAM模型
@st.cache_resource
def load_sam_model():
    model_type = "vit_h"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

# 显示分割蒙版和发光边框
def show_mask(mask, image, ax):
    color = np.random.random(3)  # 随机生成颜色
    h, w = mask.shape[-2:]

    # 创建蒙版图像并应用颜色
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    img_with_mask = image.copy()

    # 创建透明的蒙版以突出显示边缘
    alpha_mask = 0.4  # 透明度
    img_with_mask[mask != 0] = img_with_mask[mask != 0] * (1 - alpha_mask) + mask_image[mask != 0] * (255 * alpha_mask)

    # 提取蒙版的边缘
    edges = mask.astype(bool) ^ binary_erosion(mask.astype(bool))

    # 绘制发光效果
    glow_color = (0, 1, 0, 0.5)  # 发光颜色
    glow_width = 5  # 发光宽度
    for i in range(1, 4):  # 绘制多个边框以模拟发光
        # 在边缘上绘制发光效果
        ax.imshow(edges * (1 - (i * 0.2)), cmap='Greens', alpha=glow_width * 0.2, interpolation='nearest', extent=[0, w, h, 0])

    ax.imshow(img_with_mask)

# 显示框
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# SAM分割逻辑
def sam_segmentation(image_np, predictor, method, point_coords=None, box_coords=None):
    predictor.set_image(image_np)
    if method == "point" and point_coords is not None:
        input_points = np.array([point_coords])
        input_labels = np.array([1])
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
    elif method == "box" and box_coords is not None:
        box_coords = np.array(box_coords).reshape(-1, 2)
        masks, _, _ = predictor.predict(box=box_coords, multimask_output=False)
    else:
        return None
    return masks[0]

# Streamlit 前端布局
st.set_page_config(layout="wide")

st.title("✨ Segment Anything 🏜")
st.info('Let me help generate segments for any of your images. 😉')

# 初始化会话状态用于保存分割方式
if "method" not in st.session_state:
    st.session_state["method"] = "point"
    method = "point"
    
col1, col2 = st.columns([1, 4])

with col1:
    st.header("提示分割方式")
    if st.button("选择Point方式分割", key="point_btn", use_container_width=True, icon=":material/my_location:") :
        st.session_state["method"] = "point"
    
    if st.button("选择Box方式分割", key="box_btn", use_container_width=True, icon=":material/indeterminate_question_box:"):
        st.session_state["method"] = "box"

    st.header("提示分割数量")
    if st.button("single-masked", key="single_masked", use_container_width=True, icon=":material/looks_one:") :
        st.session_state["number"] = "single"
    
    if st.button("multi-masked", key="multi-masked", use_container_width=True, icon=":material/more_horiz:"):
        st.session_state["number"] = "multi"

with col2:
    # 上传图片
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png", "jpeg"])
    
    mask = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = resize_image_to_target(image, target_width, target_height)
        image_np = np.array(image)

        # 加载 SAM 模型
        predictor = load_sam_model()

        # 创建左右两个列
        img_col1, img_col2 = st.columns(2)

        method = st.session_state["method"]
        if method == "point":
            # 显示上传的图片作为交互图（图片1）
            with img_col1:
                # 直接显示上传的图片
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 255, 0.3)",  # 填充颜色
                    stroke_width=2,
                    background_image=Image.fromarray(image_np),  # 每次重新加载原始图片
                    update_streamlit=True,
                    height=image_np.shape[0],
                    width=image_np.shape[1],
                    drawing_mode="point",  # 设置为点击模式
                    key="canvas",
                )

            # 捕获用户点击位置并只保留最新的点击
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                # 只获取最后一个点击的坐标
                obj = canvas_result.json_data["objects"][-1]
                point_x = int(obj["left"])
                point_y = int(obj["top"])

                # 获取分割结果
                mask = sam_segmentation(image_np, predictor, method="point", point_coords=(point_x, point_y))

        elif method == "box":

            # 使用 Streamlit Drawable Canvas 允许用户绘制矩形框
            with img_col1:
                box_canvas = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.1)",  # 填充颜色
                    stroke_width=2,
                    background_image=Image.fromarray(image_np),  # 每次重新加载原始图片
                    update_streamlit=True,
                    height=image_np.shape[0],
                    width=image_np.shape[1],
                    drawing_mode="rect",  # 设置为矩形绘制模式
                    key="box_canvas",
                )

            # 获取用户绘制的框坐标并只保留最新的框
            if box_canvas.json_data is not None and box_canvas.json_data["objects"]:
                # 获取最后一个框的坐标
                obj = box_canvas.json_data["objects"][-1]
                box_x1 = int(obj["left"])
                box_y1 = int(obj["top"])
                box_x2 = box_x1 + int(obj["width"])
                box_y2 = box_y1 + int(obj["height"])

                # 获取分割结果
                mask = sam_segmentation(image_np, predictor, method="box", box_coords=[box_x1, box_y1, box_x2, box_y2])

        # 显示分割结果的图片（图片2）
        with img_col2:
            if mask is not None:
                with st.spinner("图片2正在加载..."):
                    time.sleep(1)  # 强制延迟1秒
                    # 创建分割可视化，清除旧的显示最新的分割结果
                    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)  # 设置 dpi 保持一致性
                    ax.imshow(image_np)
                    show_mask(mask, image_np, ax)
                    if method == "box":
                        show_box([box_x1, box_y1, box_x2, box_y2], ax)
                     # 关闭坐标轴并调整边距
                    ax.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除边界

                    # 使用 bbox_inches='tight' 选项以确保没有额外的边缘
                    st.pyplot(fig, bbox_inches='tight')
