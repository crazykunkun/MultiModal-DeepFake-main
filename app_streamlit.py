from pathlib import Path
from typing import Dict, Tuple
import io
import base64

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from infer_single import load_inference_runtime, run_single_inference


st.set_page_config(page_title="CSCL 多模态真伪检测", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path("D:/study/bishe/DEMO/model")
DEFAULT_CHECKPOINT = MODEL_ROOT / "checkpoint_49.pth"
DEFAULT_TEXT_ENCODER = MODEL_ROOT / "roberta-base"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "test.yaml"


def format_runtime_error(exc: Exception) -> str:
    raw = str(exc)
    hints = []

    if "ViT-B-16.pt" in raw:
        hints.append("缺少 ViT-B-16.pt，请确认文件位于 D:/study/bishe/DEMO/model/ViT-B-16.pt。")
    if "meter_clip16_224_roberta_pretrain.ckpt" in raw:
        hints.append("缺少 meter_clip16_224_roberta_pretrain.ckpt，请确认文件位于 D:/study/bishe/DEMO/model/。")
    if "roberta-base" in raw:
        hints.append("未找到 roberta-base 目录，请确认目录位于 D:/study/bishe/DEMO/model/roberta-base。")
    if "CUDA is unavailable" in raw:
        hints.append("当前环境没有可用 CUDA，请选择 auto（自动回退 CPU）或直接选择 cpu。")

    if hints:
        return raw + "\n\n建议：\n- " + "\n- ".join(hints)
    return raw


@st.cache_resource(show_spinner=False)
def load_runtime_cached(checkpoint: str, config_path: str, text_encoder: str, device: str):
    allow_cuda_fallback = device == "auto"
    return load_inference_runtime(
        checkpoint=checkpoint,
        config_path=config_path,
        text_encoder=text_encoder,
        device=device,
        allow_cuda_fallback=allow_cuda_fallback,
    )


def validate_required_paths(checkpoint: str, config_path: str, text_encoder: str):
    missing = []
    if not Path(checkpoint).exists():
        missing.append(f"checkpoint 不存在：{checkpoint}")
    if not Path(config_path).exists():
        missing.append(f"config 不存在：{config_path}")
    if not Path(text_encoder).exists():
        missing.append(f"text_encoder 不存在：{text_encoder}")
    return missing


def draw_pred_box(image: Image.Image, bbox_cxcywh_norm) -> Tuple[Image.Image, bool]:
    if not bbox_cxcywh_norm or len(bbox_cxcywh_norm) != 4:
        return image, False

    cx, cy, w, h = [float(v) for v in bbox_cxcywh_norm]
    if w <= 0 or h <= 0:
        return image, False

    width, height = image.size
    x1 = int(max(0, min(width - 1, (cx - w / 2.0) * width)))
    y1 = int(max(0, min(height - 1, (cy - h / 2.0) * height)))
    x2 = int(max(0, min(width - 1, (cx + w / 2.0) * width)))
    y2 = int(max(0, min(height - 1, (cy + h / 2.0) * height)))

    if x2 <= x1 or y2 <= y1:
        return image, False

    out = image.copy()
    draw = ImageDraw.Draw(out)
    line_width = max(2, min(width, height) // 200)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)
    return out, True


def render_zoomable_image(image: Image.Image, caption: str, key: str):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    canvas_id = f"zoom_canvas_{key}"
    img_id = f"zoom_img_{key}"

    components.html(
        f"""
        <div style="margin: 0 0 6px 0; font-size: 14px; color: #666;">{caption}</div>
        <div style="display:flex; gap:8px; margin-bottom:8px;">
            <button onclick="zoom_{key}(1.2)">放大</button>
            <button onclick="zoom_{key}(1/1.2)">缩小</button>
            <button onclick="reset_{key}()">重置</button>
        </div>
        <div id="{canvas_id}" style="width:100%; height:520px; border:1px solid #ddd; overflow:hidden; position:relative; background:#fafafa; cursor:grab;">
            <img id="{img_id}" src="data:image/png;base64,{b64}" style="position:absolute; left:50%; top:50%; transform:translate(-50%,-50%) scale(1); transform-origin:center center; max-width:none; user-select:none; -webkit-user-drag:none;" />
        </div>

        <script>
            const canvas_{key} = document.getElementById("{canvas_id}");
            const img_{key} = document.getElementById("{img_id}");
            let scale_{key} = 1;
            let tx_{key} = 0;
            let ty_{key} = 0;
            let dragging_{key} = false;
            let sx_{key} = 0;
            let sy_{key} = 0;

            function apply_{key}() {{
                img_{key}.style.transform = `translate(calc(-50% + ${{tx_{key}}}px), calc(-50% + ${{ty_{key}}}px)) scale(${{scale_{key}}})`;
            }}

            function zoom_{key}(ratio) {{
                scale_{key} = Math.max(0.2, Math.min(8, scale_{key} * ratio));
                apply_{key}();
            }}

            function reset_{key}() {{
                scale_{key} = 1;
                tx_{key} = 0;
                ty_{key} = 0;
                apply_{key}();
            }}

            canvas_{key}.addEventListener("wheel", (e) => {{
                e.preventDefault();
                zoom_{key}(e.deltaY < 0 ? 1.1 : 1/1.1);
            }}, {{ passive: false }});

            canvas_{key}.addEventListener("mousedown", (e) => {{
                dragging_{key} = true;
                sx_{key} = e.clientX - tx_{key};
                sy_{key} = e.clientY - ty_{key};
                canvas_{key}.style.cursor = "grabbing";
            }});

            window.addEventListener("mousemove", (e) => {{
                if (!dragging_{key}) return;
                tx_{key} = e.clientX - sx_{key};
                ty_{key} = e.clientY - sy_{key};
                apply_{key}();
            }});

            window.addEventListener("mouseup", () => {{
                dragging_{key} = false;
                canvas_{key}.style.cursor = "grab";
            }});
        </script>
        """,
        height=590,
    )


if "runtime" not in st.session_state:
    st.session_state.runtime = None

st.title("CSCL 多模态真伪检测")
st.caption("上传图片并输入文本（可为空），进行多模态内容真伪检测与定位。")

with st.sidebar:
    st.header("参数设置")
    st.caption("模型路径已写入源码，无需手动输入。")
    st.code(
        "\n".join(
            [
                f"checkpoint: {DEFAULT_CHECKPOINT}",
                f"config: {DEFAULT_CONFIG}",
                f"text_encoder: {DEFAULT_TEXT_ENCODER}",
            ]
        )
    )

    device = st.selectbox("设备", options=["auto", "cuda", "cpu"], index=0)
    threshold = st.slider("判定阈值（fake 概率）", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if st.button("加载模型", type="primary", use_container_width=True):
        checkpoint = str(DEFAULT_CHECKPOINT)
        config_path = str(DEFAULT_CONFIG)
        text_encoder = str(DEFAULT_TEXT_ENCODER)

        path_errors = validate_required_paths(checkpoint, config_path, text_encoder)
        if path_errors:
            st.error("参数校验失败：\n- " + "\n- ".join(path_errors))
            st.session_state.runtime = None
        else:
            with st.spinner("正在加载模型，请稍候..."):
                try:
                    runtime = load_runtime_cached(checkpoint, config_path, text_encoder, device)
                    st.session_state.runtime = runtime
                    st.success("模型加载成功。")
                    if runtime.get("device_message"):
                        st.info(runtime["device_message"])
                except Exception as exc:  # pylint: disable=broad-except
                    st.session_state.runtime = None
                    st.error(format_runtime_error(exc))

if st.session_state.runtime is not None:
    st.success(
        f"当前已加载模型：{st.session_state.runtime.get('checkpoint', '')} | "
        f"设备：{st.session_state.runtime.get('device', '')}"
    )
else:
    st.info("请先在左侧点击“加载模型”。")

uploaded_file = st.file_uploader("上传图片（jpg / jpeg / png）", type=["jpg", "jpeg", "png"])
text_input = st.text_area("输入文本（可为空）", value="", height=120, placeholder="例如：这张图拍摄于某发布会现场")

image_for_infer = None
if uploaded_file is not None:
    image_for_infer = Image.open(uploaded_file).convert("RGB")
    render_zoomable_image(image_for_infer, caption="上传图片预览（滚轮缩放、拖拽平移）", key="upload")

can_infer = st.session_state.runtime is not None and image_for_infer is not None
infer_clicked = st.button("开始检测", type="primary", disabled=not can_infer)

if infer_clicked:
    with st.spinner("正在推理，请稍候..."):
        try:
            result = run_single_inference(
                runtime=st.session_state.runtime,
                image=image_for_infer,
                text=text_input,
                threshold=threshold,
                image_name=uploaded_file.name if uploaded_file is not None else "",
            )

            st.subheader("检测结果")
            if result["prediction"] == "fake":
                st.error("总体判定：伪造内容")
            else:
                st.success("总体判定：真实内容")

            prob_col1, prob_col2 = st.columns(2)
            prob_col1.metric("fake 概率", f"{result['fake_probability']:.2%}")
            prob_col2.metric("real 概率", f"{result['real_probability']:.2%}")

            st.markdown("**细分类别判断**")
            flag_labels: Dict[str, str] = {
                "face_swap": "人脸替换",
                "face_attribute": "人脸属性篡改",
                "text_swap": "文本替换",
                "text_attribute": "文本属性篡改",
            }
            flag_cols = st.columns(4)
            for idx, key in enumerate(["face_swap", "face_attribute", "text_swap", "text_attribute"]):
                value = "是" if int(result["multiclass_flags"][key]) == 1 else "否"
                flag_cols[idx].metric(flag_labels[key], value)

            st.markdown("**定位框可视化**")
            boxed, has_box = draw_pred_box(image_for_infer, result.get("pred_box_cxcywh_norm", []))
            if has_box:
                render_zoomable_image(boxed, caption="预测框（红色，可缩放与拖拽）", key="boxed")
            else:
                st.info("当前结果未给出有效定位框。")

            with st.expander("查看原始 JSON 输出"):
                st.json(result)

        except Exception as exc:  # pylint: disable=broad-except
            st.error("推理失败：\n" + format_runtime_error(exc))
