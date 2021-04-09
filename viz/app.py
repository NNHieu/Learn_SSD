import base64
from io import BytesIO
import time
from PIL import ImageColor
import plotly.graph_objects as go

def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded

def pil_to_fig(im, scale_factor=1.0, showlegend=False, title=None):
    img_width, img_height = im.size
    # scale_factor = target_width/img_width
    img_width *= scale_factor
    img_height *= scale_factor
    fig = go.FigureWidget()

    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(title=title, showlegend=showlegend)

    # Configure other layout
    fig.update_layout(
        width=img_width,
        height=img_height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig

# def get_bbox(x0, y0, x1, y1, fill=False,
#              showlegend=True, name=None, color=None, 
#              opacity=0.5, group=None, text=None):
#     """
    
#     """
#     return go.Scatter(
#         x=[x0, x1, x1, x0, x0],
#         y=[y0, y0, y1, y1, y0],
#         mode="lines",
#         line=dict(dash='dot'),
#         fill="toself" if fill else None,
#         opacity=opacity,
#         marker_color=color,
#         hoveron="fills",
#         name=name,
#         hoverlabel_namelength=0,
#         text=text,
#         legendgroup=group,
#         showlegend=showlegend,
#     )

def get_bbox(x0, y0, x1, y1, **kwargs):
    """
    
    """
    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        **kwargs
    )

# colors for visualization
COLORS = list(ImageColor.colormap.values())

def add_objects(fig, bboxes, class_ids, CLASSES, opacity=0.5):
    num_obj = len(bboxes)
    for i in range(num_obj):
        label = CLASSES[class_ids[i]]
        bbox = bboxes[i]
        # only display legend when it's not in the existing classes
        text = f"class={label}"
        trace_box = get_bbox(*bbox, fill="toself",
            opacity=opacity,
            marker_color=COLORS[hash(label) % len(COLORS)],
            hoveron="fills",
            name=label,
            text=text,
            legendgroup=label)
        fig.add_trace(trace_box)

    
def add_grid(fig, size, shape, thickness=1, hover_fn=None, color="#ffffff"):
    im_width, im_height = size
    row, col = shape
    step_row, step_col = im_height / row, im_width / col
    for i in range(row):
        for j in range(col):
            cell = get_bbox(j * step_col, i * step_row,
                         (j + 1) * step_col, (i + 1) * step_row, color=color, group="grid")
            fig.add_trace(cell)
            if hover_fn is not None:
                print("add hover fun")
                fig.data[-1].on_hover(hover_fn)

class Vizualizer:
    def __init__(self, CLASSES, transforms=None) -> None:
        self.CLASSES = CLASSES
        self.transforms = transforms
    
    def show(self, image, bboxes, class_ids):
        if self.transforms is not None:
            image, bboxes, class_ids = self.transforms(image, bboxes, class_ids)
        fig = pil_to_fig(image, title='Test', showlegend=True)
        add_objects(fig, bboxes, class_ids, self.CLASSES)
        return fig