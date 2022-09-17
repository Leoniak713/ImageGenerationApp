import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_image(phrase: str) -> np.ndarray:
    return np.random.rand(256, 256, 3) * 255

def show_image(image: np.ndarray) -> go.Figure:
    fig = px.imshow(image)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
