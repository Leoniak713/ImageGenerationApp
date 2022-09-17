from dash import Dash, dcc, html


from dashboard.utils import get_image, show_image


app = Dash(
    name=__name__,
    external_stylesheets=[
        {
            "href": "https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css",
            "rel": "stylesheet",
        }
    ],
    
    )

app.layout = html.Div(
    className="container-fluid vh-100 bg-dark text-light",
    children=[
        html.H1(
            children="Dalle-app",
        ),
        html.Div(
            className="row g-3 m-3 col-md-10 m-auto",
            children=[
                html.Div(
                    className="col-md-2 m-2",
                    children="Number of images:",
                ),
                html.Div(
                    className="col-md-8 m-2",
                    children="Phrase:",
                )
            ]
        ),
        html.Form(
            className="row g-3 m-3 col-md-10 m-auto",
            children=[
                dcc.Input(
                    id="input_no_images",
                    className="col-md-2 m-2",
                    type="number",
                    placeholder="",
                    min=1,
                ),
                dcc.Input(
                    id="input_phraser",
                    className="col-md-8 m-2",
                    type="text",
                    placeholder="",
                ),
                html.Button(
                    children="RUN",
                    className="col-md-1 m-2 btn btn-outline-success",
                )
            ]
        ),
        html.Div(
            className="col m-5",
            children=[
                html.Div(
                    className="row m-auto",
                    children=[
                        dcc.Graph(
                            id="image_1_1",
                            className="col-md-4 p-2",
                            figure=show_image(get_image("phrase1"))
                        ),
                        dcc.Graph(
                            id="image_1_2",
                            className="col-md-4 p-2",
                            figure=show_image(get_image("phrase2"))
                        ),
                        dcc.Graph(
                            id="image_1_3",
                            className="col-md-4 p-2",
                            figure=show_image(get_image("phrase3"))
                        )
                    ]
                )
            ]
        )
    ]
)