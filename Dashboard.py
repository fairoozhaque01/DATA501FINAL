import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load & Preprocess Data
df = pd.read_csv("final_crime_dataset.csv")
df2 = pd.read_csv("calgary_crime_cleaned.csv")
df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))


app = dash.Dash(__name__)
app.title = "Calgary Crime Dashboard"

app.layout = html.Div([
    html.H1("Calgary Crime Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="rq-tabs", value="rq1", children=[
        dcc.Tab(label="Most Common Crimes", value="rq1"),
        dcc.Tab(label="Crime Trends", value="rq2"),
        dcc.Tab(label="Demographic Correlations", value="rq3"),
        dcc.Tab(label="Predictive Modeling", value="rq4"),
    ]),
    html.Div(id="tab-content")
])

# Callback to render tab content
@app.callback(Output("tab-content", "children"), Input("rq-tabs", "value"))
def render_tab_content(tab):
    if tab == "rq1":
        crime_counts = df2.groupby("Category")["Crime Count"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(crime_counts, x="Category", y="Crime Count", title="Top 10 Most Common Crimes in Calgary", color_discrete_sequence=["#90BE6D"])
        return html.Div([
            html.H3("Most Common Crime Categories"),
            dcc.Graph(figure=fig)
        ])

    elif tab == "rq2":
        trend_df = df.groupby("Date")["Crime Count"].sum().reset_index()
        fig = px.line(trend_df, x="Date", y="Crime Count", title="Monthly Crime Trend (2021–2024)", color_discrete_sequence=["#F94144"])
        return html.Div([
            html.H3("Monthly Crime Trends Over Time"),
            dcc.Graph(figure=fig)
        ])

    elif tab == "rq3":
        return html.Div([
            html.H3("Crime vs Socioeconomic Factors"),
            html.Label("Select Variable:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="rq3-variable-dropdown",
                options=[
                    {"label": "Estimated Mean Income", "value": "Estimated Mean Income"},
                    {"label": "Population", "value": "Population"}
                ],
                value="Estimated Mean Income",
                style={"width": "300px", "marginBottom": "20px"}
            ),
            dcc.Graph(id="rq3-scatter-plot"),
            html.Div(id="rq3-correlation-output", style={"textAlign": "center", "fontSize": "18px"})
        ])
    # not completed
    elif tab == "rq4":
        return html.Div([
            html.H3("Forecasting Future Crime (LSTM Model Results)"),
            html.P("MAE: 124.98 | RMSE: 166.38 | R²: 0.00"),
            dcc.Graph(figure=px.line(df.tail(12), x="Date", y="Crime Count", title="Actual vs Predicted Crime (example)")),
            html.P("Model includes weather, income, population, etc.")
        ])

@app.callback(
    [Output("rq3-scatter-plot", "figure"),
     Output("rq3-correlation-output", "children")],
    Input("rq3-variable-dropdown", "value")
)
def update_rq3_scatter(selected_var):

    corr = df["Crime Rate per 1K"].corr(df[selected_var])
    corr_text = f"🔍 Correlation between {selected_var} and Crime Rate: **{corr:.2f}**"

    fig = px.scatter(
        df,
        x=selected_var,
        y="Crime Rate per 1K",
        trendline="ols",
        title=f"Crime Rate vs {selected_var} (Ward-Level Averages)",
        labels={"Crime Rate per 1K": "Crime Rate per 1K", selected_var: selected_var},
        color_discrete_sequence=["#577590"]
    )
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    return fig, corr_text
print(df[["Crime Rate per 1K", "Population"]].corr())


# Run the App
if __name__ == "__main__":
    app.run(debug=True)
