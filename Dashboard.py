import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load & Preprocess Data
df = pd.read_csv("Final_LSTM_Dataset.csv")
df2 = pd.read_csv("calgary_crime_cleaned.csv")
df3 = pd.read_csv("final_crime_dataset2.csv")
df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))

# Prepare data for Random Forest
X = df[["Estimated Mean Income", "Population", "Mean Temp (°C)"]]
y = df["Crime Rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
X_test_rf = df.loc[y_test.index, ["Estimated Mean Income", "Population", "Mean Temp (°C)"]]
X_test_rf.columns = X_test_rf.columns.astype(str)  # ✅ Make sure all column names are strings
y_pred = model.predict(X_test_rf)

mae = round(mean_absolute_error(y_test, y_pred), 2)
mse = round(mean_squared_error(y_test, y_pred), 2)
rmse = round(mse ** 0.5, 2)
r2 = round(r2_score(y_test, y_pred), 2)




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
            html.H3("Forecasting Future Crime using Random Forest"),
            html.Label("Select Model:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
            id="model-dropdown",
            options=[
                {"label": "Linear Regression", "value": "lr"},
                {"label": "Random Forest", "value": "rf"},
            ],
            value="rf",
            style={"width": "300px", "marginBottom": "20px"}
        ),
        html.Div(id="model-output"),
        dcc.Graph(id="model-prediction-graph")
    ])

@app.callback(
    [Output("rq3-scatter-plot", "figure"),
     Output("rq3-correlation-output", "children")],
    Input("rq3-variable-dropdown", "value")
)
def update_rq3_scatter(selected_var):

    corr = df3["Crime Rate"].corr(df3[selected_var])
    corr_text = f"🔍 Correlation between {selected_var} and Crime Rate: **{corr:.2f}**"

    fig = px.scatter(
        df3,
        x=selected_var,
        y="Crime Rate",
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

@app.callback(
    [Output("model-output", "children"),
     Output("model-prediction-graph", "figure")],
    Input("model-dropdown", "value")
)
def update_model_results(selected_model):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import plotly.graph_objs as go

    features = ["Estimated Mean Income", "Population", "Mean Temp (°C)"]
    target = "Crime Rate"
    df_model = df.copy()
    df_model.dropna(subset=features + [target], inplace=True)
    
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if selected_model == "lr":
        model = LinearRegression()
        model_name = "Linear Regression"
    else:
        model = RandomForestRegressor(random_state=42)
        model_name = "Random Forest"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Results Text
    result_text = html.Div([
        html.P(f"Model: {model_name}"),
        html.P(f"MAE: {mae:.2f}"),
        html.P(f"MSE: {mse:.2f}"),
        html.P(f"RMSE: {rmse:.2f}"),
        html.P(f"R² Score: {r2:.2f}")
    ])

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.values, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines+markers', name='Predicted'))
    fig.update_layout(title=f"{model_name}: Actual vs Predicted Crime Rate",
                      xaxis_title="Test Sample Index", yaxis_title="Crime Rate", height=500)

    return result_text, fig


# Run the App
if __name__ == "__main__":
    app.run(debug=True)
