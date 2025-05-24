import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import fitz
from datetime import datetime
from io import StringIO
import time

with open('api_key.txt', 'r') as file:
    client = OpenAI(
      api_key=file.read().strip()
    )
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

def project(df, target, features=None, future_x=None):
    """
    This function takes in two arrays, x and y, and a future value. It fits a linear regression model to the data
    and predicts the y value for the given future x value.

    :param x: Array of independent variable values
    :param y: Array of dependent variable values
    :param future: Future value of the independent variable for which we want to predict the dependent variable
    :return: Predicted value of the dependent variable for the given future independent variable
    """

    if features == None:
        features = [f for f in df.columns if f not in ['Profit', 'Enterprise']]

    x_pred = []
    if 'R&D Spend' in features:
        x_pred.append(future_x[0])
    if 'Marketing Spend' in features:
        x_pred.append(future_x[1])
    if 'Administration' in features:
        x_pred.append(future_x[2])

    x_pred = np.array(x_pred).reshape(-1, len(features))



    x = df[features].values if features else df.values
    y = df[target].values
    # Reshape x to be a 2D arrays
    x = x.reshape(-1, len(features))
    y = y.reshape(-1, 1)
    # Create a linear regression lr_model
    model = LinearRegression()
    model.fit(x, y)

    coeff = model.coef_[0]
    intercept = model.intercept_

    return model, model.predict(x_pred)[0][0]


def plot_relationship(df, feature, target, lr_model=None, projected_value=None, future_x=None):
    fig = px.scatter(df, x=feature, y='Profit', color='Enterprise', title=f'{target} vs {feature}')
    fig.update_layout(transition_duration=500,
                        paper_bgcolor="#f7f7f7",
                        plot_bgcolor="#f7f7f7",
                        font=dict(family="Arial", color="#371456", size=16),
                        xaxis_title=feature,
                        yaxis_title=target,
                        template='plotly_white'
                      )

    if lr_model:
        fig.add_trace(go.Scatter(
            x=[future_x[['R&D Spend', 'Administration', 'Marketing Spend'].index(feature)]],
            y=[projected_value],
            mode='markers',
            marker=dict(size=30, symbol='x'),
            name='Projected Profit'
        ))
    fig.update_traces(marker=dict(size=12, ),
                      selector=dict(mode='markers'))
    return fig

def extract_text_from_pdf(pdf_path):
    """
    This function takes in a PDF file path and extracts text from it using PyMuPDF.
    :param pdf_path: Path to the PDF file
    :return: Extracted text from the PDF
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_csv_from_markdown(text):
    if '```csv' in text:
        return text.split('```csv')[1].split('```')[0].strip()
    return text.strip()

def gpt_pred(text_input, filename):
    prompt = f"""
        Extract the following structured data from the text and return it as CSV in the format that is suitable for StringIO and pandas read_csv. The Date should be able to convert to datetime. The Price should be float. The Income/Expense should be either 'Income' or 'Expense'. The Category should be a string. The CSV should have the following columns:
        Columns: Date, Price, Income/Expense, Category

        Text:
        {text_input}
        """
    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      store=True,
      messages=[
        {"role": "user", "content": prompt}
      ]

    )
    csv_output = completion.choices[0].message.content
    csv_output = extract_csv_from_markdown(csv_output)
    print(csv_output)
    df = pd.read_csv(StringIO(csv_output))
    df.to_csv(f"{filename.split('.')[0]}.csv", index=False)

def plot_budget(df, plot_type='Budget Differential'):
    # Pie chart with breakdown via expense categories
    df['Date'] = pd.to_datetime(df['Date'])

    expenses_data = df[df['Income/Expense'] == 'Expense']
    expenses_data['Cumulative Expenses'] = expenses_data['Price'].cumsum()

    income_data = df[df['Income/Expense'] == 'Income']
    income_data['Cumulative Income'] = income_data['Price'].cumsum()

    if plot_type in ['Spending Breakdown', 'Income Breakdown']:
        fig1 = px.pie(expenses_data, values='Price', names='Category', title='Expenses Breakdown')
        fig2 = px.pie(income_data, values='Price', names='Category', title='Expenses Breakdown')
        df = df[df['Income/Expense'] == 'Income']
        # Create subplot layout
        subfig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

# Add pie chart traces to subplot
        # Add each pie trace with its own showlegend setting
        pie1 = go.Pie(labels=fig1.data[0].labels,
                      values=fig1.data[0].values,
                      title='Expenses Breakdown',
                      legendgroup='group1',
                      showlegend=True,
                      domain=dict(x=[0, 0.48]))  # explicitly set domain

        pie2 = go.Pie(labels=fig2.data[0].labels,
                      values=fig2.data[0].values,
                      title='Income Breakdown',
                      legendgroup='group2',
                      showlegend=True,
                      domain=dict(x=[0.52, 1]))  # explicitly set domain

# Add traces to subplots
        subfig.add_trace(pie1, row=1, col=1)
        subfig.add_trace(pie2, row=1, col=2)

# Turn off unified legend
        subfig.update_layout(
            showlegend=True,
            legend=dict(tracegroupgap=100)
        )
        fig=subfig
    elif plot_type == 'Net Income':
        fig = px.line(expenses_data, x='Date', y='Cumulative Expenses', title='Expenses v Income', color='Income/Expense',)
        fig.add_trace(
            go.Scatter(
                x=income_data['Date'],
                y=income_data['Cumulative Income'],
                mode='lines',
                name='Cumulative Income',
                line=dict(color='green')
            )
        )
        differential = [0]
        for i in range(1,len(df)):
            if df['Income/Expense'][i] == 'Income':
                differential.append(differential[i-1] + df['Price'][i])
            else:
                differential.append(differential[i-1] - df['Price'][i])

        df['Net Income'] = differential

        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Net Income'],
                mode='lines',
                name='Net Income',
                line=dict(color='orange', dash='dash')
            )
        )
    max_expenses = expenses_data['Cumulative Expenses'].max() if expenses_data is not None else 0
    max_income = income_data['Cumulative Income'].max() if income_data is not None else 0

    fig.update_layout(transition_duration=500,
                        paper_bgcolor="#f7f7f7",
                        plot_bgcolor="#f7f7f7",
                        font=dict(family="Arial", color="#371456", size=16),
                        template='plotly_white'
                      )
    return fig, max_expenses, max_income


def add_new_data(filename, fig, max_exp, max_inc):
    text = extract_text_from_pdf(filename)
    gpt_pred(text, filename)
    # Assuming the CSV is structured correctly
    df = pd.read_csv(f'{filename.split('.')[0]}.csv')
    print(df.head())

    differential = [max_inc - max_exp]
    for i in range(1,len(df)):
        if df['Income/Expense'][i] == 'Income':
            differential.append(differential[i-1] + df['Price'][i])
        else:
            differential.append(differential[i-1] - df['Price'][i])

    df['Budget Differential'] = differential
    print(df.head())

    df['Date'] = pd.to_datetime(df['Date'])
    expenses_data = df[df['Income/Expense'] == 'Expense']
    income_data = df[df['Income/Expense'] == 'Income']
    expenses_data['Cumulative Expenses'] = expenses_data['Price'].cumsum()
    expenses_data['Cumulative Expenses'] = [max_exp + i for i in expenses_data['Cumulative Expenses']]
    income_data['Cumulative Income'] = income_data['Price'].cumsum()
    income_data['Cumulative Income'] = [max_inc + i for i in income_data['Cumulative Income']]
    fig.add_trace(
        go.Scatter(
            x=expenses_data['Date'],
            y=expenses_data['Cumulative Expenses'],
            mode='lines',
            name='Cumulative Expenses',
            line=dict(color='red')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=income_data['Date'],
            y=income_data['Cumulative Income'],
            mode='lines',
            name='Cumulative Income',
            line=dict(color='red')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Budget Differential'],
            mode='lines',
            name='Budget Differential',
            line=dict(color='red', dash='dash')
        )
    )
#    fig.update_layout(transition_duration=500)
    return fig
