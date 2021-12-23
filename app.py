# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#import chart_studio.plotly as py
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random as r
import pyrebase
from scipy.signal import savgol_filter

import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

firebaseConfig = {
  'apiKey': "AIzaSyDi7ICIgX9QAxtVIPsJZTKU6jomlgS9eIM",
  'authDomain': "bintracker-e6fdf.firebaseapp.com",
  'databaseURL': "https://bintracker-e6fdf-default-rtdb.europe-west1.firebasedatabase.app",
  'projectId': "bintracker-e6fdf",
  'storageBucket': "bintracker-e6fdf.appspot.com",
  'messagingSenderId': "688874903644",
  'appId': "1:688874903644:web:39533aeb09e7ac08ba2e99",
  'measurementId': "G-3MXSHZ51NM"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

sensor_data = db.child('readings_test5').get().val()

# FUNCTIONS

def fillGaps(df, timeCol, weightCol, colsList):
    
    df2 = pd.DataFrame(columns = colsList)
    df2 = pd.concat([df2, df.iloc[0].to_frame().T])
    for i in range(1,len(df)):
        # as long as there is a gap
        newTime = df[timeCol][i-1]
        count = 1
        while df[timeCol][i]-newTime > 320:
            initialize = [0.1 for i in range(len(colsList))]
            data = pd.DataFrame([initialize], columns = colsList)
            newTime+=300
            data[timeCol][0] = newTime
            for col in colsList:
                if col!=timeCol:
                    if col == weightCol:
                        new = float(df[col][i-1])+r.uniform(-0.01*(1+0.7*np.sqrt(count)), 0.01*(1-0.7*np.sqrt(count)))
                        count+=1
                    else:
                        new = float(df[col][i-1])+r.uniform(-0.01, 0.01)
                    data[col][0] = float(new)
            df2 = pd.concat([df2, data.iloc[0].to_frame().T], ignore_index=True)
        df2 = pd.concat([df2, df.iloc[i].to_frame().T], ignore_index=True)     
    return df2

#differentiate
def differentiate(df, ycol, newCol):
    dwdt = [0]
    for i in range(1,len(df)):
        dwdt.append(df[ycol][i]-df[ycol][i-1])
    dwdt[0]=dwdt[1]
    df[newCol]=dwdt
    return

def isDecrease(x):
    if x<=0:
        return 1
    else:
        return 0

# Levelling data to measure weight according to increases in the readings
def level(d, ycol, gradcol, eventcol, newCol):
    
    w_new = []
    base = d[ycol][0]
    lowest = base
    prevEvent = 0
    
    for i in range(len(d)):
        if d[eventcol][i]>prevEvent:
                base = d[ycol][i]
                lowest = base
                prevEvent = d[eventcol][i]

        w = d[ycol][i]
        if i==0:
            d_prev = d[gradcol][i]
            w_prev = d[ycol][i]
        else:
            d_prev = d[gradcol][i-1]
            w_prev = d[ycol][i-1]

        if d[gradcol][i] == 0 and d_prev == 1:
            lowest = w_prev
            w = w-lowest+base
            
        elif d[gradcol][i] == 1 and d_prev == 0:
            w = w-lowest+base
            base=w
            
        elif d[gradcol][i] == 1:
            w = base + r.uniform(-0.010, 0.010)

        elif d[gradcol][i] == 0:
            w = w-lowest+base

        w_new.append(w)
    d[newCol] = w_new
    return

# Identify bin emptying events
def binEmpty(df):
    binEvents = [0]
    eventNo = 0
    for i in range(1,len(df)):
        if (df['weight'][i-1]-df['weight'][i])>0.5:
            eventNo+=1
        binEvents.append(eventNo)
    df['bin_event'] = binEvents
    return

def replaceSTD(df, stdCol, avgCol, newAvgCol, maxStd):
    df[newAvgCol]=np.nan
    for i in range(len(df)):
        if df[stdCol][i]>maxStd:
            df[newAvgCol][i]=df[newAvgCol][i-1]
        else:
            df[newAvgCol][i]=df[avgCol][i]
    return df

def preProcess(sensorData):
    df0 = pd.DataFrame(sensorData).transpose()
    df0.reset_index(drop=True, inplace=True)
    df0['time']=df0['time'].apply(lambda x: float(x))

    df = fillGaps(df0, 'time','weight', list(df0.columns))
    df['datetime']=df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    for dist in ['dist1','dist2','dist3']:
        for i in range(len(df)):
            if df[dist][i]>500:
                df[dist][i]=1

    df['avgdist'] = df[['dist1', 'dist2', 'dist3']].mean(axis=1)
    df['dist_std'] = df[['dist1', 'dist2', 'dist3']].std(axis=1)

    # Levelling weight readings
    df['weight_fil'] = savgol_filter(df['weight'], 45, 2)
    differentiate(df, 'weight_fil', 'dw/dt_f')
    df['decrease']=df['dw/dt_f'].apply(lambda x: isDecrease(x))
    binEmpty(df)
    level(df, 'weight', 'decrease', 'bin_event', 'w_new')
    df = replaceSTD(df,'dist_std', 'avgdist','new_avgdist',15)
    df['dist_ewm']=df['new_avgdist'].ewm(span=7,adjust=False).mean()
    return df

def binUse(df):
    dff = df.drop(columns=['weight_fil','decrease','dw/dt_f'])
    dff['weight_emw']=dff['w_new'].ewm(span=5,adjust=False).mean()
    dff['w_diff']=dff['weight_emw'].diff()
    dff['bin_use']=dff['w_diff'].apply(lambda x: int(x>0.005))
    dff['date']=dff['datetime'].dt.date
    dff['time']=dff['datetime'].dt.time
    dff['date'] = df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
    dff['time'] = df['datetime'].apply(lambda x: (x.hour * 60 + x.minute) * 60 + x.second)
    
    df10 = dff[dff['date']=='2021-12-10']
    df10.reset_index(drop=True, inplace=True)
    df10 = df10[df10['bin_use']==1]

    df11 = dff[dff['date']=='2021-12-11']
    df11.reset_index(drop=True, inplace=True)
    df11 = df11[df11['bin_use']==1]

    df12 = dff[dff['date']=='2021-12-12']
    df12.reset_index(drop=True, inplace=True)
    df12 = df12[df12['bin_use']==1]

    df13 = dff[dff['date']=='2021-12-13']
    df13.reset_index(drop=True, inplace=True)
    df13 = df13[df13['bin_use']==1]

    df14 = dff[dff['date']=='2021-12-14']
    df14.reset_index(drop=True, inplace=True)
    df14 = df14[df14['bin_use']==1]

    df15 = dff[dff['date']=='2021-12-15']
    df15.reset_index(drop=True, inplace=True)
    df15 = df15[df15['bin_use']==1]

    df16 = dff[dff['date']=='2021-12-16']
    df16.reset_index(drop=True, inplace=True)
    df16= df16[df16['bin_use']==1]

    df17 = dff[dff['date']=='2021-12-17']
    df17.reset_index(drop=True, inplace=True)
    df17 = df17[df17['bin_use']==1]

    df18 = dff[dff['date']=='2021-12-18']
    df18.reset_index(drop=True, inplace=True)
    df18 = df18[df18['bin_use']==1]
    
    fig_use,ax = plt.subplots()
    plt.eventplot(df10['time'], orientation='horizontal', lw=7, lineoffsets=0, linelengths=0.2, colors='r')
    plt.eventplot(df11['time'], orientation='horizontal', lw=7, lineoffsets=-0.25, linelengths=0.2, colors='orange')
    plt.eventplot(df12['time'], orientation='horizontal', lw=7, lineoffsets=-0.5, linelengths=0.2, colors='yellow')
    plt.eventplot(df13['time'], orientation='horizontal', lw=7, lineoffsets=-0.75, linelengths=0.2, colors='b')
    plt.eventplot(df14['time'], orientation='horizontal', lw=7, lineoffsets=-1, linelengths=0.2, colors='g')
    plt.eventplot(df15['time'], orientation='horizontal', lw=7, lineoffsets=-1.25, linelengths=0.2, colors='cyan')
    plt.eventplot(df16['time'], orientation='horizontal', lw=7, lineoffsets=-1.5, linelengths=0.2, colors='purple')
    plt.eventplot(df17['time'], orientation='horizontal', lw=7, lineoffsets=-1.75, linelengths=0.2, colors='pink')
    plt.eventplot(df18['time'], orientation='horizontal', lw=7, lineoffsets=-2, linelengths=0.2, colors='brown')
    
    return fig_use

df = preProcess(sensor_data)

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=df['datetime'][np.arange(0,len(df),6)], y=df['w_new'][np.arange(0,len(df),6)], name="Weight"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['datetime'][np.arange(0,len(df),6)], y=df['dist_ewm'][np.arange(0,len(df),6)], name="Average Depth"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Weight and Average Depth over Time"
)

# Set x-axis title
fig.update_xaxes(title_text="datetime")

# Set y-axes titles
fig.update_yaxes(title_text="Weight (kg)", secondary_y=False)
fig.update_yaxes(title_text="Average Depth (cm)", secondary_y=True)


#figUse = binUse(df)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Household Bin Use", style={'text-align':'center', 'font-family': 'Verdana', 'font-style':'bold'}),
    html.Div("Tracking the weight and fill level of the flat's kitchen bin",style={'text-align':'center', 'font-family': 'Verdana'}),
    # All elements from the top of the page
    html.Div([
        html.H2("Correlation between Bin Weight and Trash Depth", style={'font-family': 'Verdana'}),
        dcc.Graph(id='weight_depth_graph', figure=fig)
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([ 
        html.H2("Correlation between Bin Weight and Trash Depth", style={'font-family': 'Verdana'}),
        html.Div('Select a bin event (an instance of the bin filling up) to see the relationship between the trash weight and level', style={'font-family': 'Verdana'}),
        html.Br(),
        dcc.Dropdown(id='select_event',
                    options = [
                        {'label': 'All', 'value': 'All'},
                        {'label': 0, 'value': 0},
                        {'label': 1, 'value': 1},
                        {'label': 2, 'value': 2},
                        {'label': 3, 'value': 3}],
                    multi = False,
                    clearable = False,
                    value = 'All',
                    style = {"width": "40%", 'font-family': 'Verdana', 'text-align':'center'}
                    ),
        html.Br(),
        html.Div(id='output_container',children=[], style={'font-family': 'Verdana'}),
        html.Div(id='rmse_container',children=[], style={'font-family': 'Verdana'}),
        html.Div(id='rscore_container',children=[], style={'font-family': 'Verdana'}),
        html.Div(id='grad_container',children=[], style={'font-family': 'Verdana'}),
        html.Div(id='int_container',children=[], style={'font-family': 'Verdana'}),
        dcc.Graph(id='correlation_graph')
    ])
])

@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='rmse_container', component_property='children'),
     Output(component_id='rscore_container', component_property='children'),
     Output(component_id='grad_container', component_property='children'),
     Output(component_id='int_container', component_property='children'),
     Output(component_id='correlation_graph', component_property='figure')],
    [Input(component_id='select_event', component_property='value')]
)
def update_graph(event):
    container = "Bin event selected: {}".format(event)
    data = df[['w_new','new_avgdist', 'dist_std', 'bin_event']]
    if event!='All':
        data = data[data['bin_event']==event]
    x = data['new_avgdist'].values.reshape(-1,1)
    y = data['w_new'].values
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.3)
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(x_train,y_train)
    y_pred = lasso.predict(x_test)
    rscore = lasso.score(x_test,y_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    coef = lasso.coef_[0]
    intercept = lasso.intercept_
    
    rmse_c = "Root mean squared error: {}".format(rmse)
    rscore_c = "Coefficient of determination (R2 score): {}".format(rscore)
    coef_c = "Gradient: {}".format(coef)
    int_c = "Intercept: {}".format(intercept)
    x_train = x_train.reshape(1,-1)[0]
    x_test = x_test.reshape(1,-1)[0]
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=x_train,y=y_train, mode='markers', name="Training Data"))
    fig.add_trace(go.Scatter(x=x_test,y=y_test, mode='markers', name='Test Data'))
    fig.add_trace(go.Scatter(x=x_test,y=y_pred, name='Predicted Values'))
    return container,rmse_c, rscore_c, coef_c, int_c,fig

if __name__ == '__main__':
    app.run_server(debug=True)