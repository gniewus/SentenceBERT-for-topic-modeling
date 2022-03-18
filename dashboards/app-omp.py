import dash
from datetime import datetime
import json
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html 
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly
import plotly.express as px
from dateutil import parser
import sys,random
from pprint import pprint
from dash_table.Format import Format, Scheme, Trim

from wordcloud import WordCloud
from io import BytesIO
import base64
import os
import dash_table
sys.path.append(os.getcwd())
sys.path.append("..")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
application = app.server


df = pd.read_csv("../data/omp_bert_results.csv",sep=';',engine='python',error_bad_lines=False)
df['topic_label'] = df.SBERT_TOPIC_LABEL
df['topic_size'] = df.SBERT_TOPIC_SIZE
df['topic_description'] = df.SBERT_TOPIC_DESC
df['PageViews'] = 1.0
df = df[df.topic_label != -1]

df['created_at'] = pd.to_datetime(df['created_at'])
least_recent_date = df['created_at'].min().date()
most_recent_date = df['created_at'].max().date()
df['created_at']=df['created_at'].fillna(most_recent_date)
df['created_at'] = pd.to_datetime(df['created_at'])
#@app.callback(Input("date-picker", "start_date"), Input("date-picker", "end_date"))
def base_plot(  result):

    result["topic_label"] = result.topic_label.apply(str)
    result['date'] = result['created_at'].apply(str)

    result['created_at'] = pd.to_datetime(result['created_at'])
    fig = px.scatter(result, y="x", x="y", hover_name="title", hover_data={"date": True, 'topic_label': True},  # ["created_at"],
                     color="topic_label", opacity=0.89, color_discrete_sequence=plotly.colors.sequential.Rainbow)
    fig["layout"].pop("updatemenus")
    fig.update_traces(marker=dict(size=9,
                                  line=dict(width=.15,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(height=650)\
        .update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


def get_top_topics_table_data(df, start, end):
    #print(start, end)
    if not start and not end:
        topic_labels = df.topic_label.unique()
    elif start and not end:
        topic_labels = df[df.created_at.dt.date > start].topic_label.unique()
    elif not start and end:
        topic_labels = df[df.created_at.dt.date < end].topic_label.unique()
    elif start and end:
        topic_labels = df[(df.created_at.dt.date <= pd.to_datetime(end).date()) & (
            df.created_at.dt.date >= pd.to_datetime(start).date())].topic_label.unique()
    else:
        topic_labels = df.topic_label.unique()
    # tmp = df.loc[df.topic_label.isin(topic_labels)].groupby(["topic_label", 'topic_description', "new_topic_size"]).agg({'PageViews': ['sum'],"article_uid":["count"]})
    # tmp.columns = tmp.columns.get_level_values(0)
    # tmp = tmp.rename(columns={"article_uid":"count"})

    # pprint(df.iloc[topic_labels].groupby(["topic_label", 'topic_description', "new_topic_size"]).agg({'PageViews': ['sum'],"uid":["count"]})\
    #         #.sort_values(by=["PageViews"], ascending=False)\
    #         .reset_index().columns)

    tmp = df.loc[df.topic_label.isin(topic_labels)].groupby(["topic_label", 'topic_desc','topic_description', "topic_size"]).sum()\
            .sort_values(by=["PageViews"], ascending=False)\
            .reset_index()[["topic_label", "topic_desc",'topic_description', "topic_size", "PageViews"]]\
            .head(20).copy()
    tmp["PageViews/Article"]= tmp["PageViews"]/tmp["topic_size"]
    return tmp.to_dict("records")


app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            [
                                dbc.CardBody([
                                    html.H1(
                                        children='OMP Dataset exploration dashboard', id="top-title"),
                                    html.P(
                                        children='''Here you can explore most popular topics present in the artciles of OMP dataset.''', id="desc")
                                ])
                            ])
                    ])
                ], width=8),
                dbc.Col([
                    html.Div([
                        dbc.Card(

                            [dbc.CardBody([
                                dcc.DatePickerRange(
                                    id="date-picker",
                                    # is_RTL=True,
                                    initial_visible_month=most_recent_date,
                                    number_of_months_shown=3,
                                    first_day_of_week=1,
                                    end_date=most_recent_date,
                                    start_date=least_recent_date,
                                    max_date_allowed=most_recent_date,
                                    min_date_allowed=least_recent_date),
                                dcc.Input(
                                    id="search",
                                    type="search",
                                    placeholder="Search")
                            ])]
                        )
                    ]),



                ], width=4),
            ], align='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([

                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                html.H2(
                                    children='Top 15 Topics according to PageViews', id="title"),
                                html.P(
                                    children='''See the details of the best performing topics. Select one by clicking the radio button left to see the articles included.'''),
                                dash_table.DataTable(
                                    id="top-topics",
                                    style_data={
                                       'whiteSpace': 'normal',
                                       'height': 'auto',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis'},
                                    style_as_list_view=True,
                                    sort_action='native',
                                    row_selectable="single",
                                    style_header={
                                       'backgroundColor': 'rgb(30, 30, 30)'},
                                    style_cell={
                                        'fontSize':17, 'font-family':'sans-serif',
                                        'backgroundColor': 'rgb(50, 56, 62)',
                                        'color': 'white',
                                        'whiteSpace': 'normal',
                                        'height': 'auto',
                                        'lineHeight': '15px'
                                    },


                                    data=get_top_topics_table_data(
                                        df, least_recent_date, most_recent_date),
                                    columns=[{'id': "topic_label", 'name': "topic_label", 'hideable': True},
                                             {"id": "topic_desc", "name": "OMP Topic",'hideable': True},
                                             {"id": "topic_description", "name": "BERT Topic"}, 
                                             {"id": "topic_size", "name": "Total size of topic", "type": "numeric"},
                                             {"id": "PageViews", "name": "Comments", "type": "numeric","format":dict(specifier=',.0f', locale=dict(separate_4digits=False))
                                             }],
                                             #{"id": "PageViews/Article", "name": "Page Views per Article", "type": "numeric","format":Format(precision=2, scheme=Scheme.decimal).group(True)}],
                                    #[{'id': c, 'name': c} for c in ["topic_label", "topic_desc", "topic_size", "PageViews"]]
                                    hidden_columns=["topic_label"]
                                )
                            ])
                        ])
                    ]),


                ], width=6),
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            [
                                dbc.CardBody([
                                    html.H2(
                                        children='Topics per article count', id="graph-title"),
                                    dcc.Graph(
                                        id='graph3',
                                        figure=base_plot(df)
                                    ),
                                    
                                ])]
                        )
                    ])
                ], width=6),
            ], align='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            [
                                dbc.CardBody([
                                    html.H3(children='Articles preview',
                                            id="prev-title"),
                                    dash_table.DataTable(
                                        id='table',
                                        columns=[{"name": i, "id": i}
                                                 for i in ["topic_label", "title", "PageViews", "created_at"]],
                                        hidden_columns=["topic_label"],
                                        sort_action='native',
                                        data=df[["topic_label", "title", 'created_at',"PageViews"]].head(
                                            10).to_dict('records'),
                                        # style_cell=dict(textAlign='left'),
                                        style_header={
                                            'backgroundColor': 'rgb(30, 30, 30)'},
                                        style_cell={
                                            'fontSize':17, 'font-family':'sans-serif',
                                            'backgroundColor': 'rgb(50, 56, 62)',
                                            'color': 'white'
                                        })
                                ])]
                        )
                    ]),


                ], width=8),
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                html.H1(
                                    children='Wordcloud of the topic', id="wc-title"),
                                html.Img(id="image_wc", title='')
                            ])
                        )
                    ]),


                ], width=4)
            ], align='center'),
        ]), color='dark'
    )
])


@app.callback(
    Output("top-topics", 'data'),
    [Input("date-picker", "start_date"), Input("date-picker", "end_date")],
    [State('top-topics', 'data'), State('top-topics', 'columns')]
)
def update_top_topics(start, end, data, columns):
    print("start:", start)
    print("end:", end)
    if start or end:
        data = get_top_topics_table_data(df, start, end)

    return data


@app.callback(
    Output('table', 'data'),
    [Input('graph3', 'hoverData'), Input('top-topics', 'selected_rows')],
    [State('table', 'data'), State('top-topics', 'data')]
)
def update_articles_preview(value, selected_rows, data, data_top_topics):
    ctx = dash.callback_context

    if ctx.triggered:
        if ("graph3" in ctx.triggered[0]['prop_id']):
            topic_label = value['points'][0]["customdata"][1]
            data = df[df.topic_label == topic_label][["topic_label",
                                                      "title", 'created_at']].head(10).to_dict('records')
        else:
            selected_topic_label = data_top_topics[selected_rows[0]
                                                   ]["topic_label"]
           # print(selected_topic_label)
            # print(df.shape)
            #print(df[df.topic_label == selected_topic_label][["topic_label","title", 'created_at']].head(10).to_dict('records'))
            return df[df.topic_label == selected_topic_label][["topic_label",
                                                               "title", 'created_at',"PageViews"]].head(10).to_dict('records')
    return data




def plot_wordcloud(d):

    wc = WordCloud(background_color='rgb(50, 56, 62)', width=740, height=340)
    wc.fit_words(d)
    return wc.recolor(color_func=grey_color_func, random_state=3).to_image()
    
@app.callback(Output('graph3','figure'),[Input("date-picker", "start_date"), Input("date-picker", "end_date")],State('graph3', 'data'))
def update_base_plot(start,end,data):
    print(start,end,'jjjj')

    if not start and not end:
        d = df
    else:
        d = df[df.created_at > start ]

    return base_plot(d)

@app.callback(Output('image_wc', 'src'), [Input('image_wc', 'id'), Input("graph3", "hoverData"),Input("top-topics","selected_rows")])
def make_image(inp, value,selected_rows, data_top_topics):
    img = BytesIO()
    sep='| '
    topic_description ='topic_description'
    if value:
        # print(value['points'])
        topic_label = value['points'][0]["customdata"][1]
        tmp = df[df.topic_label == topic_label].head(1)[topic_description].values
        tmp = tmp[0].replace("['", "").replace("']", '').split(sep)
        words_freq = dict(zip(tmp, list(range(1, len(tmp), 1))[::-1]))
        plot_wordcloud(d=words_freq).save(img, format='PNG')

    if selected_rows:
        selected_topic_label = data_top_topics[selected_rows[0]
                                                   ]["topic_label"]
        tmp = df[df.topic_label == selected_topic_label].head(1)[topic_description].values
        tmp = tmp[0].replace("['", "").replace("']", '').split(sep)
        words_freq = dict(zip(tmp, list(range(1, len(tmp), 1))[::-1]))
        plot_wordcloud(d=words_freq).save(img, format='PNG')
    #print(value,selected_rows)
    plot_wordcloud(d={"bild": 1, "zeitung": 2}).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(px.colors.qualitative.Dark24)#"rgb({}, {}, {})".format(random.randint(100, 244),random.randint(100, 244),random.randint(100, 244))

if __name__ == '__main__':
    app.run_server(debug=True)
