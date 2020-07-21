import json
import plotly
from plotly.graph_objs import Bar,Pie
import pandas as pd
import joblib
import pickle

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def get_categ_cnt(df,col_list,op_cat_colname, op_num_colname):
    '''
    INPUT 
        df - a dataframe schema name
        col_list - a Python list with all column names on which count has to be done 
        op_cat_colname - category column name
        op_num_colname - count column name (should be int)
    OUTPUT
        df_result - a sorted data farme with each category name and count of the columns
    '''
    result_dict = {}
    for each_col in col_list:
        result_dict[each_col] = df[each_col].sum()
    
    df_result = pd.DataFrame(pd.Series(result_dict)).reset_index()
    df_result.columns = [op_cat_colname, op_num_colname]
    return df_result.sort_values(op_num_colname, ascending=False)

def return_graphs(df):
    """Creates  plotly visualizations
    Args:
        df - Schema name 
    Returns:
        list (dict): list containing the four plotly visualizations

    """
    col_list=['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']
    
    # Get rows with weather_related only
    weather_related = ( df["weather_related"] == 1)
    df_weather = df[weather_related]
    
    #Get dataframe with weather related messages only
    weather_data =  get_categ_cnt(df_weather,col_list,'Weather_Category','count') 
    weather_cat = weather_data['Weather_Category'].tolist()
    count = weather_data['count'].tolist()
    
    #Get weather related death only
    weather_related_death = ( df["death"] == 1 ) & ( df["weather_related"]==1 )
    df_new = df[weather_related_death]
    weather_death =  get_categ_cnt(df_new,col_list,'Weather_Category','Death_count') 
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    graphs = [
        {
            'data': [
                Bar(
                    x=weather_death.Weather_Category.tolist(),
                    y=weather_death.Death_count.tolist()
                )
            ],

            'layout': {
                'title': 'Bar chart to show death counts for various Weather Conditions ',
                'yaxis': {
                    'title': "Death Count"
                },
                'xaxis': {
                    'title': "Weather Conditions"
                }
            }
        }
        , {
            'data': [
                Pie(
                labels = weather_cat, 
                values = count
                )
            ],

            'layout': {
                'title': 'Pie Chart to show break down by Various Weather related conditions'
            }
        },
        {
            'data': [
                Pie(
                labels = genre_names, 
                values = genre_counts
                )
            ],

            'layout': {
                'title': 'Pie Chart to show message Break-down by Genre'
            }
        }
    ]

    return graphs
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")
print("Model loaded")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # create visuals
    graphs = return_graphs(df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
