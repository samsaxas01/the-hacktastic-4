from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Dashboard page
@app.route("/")
@app.route("/home")
def home():
    return render_template("dashboard.html")

@app.route("/queries")
def queries():
    return render_template("queries.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return redirect(url_for('index'))

# Other endpoints
@app.route("/login")
def login():
    return "Login successfully...."

@app.route("/logout")
def logout():
    return redirect(url_for("home"))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('choose_action', filename=file.filename))

@app.route('/choose_action/<filename>')
def choose_action(filename):
    return render_template('choose_action.html', filename=filename)

@app.route('/view_statistics/<filename>')
def view_statistics(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    stats = data.describe(include='all').to_dict()
    return render_template('view_statistics.html', stats=stats)

@app.route('/visualize/<filename>', methods=['GET', 'POST'])
def visualize(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    attributes = data.columns.tolist()

    if request.method == 'POST':
        x_column = request.form.get('xColumn')
        y_column = request.form.get('yColumn')
        graph_type = request.form.get('graphType')

        plt.figure(figsize=(10, 6))

        if graph_type == 'scatter':
            sns.scatterplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'line':
            sns.lineplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'bar':
            sns.barplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'box':
            sns.boxplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'stacked':
            data.groupby(x_column)[y_column].sum().plot(kind='bar', stacked=True)

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{x_column} vs {y_column}')

        # Automatically adjust graph size if too many rows
        num_rows = len(data)
        if num_rows > 1000:
            plt.gcf().set_size_inches(12, 8)
        else:
            plt.gcf().set_size_inches(10, 6)

        # Save to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('visualize.html', 
                               attributes=attributes,
                               image=f"data:image/png;base64,{img_base64}",
                               selected_x=x_column,
                               selected_y=y_column,
                               selected_graph_type=graph_type)

    return render_template('visualize.html', attributes=attributes)

@app.route('/get_columns')
def get_columns():
    filename = 'sentiment.csv'  # Replace with the actual filename or handle it dynamically
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    data = pd.read_csv(filepath)
    columns = data.columns.tolist()
    return jsonify(columns)

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    try:
        # Load dataset
        data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv'))

        # Get data from request
        x_column = request.json.get('xColumn')
        y_column = request.json.get('yColumn')
        graph_type = request.json.get('graphType')

        if x_column not in data.columns or y_column not in data.columns:
            return jsonify({'error': 'Invalid column names'}), 400

        plt.figure(figsize=(10, 6))

        # Generate graph based on type
        if graph_type == 'scatter':
            sns.scatterplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'line':
            sns.lineplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'bar':
            sns.barplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'box':
            sns.boxplot(data=data, x=x_column, y=y_column)
        elif graph_type == 'stacked':
            data.groupby(x_column)[y_column].sum().plot(kind='bar', stacked=True)
        else:
            return jsonify({'error': 'Invalid graph type'}), 400

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{x_column} vs {y_column}')

        # Automatically adjust graph size if too many rows
        num_rows = len(data)
        if num_rows > 1000:
            plt.gcf().set_size_inches(12, 8)
        else:
            plt.gcf().set_size_inches(10, 6)

        # Save to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

        return jsonify({'image': img_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/query1")
def query1():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)
    sentiment_counts = df['Sentiment'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Distribution of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Distribution of Sentiments")

@app.route("/query2")
def query2():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Platform', hue='Sentiment', data=df, palette='Set2')
    plt.title('Sentiment Distribution Across Platforms')
    plt.xlabel('Platform')
    plt.ylabel('Count')
    #plt.legend(title='Sentiment')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Sentiment Variation Across Platforms")

@app.route("/query3")
def query3():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    hashtags = []
    for tags in df['Hashtags']:
        hashtags.extend(tags.split())
    hashtag_counts = Counter(hashtags).most_common(5)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[tag[0] for tag in hashtag_counts], y=[tag[1] for tag in hashtag_counts], palette='pastel')
    plt.title('Top 5 Most Used Hashtags')
    plt.xlabel('Hashtag')
    plt.ylabel('Count')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Top 5 Most Used Hashtags")

@app.route("/query4")
def query4():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    df['Month'] = pd.Categorical(df['Month'], categories=range(1, 13), ordered=True)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Month', hue='Sentiment', data=df, palette='coolwarm')
    plt.title('Sentiment Variation Across Months')
    plt.xlabel('Month')
    plt.ylabel('Count')
    #plt.legend(title='Sentiment')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Sentiment Change Over Months")

@app.route("/query5")
def query5():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Sentiment', y='Favorites', data=df, palette='rainbow')
    plt.title('Distribution of Favorites Across Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Favorites')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Distribution of Favorites Across Sentiments")

@app.route("/query6")
def query6():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    df.rename(columns=lambda x: x.strip(), inplace=True)

    sentiment_stats = df.groupby('Sentiment')[['Retweets', 'Likes']].mean()

    plt.figure(figsize=(10, 6))
    sentiment_stats.plot(kind='bar', colormap='Paired')
    plt.title('Average Retweets and Likes by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Count')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Retweets', y='Likes', data=df, hue='Sentiment', palette='Set2')
    plt.title('Correlation between Retweets and Likes')
    plt.xlabel('Retweets')
    plt.ylabel('Likes')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64_scatter = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', 
                           image=f"data:image/png;base64,{img_base64}", 
                           scatter_image=f"data:image/png;base64,{img_base64_scatter}",
                           title="Correlation Between Retweets and Likes")

@app.route("/query7")
def query7():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    daily_counts = df['Day'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=daily_counts.index, y=daily_counts.values, marker='o')
    plt.title('Daily Distribution of Posts')
    plt.xlabel('Day')
    plt.ylabel('Count')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Daily Distribution of Posts")

@app.route("/query8")
def query8():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Country', hue='Sentiment', data=df, palette='Set1')
    plt.title('Sentiment Distribution Across Countries')
    plt.xlabel('Country')
    plt.ylabel('Count')
    #plt.legend(title='Sentiment')
    plt.xticks(rotation=45)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Sentiment Variation by Country")

@app.route("/query9")
def query9():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    sns.histplot(x='Hour', data=df, hue='Platform', multiple='stack', bins=24, palette='Set3')
    plt.title('Hourly Distribution of Posts by Platform')
    plt.xlabel('Hour')
    plt.ylabel('Count')
    #plt.legend(title='Platform')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Peak Hours for Posting")

@app.route("/query10")
def query10():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.csv')
    df = pd.read_csv(filepath)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['DayOfWeek'] = df['Timestamp'].dt.day_name()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='DayOfWeek', hue='Sentiment', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='coolwarm')
    plt.title('Sentiment Distribution Across Days of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    #plt.legend(title='Sentiment')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('query.html', image=f"data:image/png;base64,{img_base64}", title="Sentiment Distribution Across Days of the Week")

if __name__ == '__main__':
    app.run(debug=True)