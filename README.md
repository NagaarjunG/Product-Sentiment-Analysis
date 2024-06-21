# Product-Sentiment-Analysis  AND  WEB Scarpping from online platform


# Overview
Product sentiment analysis is crucial for understanding customer feedback and making data-driven decisions. This app analyzes sentiment using natural language processing (NLP) techniques and provides insights into the sentiments expressed in product reviews.

# Features
Sentiment Analysis: Analyzes the sentiment (positive, negative, neutral) of product reviews using TextBlob.
Comparison: Compares sentiment scores across multiple products based on user queries.
Visualization: Includes a bar chart visualization of average polarity scores for each product, aiding in visual comparison of sentiment.


# Usage
Enter Your Query:

Type a query related to product features or reviews in the text input provided.

# View Sentiment Analysis Results:

The app will display sentiment analysis results for different products based on your query.
It highlights the product with the most relevant sentiment based on your query.

#Visualization (Optional):

A bar chart shows the average polarity scores of products based on their reviews.
It helps visualize which products have more positive or negative sentiments.

# File Structure
app.py: Main Python script containing the Streamlit application code.
requirements.txt: List of Python dependencies.
data/: Directory containing CSV files of product reviews.
images/: Directory for storing images used in the Streamlit app.

# Additional Notes

Error Handling: The app includes robust error handling for file loading, data processing, and sentiment analysis.
Scalability: It can handle multiple CSV files and adapt to different product review datasets.
Customization: Feel free to customize the app's thresholds, visualizations, or sentiment interpretation logic based on your needs.

# Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements...feel free to reach out me!!!




# This Python script uses Selenium to scrape product reviews from online platform and saves them into a CSV file.

# Overview
The script automates the process of extracting reviews, ratings, likes, and dislikes for a given product URL on online platform. It iterates through multiple pages of reviews, clicks on "READ MORE" links to expand review texts, and handles pagination to collect all available data.

# Requirements

Python 3.x
Selenium
pandas
numpy
Chrome WebDriver
Ensure you have installed the necessary Python packages listed in requirements.txt. Also, download the Chrome WebDriver and adjust the path in the script accordingly.


# Output

The script will generate a CSV file named <product_name>_reviews.csv containing extracted reviews, ratings, likes, and dislikes.

# Script Details

WebDriver Initialization: Uses Chrome WebDriver to automate browsing.
Data Extraction: Extracts review texts, ratings, likes, and dislikes from Flipkart product pages.
Data Cleaning: Cleans special characters and handles missing data using pandas and numpy.

# Notes
Adjust the time.sleep() intervals as needed based on network speed and page loading times.
Ensure compliance with Flipkart's terms of service and use the script responsibly.

# Example usage

# product_url = "https://www.online.com/product-reviews/product-id"
# product_name = "Product Name"
# scrape_flipkart_reviews(product_url, product_name)





