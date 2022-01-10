# Crypto-prediction


<h2 align=center> Predicting Cryptocurrency Price Movements Using Twitter Sentiment Analysis and Tree-Based Algorithms</h1>

<p align=center>Tem Cavanagh</p>

<p align=center><i>University of Newcastle, Australia</i></p>

<p align=center> <b>Abstract.</b> Increasingly, the use of sentiment analysis derived from social media platforms such as Twitter has been used for the prediction of movements in financial markets. The rise of cryptocurrencies as an emerging class of financial instruments and their unique market characteristics presents an interesting intersectionality for research into this area of computing and data science. This paper presents a predictive model for forecasting the hourly Bitcoin price movements using tree-based algorithms and sentiment analysis extracted from Twitter data. A review of the relevant literature and related works which this project expands upon is presented, in addition to an analysis of relevant data science techniques. The resulting predictive model utilises an XGBoost gradient boosted random forest model which demonstrates a 63.16% accuracy in predicting the hourly price movements of Bitcoin which represents an incremental improvement on the most comparable work of this nature.</p>

<p align=center> <b>Keywords:</b> Sentiment analysis, Decision trees, Machine learning.</p>

<br>

<h3 align=center>1. Introduction</h3>

Whether it is possible to accurately predict future asset prices has been long debated. Traditional financial theory posits that asset prices are reflections of all available information for such assets, this concept is known as the efficient-market hypothesis. [1]
 Kahneman and Tversky (1979) have presented the &#39;prospect theory&#39; which holds that financial decisions of individuals are heavily influenced by the perceived risks and emotions of individual and not economic value alone. [2]
 The impact of public sentiment on asset prices, particularly since the advent of social media platforms, has been analysed considerably which has given rise to significant literature concerning the prediction of asset prices using sentiment or opinion.

Due to the large volume of short form content, Twitter has increasingly become a treasure trove for the mining of subjectively opinionated data points. Researchers have regularly used Twitter data to gain valuable information in topics of interest. On this basis, Twitter has become a widely used source for data sets relating to sentiment analysis. [3]

Sentiment analysis involves &#39;&#39;the computational study of people&#39;s opinions, appraisals, attitudes, and emotions toward entities, individuals, issues, events, topics and their attributes&quot;. [4]
 There are a variety of approaches and techniques that can be used in sentiment analysis. [5]
 Ultimately, sentiment analysis seeks to discover and extract subjective information from text such as the sentiment, polarity and subjectivity of the text. Polarity classification attempts to determine the degree to which a statement reflects either a positive or negative sentiment.

A common approach to sentiment analysis and polarity classification is through the lexicon-based approach. The lexicon-based approach uses a lexicon of words and their corresponding sentiment values which are then compared against an input statement. This approach can achieve a high classification accuracy and allows for the creation and use of tailored domain specific lexicons. An example of a lexicon-based sentiment analysis tool is VADER (Valence Aware Dictionary and sEntiment Reasoner) which analyses and computes the sentiment of textual observations and has been tailored specifically for the domain of social media sentiment analysis. [6]

Tree based algorithms are a class of machine learning algorithms which utilise tree-like structures to perform classification and regression tasks. [7]
 Two prominent examples of tree-based algorithms are decision trees and random forests. Gradient boosting utilises ensembles of typically weaker prediction models in order to derive a stronger predictive model. [8]

This report presents a machine learning project that collects and analyses Twitter sentiment data in order to accurately predict the near future movements in the price of Bitcoin. The problem that this project attempts to solve can be summarised as follows:  **&quot;Train a predictive model that utilises near-real time human sentiment analysis in order to maximise the predictive accuracy of future asset price movements.&quot;**

The following section will present a review of the relevant literature and related works that this project attempts to expand upon.

<br>

<h3 align=center>2. Literature review and related works</h3>

Tetlock (2007) demonstrated the influence of pessimistic media to negatively affect stock market prices and the volume of stocks traded. [9]
 The work of Bollen, Pepe, et al. (2011) has provided the rationale for a number of research attempts which are similar to this project proposal. [10]
 Although not specifically related to cryptocurrency markets, Bollen, Pepe, et al. (2011) derived a model for predicting movements in the DOW Jones Industrial average stock market using Twitter sentiment which achieved a remarkable prediction accuracy of 86.7%. In their model, Bollen, Pepe, et al. (2011) employed the use of regression modelling and neural networks in order to determine the predictive power of Twitter sentiment. Further, it was demonstrated that the predictive power of Twitter sentiment was observed as being the strongest within 1 to 4 days following the making of such predictions. This work strengthened the idea that financial markets are impacted by public sentiment.

Nguyen et al. (2015) analysed the sentiment of users of the Yahoo Finance Message Board in order to build a predictive model for stock prices which utilised a linear kernel Support Vector Machine (SVM) to perform sentiment analysis and extract topic sentiment from texts. The resulting predictive accuracy of the model demonstrated a 2% increase in predictive capabilities when compared against predictions using historical stock prices alone. [11]

Sul et al. (2014) analysed Twitter posts to determine the cumulative emotional sentiment and corresponding stock prices for the S&amp;P 500. [12]
 The results demonstrated a significant correlation between Twitter sentiment and stock prices for the specific stocks mentioned in the tweets. Additionally, correlative links were shown between the number of followers of a Twitter user and the delayed impact of sentiment on stock prices.

Shah et al. (2018) analysed the impact of news sentiment on the stock market and developed a dictionary-based sentiment analysis model for the financial domain. [13]
 The resulting predictive model achieved an accuracy of 70% when predicting the short-term price movements in stock prices.

Abraham et al. (2018) developed a model which predicted the movements of Bitcoin and Ethereum cryptocurrency prices using a combination of tweet volumes, Twitter sentiment analysis and Google Trends data as input variables to a multiple linear regression model. [14]
 Notably, this work highlights a multivariate approach to the prediction of cryptocurrency prices, and the nature of Twitter sentiment to remain positive despite pricing downtrends due to investors&#39; own interests beyond that of the immediate market price.

Valencia et al. (2019) employed neural networks, support vector machines and random forests in order to develop sentiment analysis models capable of predicting price movements in cryptocurrencies. [15]
 It was demonstrated that a Bitcoin prediction model using Twitter data as an input to a multi-layer perceptron neural network achieved 0.72 accuracy and 0.74 precision scores. This paper reinforced the findings that sentiment analysis through platforms such as Twitter may be used to predict price movements in cryptocurrency markets. However, this paper proposed opportunities for the incorporation of more complex neural network architectures in order to strengthen model performances and develop optimal trading strategies.

Kraaijeveld and De Smedt (2020) have put forward a highly detailed analysis of many of the topics that relate to this proposal particularly in relation to the methodology of such a project. [16]
 Kraaijeveld and De Smedt (2020) expand upon previous literature outlined here and also highlight the importance of data cleaning and preparation for modelling, particularly in relation to the presence of Twitter bot accounts which may distort datasets used when modelling. Additionally, Sattarov et al. (2020) extracted tweets relating to Bitcoin and analysed the respective sentiment scores of those tweets and their correlation to Bitcoin price movements which resulted in a prediction accuracy of 62.48% using a random forest classifier model. [17]

Seif et al. (2018) proposed a real-time stock market predictive model using sentiment analysis and supervised learning. [18]
 Importantly, this project was framed as a big data project due to the required volume and velocity of the data that was used to facilitate a real-time predictive model. The authors made use of big data frameworks such as Apache Spark for parallel processing to test the predictive capabilities of supervised machine learning methods such as logistic regression, random forests and SVMs.

Mohapatra et al. (2019) built a real-time cryptocurrency price prediction model which uses Twitter sentiment in their project called &#39;KryptoOracle&#39;. [19]
 The authors followed a similar approach to building the project that was proposed by Seif et al. (2018) in that the conceptual framework of the project was posed from a big data perspective and Apache Spark was used to handle large volumes of incoming live Twitter and cryptocurrency price data. In this project, which used structured and tabulated data, the superior performance of tree-based machine learning algorithms was demonstrated. The authors utilised XGBoost which is a gradient boosting library for tree-based algorithms that is designed to be highly efficient and flexible particularly in relation structured and tabulated data. [20]
 Moreover, Ibrahim (2021) has demonstrated the superior performance of gradient boosted models in predicting Bitcoin market movements when compared to other machine learning methods. [21]

From the relevant literature and related works that have been discussed above, the topic area in this project presents an opportunity for further research to be done. The following section presents the detailed methodology used to complete this project.

<br>

<h3 align=center>3. Methodology</h3>

As an illustrative example, Figure 1 below provides a simple overview of the project architecture and methodology that was used in developing this project.

![image](https://user-images.githubusercontent.com/50828923/148835471-0d1e8e68-ab3b-429b-bb63-643c3f246520.png)

**Figure 1** : Project overview

  **3.1 Data collection and processing**

A key requirement for the project was the collection of near real-time Twitter and Bitcoin price data. To collect the required data, python scripts were written which collect 1,000 new tweets relating to Bitcoin and the price of Bitcoin on a continual hourly basis. Access to the Twitter API was made possible by the Tweepy python library and the Yahoo Finance python library was used collect to Bitcoin price data. [22]

The Tweepy library itself enables a degree of filtering and selection when extracting Twitter data, however further data cleaning and pre-processing was necessary before the data was passed to modelling.Ultimately, data cleaning and pre-processing attempted to minimise the inclusion of any observations produced by Twitter bots, and those which do not reflect individual users&#39; sentiments before the data is passed for sentiment analysis. To that end, the data cleaning steps included the removal of hashtags (#), the removal of user tag symbols (@), the removal of websites and external links, the removal of instances containing keywords which may be associated with spam and the condensing of multi-line tweets.

The Bitcoin price data required minimal cleaning however additional features were computed and added to the data in the form of hourly price change and a binary classification of hourly price movement (+1/-1) for future use in prediction modelling.

  **3.2 Sentiment analysis**

The project utilised the VADER sentiment analysis library due to its computational efficiency. Given the large volume of tweets that were extracted for analysis in this project, the efficiency of the VADER library was paramount. The resulting VADER sentiment compound was used to compute a sentiment score in conjunction with the features of &#39;Likes&#39;, &#39;Followers&#39; and &#39;Retweets&#39; as input variables for each tweet, as demonstrated by the equation below:

![image](https://user-images.githubusercontent.com/50828923/148835561-2928dcf6-a82c-4507-b9c7-cecba9ed99f8.png)


The equation above attempts to quantify the impact which each individual tweet may have on overall sentiment by multiplying the VADER compound by the user reactions. Additionally, where a user has no followers, as is the case with Twitter bots, the equation effectively removes that sentiment observation from further analysis by returning a sentiment score of zero.

  **3.3 Feature engineering**

Following the extraction, cleaning and sentiment analysis of the Twitter and Bitcoin price data, the datasets were merged. To merge the Twitter and Bitcoin price datasets, the Twitter dataset was first grouped into hourly aggregations of observations, the result from grouping of the Twitter dataset was the creation of additional features where the sums of the individual hourly groupings have become features along with sentiment compound and sentiment score movement classifications.

Additional features which were drawn from existing features were added to the dataset. These features included features relating to sentiment compound and sentiment score movements and the binary feature of hourly price movements which was the target feature for prediction in this project.

  **3.4 Predictive modelling**

The project compared tree-based algorithms when building a predictive model. There are six models which were tested and compared in order to arrive at a final predictive model. These six models are as follows:

1. Scikit Learn Decision Tree Classifier. [23]
2. Scikit Learn Random Forest Classifier. [24]
3. Scikit Learn Extra Trees Classifier. [25]
4. Scikit Learn Gradient Boosting Classifier. [26]
5. XGBoost Classifier. [27]
6. XGBoost Random Forest Classifier. [28]

Initially, each model was tested using the default parameters of the individual model and all of the input features were used as predictors.

**Feature selection**

After the initial modelling stage, feature selection was conducted in order to reduce the number of input features used by the models in an attempt to increase the predictive capabilities of each model. Additionally, the feature selection stage ensured that the models used only those features which were most relevant to the sentiment that was derived from the collected data.

**Hyperparameter Tuning**

The final stage of predictive modelling involved the hyperparameter tuning of the best performing model when using the selected features as inputs. This stage involved a degree of experimentation in order to optimise the hyperparameters of the selected model in order to increase the predictive power of the selected model.

  **3.5 Predictive performance evaluation**

In order to evaluate the performance of the predictive model in this binary classification problem, the dataset was split into training and test sets at a 90/10 train/test split ratio. All models in this project were trained and tested using this split dataset. The results of each of the predictive models were then evaluated using the respective classification report and confusion matrix of each model when applied to the test set.

The following section presents the project results which have been observed as outcomes from the discussed methodology.

<br>

<h3 align=center>4. Project results</h3>

The project used Jupyter notebooks with Python 3.8 kernels to build the source code for this project. Cloud services were used to ensure the hourly execution of the relevant data collection and data cleaning scripts. The steps taken to compute sentiment scores of collected tweets, merge the collected Twitter and Bitcoin price datasets, and make predictions were carried out on a local CPU machine. The following subsections will provide detailed project results.

  **4.1 Data collection and processing**

Twitter data consisting of 1,000 new and unique tweets was collected each hour for a period of approximately 33 days. The newly collected tweets were cleaned each hour in accordance with the data processing steps set out in the methodology. At the conclusion of the data collection period, a total of 635,827 tweets had been collected and cleaned. Additionally, a total of 793 hourly price observations were collected for Bitcoin, where the observed price of Bitcoin has ranged significantly from approximately USD$40,000 to over USD$60,000. Figure 2, below, shows the number of daily tweets that have been collected and stored for modelling.

![image](https://user-images.githubusercontent.com/50828923/148835622-e9d73253-5cb8-47eb-ab35-c05761db5b6c.png)

**Figure 2:** Number of Tweets collected each day

F ![](RackMultipart20220107-4-1fyuken_html_d7227341a0ef24d4.png)
 ollowing the collection of the relevant Twitter and Bitcoin price data, sentiment analysis was carried out in accordance with the sentiment analysis methodology which was previously outlined in Section 3.1. Exploratory analysis of the collected dataset was carried out as demonstrated by Figure 3, below, which illustrates a comparison of the changes between the hourly sentiment score and Bitcoin price as the data collection for this project progressed.

![image](https://user-images.githubusercontent.com/50828923/148835683-5cb9d2e8-47ea-4fe8-96b4-5aef1e3b7e56.png)

**Figure 3:** Comparison of hourly Bitcoin price and Twitter sentiment score

At the conclusion of the data collection and processing stages, the resulting dataset which was used for predictive modelling contains a total of 752 hourly observations which consist of the features that are shown in Table 1.

**Table 1:** Features used for predictive modelling

| **Feature** | **Description** |
| --- | --- |
| Tweets | Count of hourly tweets collected |
| Likes | Sum of likes received on collected tweets |
| Retweets | Sum of retweets received on collected tweets |
| Compound | Sum of VADER sentiment compound scores |
| Score | Sum of sentiment scores |
| Compound change | Compound score change from the previous hour |
| Score change | Sentiment score change from the previous hour |
| Compound movement | Binary label associated with positive/negative compound change |
| Score movement | Binary label associated with positive/negative score change |
| Open | The hourly opening market price of Bitcoin |
| Movement (Target) | Binary label associated with the hourly Bitcoin price movement |

  **4.2 Predictive modelling**

The initial predictive modelling stage utilised the features outlined in Table 1 as feature inputs to build several tree-based predictive models for predicting the hourly movement of Bitcoin price. The results of these models are shown below in Table 2.

**Table 2:** Precision, Recall, F1-scores of full featured tree-based predictive models

| **Model** | **Precision** | **Recall** | **F1-score** |
| --- | --- | --- | --- |
| Decision Tree Classifier | 0.50 | 0.50 | 0.50 |
| Random Forest Classifier | 0.51 | 0.51 | 0.51 |
| Extra Trees Classifier | 0.47 | 0.47 | 0.47 |
| Gradient Boosting Classifier | 0.48 | 0.49 | 0.48 |
| XGBoost Classifier | 0.55 | 0.55 | 0.55 |
| XGBoost Random Forest | 0.55 | 0.54 | 0.53 |

Following the initial stage of predictive modelling in which all features within the dataset were used, a further round of modelling was carried out where only those features which relate to sentiment analysis were selected. The selected features were &#39;Compound&#39;, &#39;Score&#39;, &#39;Compound change&#39; and &#39;Score change&#39;. The results of these models using the selected features are shown in Table 3.

**Table 3:** Precision, Recall, F1-scores of select featured tree-based predictive models

| **Model** | **Precision** | **Recall** | **F1-score** |
| --- | --- | --- | --- |
| Decision Tree Classifier | 0.48 | 0.49 | 0.48 |
| Random Forest Classifier | 0.53 | 0.53 | 0.52 |
| Extra Trees Classifier | 0.57 | 0.57 | 0.57 |
| Gradient Boosting Classifier | 0.60 | 0.59 | 0.59 |
| XGBoost Classifier | 0.55 | 0.55 | 0.55 |
| XGBoost Random Forest | 0.61 | 0.61 | 0.61 |

The resulting performance metrics of the predictive models which use only the selected features have indicated that the model with the best performance is the XGBoost Random Forrest classifier. This model was then selected for hyperparameter tuning in order to test whether the predictive performance of the model could be improved.

The experimentation with the hyperparameter tuning of the XGBoost Random Forest classifier model resulted in the following optimised parameters:

XGBRFClassifier(n\_estimators=200, subsample=0.9, use\_label\_encoder=False)

The predictive performance of the tuned XGBoost Random Forest classifier is presented by the classification report below in Table 4 along with the confusion matrix in Figure 4 and the predictive model&#39;s feature importances in Figure 5.

**Table 4:** XGBRF Classification Report

|| Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| 0 | 0.63 | 0.59 | 0.61 | 37 |
| 1 | 0.63 | 0.67 | 0.65 | 39 |
| Accuracy | | | 0.63 | 76 |
| Macro average | 0.63 | 0.63 | 0.63 | 76 |
| Weighted average | 0.63 | 0.63 | 0.63 | 76 |

![image](https://user-images.githubusercontent.com/50828923/148835786-dc42db82-55e8-4520-8c0c-63a74b37fd79.png)

**Figure 4:** XGBRF Confusion Matrix 

![image](https://user-images.githubusercontent.com/50828923/148835827-aed45fe5-7abf-4b56-8004-705e1db60191.png)

**Figure 5:** XGBRF Feature Importances

<h3 align=center>5. Discussion</h3>

This project has implemented an automated method for the collection of updated Twitter data and Bitcoin price data in order to build a predictive model through experimentation. The results of the experimentation and comparison using tree-based predictive models has demonstrated the superior performance of ensemble methods and gradient boosted models when assigned to the predictive task in this project. Using the default model hyperparameters, the Scikit Learn Extra Trees ensemble method and the gradient boosted models, which consist of a Scikit Learn Gradient Boosting Classifier, XGBoost Classifier and XGBoost Random Forest Classifier demonstrated superior predictive performances when compared to the more non-gradient boosted tree-based classification models of decision trees and random forests. This was true for both experimentations conducted using the full-featured dataset and the selected feature dataset. The selection and hyperparameter tuning of the XGBoost Random Forest Classifier in order to build a final predictive model for this project has resulted in a model which demonstrates a predictive accuracy of 63.16% as shown by Table 4 and Figure 4.

The results observed in this project fall short of the predictive accuracy of approximately 89% that was observed by Ibrahim (2021). An important distinction must be drawn between the current project and the work of Ibrahim (2021) in that the latter work did not utilise near real-time extracted Twitter data and instead utilised a previously published dataset. Similarly, though the work of Ibrahim (2021) and the results presented here confirm the superior predictive power of gradient boosted models such as XGBoost in a task such as this. Moreover, Ibrahim (2021) utilised a &#39;Composite Ensemble Prediction Model&#39; (CEPM) which utilised XGBoost as the basis of the CEPM model which leaves open the possibility for further research into the potential of a similar CEPM model being utilised in this project.

Comparatively, the results of this project demonstrate an incremental improvement in predictive capabilities when compared to the work of Sattarov et al. (2020) which observed a predictive accuracy 62.48% when utilising Twitter sentiment and random forests to forecast Bitcoin price. Arguably, the work of Sattarov et al. (2020) is the most comparable to the work carried out in this project and as such, the incremental improvement in predictive performance in that regard is important. Moreover, the improvement in predictive accuracy when gradient boosted models are utilised is an area for further research, particularly in relation to the potential for performance improvement offered by the hyperparameter tuning of such models and in relation to the exploration and engineering of additional sentiment features that may be used as inputs for predictive modelling.

<br>

<h3 align=center>6. Conclusion</h3>

This project has explored the potential of building a machine learning model that collects and analyses Twitter sentiment data to accurately predict the near future movements in the price of Bitcoin. The problem that this project has attempted to solve can be summarised as follows: _ **&quot;Train a predictive model that utilises near-real time human sentiment analysis in order to maximise the predictive accuracy of future asset price movements.&quot;** _

This project represents the culmination of work towards the completion of the Master of Data Science program and as such, an additional goal of this project was to utilise the specific knowledge and skills in the domain of data science gained throughout this program. To that end, the project has followed a similar design to the data science pipeline whereby the processes of data collection, data cleaning and analysis, model selection, feature engineering, hyperparameter tuning, and performance evaluation were used throughout this project.

The predictive accuracy of the final XGBoost Random Forest Classifier is 63.16% which represents an incremental improvement on the most comparable work of this nature. As future research, further feature engineering may improve input features into the predictive model and so too may hyperparameter tuning or ensemble modelling increase the predictive accuracy.

 The source code for this project is available online through GitHub: [**github.com/temcavanagh/Crypto-prediction**](http://www.github.com/temcavanagh/Crypto-prediction)

<br>

<h3 align=center>References</h3>

[1](#sdendnote1anc) Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. _The Journal Of Finance, 25_(2), 383. https://doi.org/10.2307/2325486

[2](#sdendnote2anc) Kahneman, D., &amp; Tversky, A. (1979) Prospect Theory: An Analysis of Decision Under Risk. _Econometrica_, _47_(2), 263. https://doi.org/10.2307/1914185

[3](#sdendnote3anc) Bollen, J., Mao, H., &amp; Zeng, X. (2011). Twitter mood predicts the stock market. _Journal Of Computational Science_, _2_(1), 1-8. https://doi.org/10.1016/j.jocs.2010.12.007

[4](#sdendnote4anc) Liu B., Zhang L. (2012) A Survey of Opinion Mining and Sentiment Analysis. In: Aggarwal C., Zhai C. (eds) Mining Text Data. Springer, Boston, MA. https://doi.org/10.1007/978-1-4614-3223-4\_13

[5](#sdendnote5anc) Feldman, R. (2013). Techniques and applications for sentiment analysis. _Communications Of The ACM_, _56_(4), 82-89. https://doi.org/10.1145/2436256.2436274

[6](#sdendnote6anc) Hutto, C., &amp; Gilbert, E. (2014). Vader: A parsimonious rule-based model for sentiment analysis of social media text. _Proceedings Of The International AAAI Conference On Web And Social Media_, _8_(1), 216-225. https://ojs.aaai.org/index.php/ICWSM/article/view/14550

[7](#sdendnote7anc) Géron, A. (2019). _Hands-on machine learning with Scikit-Learn and TensorFlow_ (2nd ed., p. 175). O&#39;Reilly Media.

[8](#sdendnote8anc) Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). _The Elements of Statistical Learning_ (pp. 337–384). Springer.

[9](#sdendnote9anc) Tetlock, P. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market. _The Journal Of Finance_, _62_(3), 1139-1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

[10](#sdendnote10anc) Bollen, J., Mao, H., &amp; Pepe, A. (2011). Modeling public mood and emotion: Twitter sentiment and socio- economic phenomena. _Proceedings Of The International AAAI Conference On Web And Social Media_, _5_(1). https://arxiv.org/abs/0911.1583

[11](#sdendnote11anc) Nguyen, T., Shirai, K., &amp; Velcin, J. (2015). Sentiment analysis on social media for stock movement prediction. _Expert Systems With Applications_, _42_(24), 9603-9611. https://doi.org/10.1016/j.eswa.2015.07.052

[12](#sdendnote12anc) Sul, H., Dennis, A., &amp; Yuan, L. (2014). Trading on Twitter: The Financial Information Content of Emotion in Social Media. _2014 47Th Hawaii International Conference On System Sciences_, 806-815. https://doi.org/ 10.1109/hicss.2014.107

[13](#sdendnote13anc) Shah, D., Isah, H., &amp; Zulkernine, F. (2018). Predicting the Effects of News Sentiments on the Stock Market. _2018 IEEE International Conference On Big Data (Big Data)_, 4705-4708. https://doi.org/ 10.1109/bigdata.2018.8621884

[14](#sdendnote14anc) Abraham, J., Higdon, D., Nelson, J., &amp; Ibarra, J. (2018). Cryptocurrency Price Prediction Using Tweet Volumes and Sentiment Analysis. _SMU Data Science Review_, _1_(3). https://scholar.smu.edu/datasciencereview/vol1/iss3/1

[15](#sdendnote15anc) Valencia, F., Gómez-Espinosa, A., &amp; Valdés-Aguirre, B. (2019). Price Movement Prediction of Cryptocurrencies Using Sentiment Analysis and Machine Learning. _Entropy_, _21_(6), 589. https://doi.org/10.3390/ e21060589

[16](#sdendnote16anc) Kraaijeveld, O., &amp; De Smedt, J. (2020). The predictive power of public Twitter sentiment for forecasting cryptocurrency prices. _Journal Of International Financial Markets, Institutions And Money_, _65_, 101188. https://doi.org/10.1016/j.intfin.2020.101188

[17](#sdendnote17anc) Sattarov, O., Jeon, H., Oh, R., &amp; Lee, J. (2020). Forecasting Bitcoin Price Fluctuation by Twitter Sentiment Analysis. _2020 International Conference On Information Science And Communications Technologies_. https://doi.org/10.1109/icisct50599.2020.9351527

[18](#sdendnote18anc) Seif, M., Hamed, E., &amp; Hegazy, A. (2018). Stock Market Real Time Recommender Model Using Apache Spark Framework. _The International Conference On Advanced Machine Learning Technologies And Applications_, 671-683. https://doi.org/10.1007/978-3-319-74690-6\_66

[19](#sdendnote19anc) Mohapatra, S., Ahmed, N., &amp; Alencar, P. (2019). KryptoOracle: A Real-Time Cryptocurrency Price Prediction Platform Using Twitter Sentiments. In _2019 IEEE International Conference on Big Data (Big Data),_ 5544-5551. https://doi.org/10.1109/BigData47090.2019.9006554

[20](#sdendnote20anc) Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings Of The 22Nd ACM SIGKDD International Conference On Knowledge Discovery And Data Mining_, 785-794. https://doi.org/10.1145/2939672.2939785

[21](#sdendnote21anc) Ibrahim, A. (2021). Forecasting the Early Market Movement in Bitcoin Using Twitter&#39;s Sentiment Analysis: An Ensemble-based Prediction Model. _2021 IEEE International IOT, Electronics And Mechatronics Conference_. https://doi.org/10.1109/iemtronics52119.2021.9422647

[22](#sdendnote22anc) _Tweepy Documentation — tweepy 4.1.0 documentation_. Docs.tweepy.org. (2021). Retrieved 17 October 2021, from https://docs.tweepy.org/en/stable/.

[23](#sdendnote23anc) _sklearn.tree.DecisionTreeClassifier_. scikit-learn. (2021). Retrieved 17 October 2021, from https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html.

[24](#sdendnote24anc) _sklearn.ensemble.RandomForestClassifier_. scikit-learn. (2021). Retrieved 17 October 2021, from [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

[25](#sdendnote25anc) _sklearn.ensemble.ExtraTreesClassifier_. scikit-learn. (2021). Retrieved 17 October 2021, from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html.

[26](#sdendnote26anc) _sklearn.ensemble.GradientBoostingClassifier_. scikit-learn. (2021). Retrieved 17 October 2021, from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html.

[27](#sdendnote27anc) _Python API Reference — xgboost 1.6.0-dev documentation_. Xgboost.readthedocs.io. (2021). Retrieved 17 October 2021, from https://xgboost.readthedocs.io/en/latest/python/python\_api.html.

[28](#sdendnote28anc) _Python API Reference — xgboost 1.6.0-dev documentation_. Xgboost.readthedocs.io. (2021). Retrieved 17 October 2021, from https://xgboost.readthedocs.io/en/latest/python/python\_api.html#xgboost.XGBRFClassifier.apply.
