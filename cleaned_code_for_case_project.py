import pandas as pd
import numpy as np
import seaborn as sns
import statistics as sts
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtik
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice
from dateutil.relativedelta import relativedelta

path = r"C:\Users\aroli\Downloads\DsResearch (1)\DsResearch\Media and Technology\Media and Technology\yt.csv"
df = pd.read_csv(path, encoding="latin1")

def interquaratile_and_1stquaratile(x):
    x = sorted(x)
    if(len(x)%2 == 0):
        part1 = x[:int(len(x)/2)]
        part2 = x[int(len(x)/2):]
        Q1 = (x[int(len(part1)/2)] + x[int((len(part1)/2)-1)])/2
        Q3 = (x[int((len(part1)/2)+len(x)/2)]) + x[int((len(part1)/2)-1+len(x)/2)]
    else:
        Q2 = int(len(x)/2)
        part1 = x[:Q2+1]
        part2 = x[Q2:]
        Q1 = x[int(Q2/2)]
        Q3 = x[int((Q2)/2+len(x)/2)]

    IQR = Q3-Q1
    return IQR,Q1

def top_10_youtubers():
    top_10 = [(i,j) for i,j in zip(df["Youtuber"],df["subscribers"])]
    top_10 = top_10[0:11]
    youtubers, subscribers = zip(*top_10)
    x = list(range(len(youtubers)))
    sns.barplot(x=list(i+0.5 for i in x),y= subscribers, palette='Set1')
    plt.xlabel("Youtuber")
    plt.gca().yaxis.set_major_formatter(mtik.FuncFormatter(lambda x,_:f"{int(x):,}"))
    plt.ylabel("No of subscribers")
    plt.xticks(ticks=list(i+0.5 for i in x), labels=youtubers, rotation=90, fontsize=10)
    plt.show()

def top_avg_category():
    category = set(df.dropna(subset=["subscribers"])["category"])
    dictionary = {}
    count = 0
    for i in category:
        df_filtered = df[df["category"]==i].dropna(subset=["subscribers"])
        count = 0
        for j in df_filtered["subscribers"]:
            count += j
        if(len(df_filtered["subscribers"]) == 0):
            continue
        else:
            mean = count/len(df_filtered["subscribers"])
            dictionary[i] = mean
    sorted_arrangement = sorted(dictionary.items(), key=lambda x:x[1])
    print(f"the category with most amount of average subscribers is {sorted_arrangement[-1][0]} with an average of {int(sorted_arrangement[-1][1])} subscribers")


def upload_count():
    df_temp=df[["uploads",'category']].dropna()
    print(df_temp)
    counts_of_occurences=df_temp['category'].value_counts().to_dict()
    dictionary={}
    for i,j in zip(df_temp['category'],df_temp['uploads']):
        dictionary[i]=j
    for i,j in zip(df_temp['category'],df_temp['uploads']):
        dictionary[i]=dictionary.get(i,j)+j
    for category_main,total_upload_counts in dictionary.items():
        for category_internal,no_of_occurences in counts_of_occurences.items():
            if(category_internal==category_main):
                dictionary[category_main]=total_upload_counts/no_of_occurences
    dictionary=dict(sorted(dictionary.items(),key=lambda x:x[1],reverse = True))
    # print(dictionary)
    sns.barplot(x=[j for i,j in dictionary.items()],y=[i for i,j in dictionary.items()],palette='Set2')
    plt.xlabel("average number of videos uploaded in the category")
    plt.ylabel("category")
    plt.title("The visualization of average numbers of youtube videos uploaded in each category")
    plt.show()


def top_country_youtubechannel():
    countries = (df["Country"].dropna())
    dictionary = {}
    for i in countries:
        dictionary[i] = dictionary.get(i,0)+1
    dictionary = dict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))
    # print(f"the country with most number of youtube channels is {dictionary[0][0]} with {dictionary[0][1]} channels.")
    # print(dictionary)
    dictionary=dict(islice(dictionary.items(),5))
    # print(dictionary)
    sns.barplot(x=[i for i,j in dictionary.items()],y=[j for i,j in dictionary.items()],palette='coolwarm')
    plt.xlabel("The top 5 countries with most number of youtube channels")
    plt.ylabel("The number of channels")
    plt.title("Top 5 countries according to the channel count")
    plt.show()

def distribution_of_channels_across_categories():
    dictionary = {}
    category = df["category"].dropna()
    cat = []
    no_channels = []
    for i in category:
        dictionary[i] = dictionary.get(i,0)+1
    for key,values in dictionary.items():
        cat.append(key)
        no_channels.append(values)
    sns.barplot(x=no_channels,y=cat, palette="Set1")
    plt.xticks(rotation=45)
    plt.xlabel("Number of channels")
    plt.ylabel("Categories")
    plt.title("Distribution of number of channels in different categories.")
    plt.show()

def correlation():
    subs = df["subscribers"].dropna()
    views = df["video views"].dropna()
    correlation = subs.corr(views)
    print(f"The correlation between subscribers and number of views is {round(correlation,2)}")

def monthly_earnings_categories():
    categories = df["category"].dropna()
    monthly_min = df["lowest_monthly_earnings"].dropna()
    monthly_max = df["highest_monthly_earnings"].dropna()

    def category_vs_lowest_monthly():
        dictionary = {}
        for i,j in zip(categories,monthly_min):
            dictionary[i] = dictionary.get(i,j)+j
        category = []
        monthly_minimum_dist = []
        for key,values in dictionary.items():
            category.append(key)
            monthly_minimum_dist.append(values)
        sns.barplot(x=category, y=monthly_minimum_dist, palette='Set2')
        plt.xlabel("categories of the youtube channel")
        plt.ylabel("monthly minimum earnings of the channel")
        plt.xticks(ticks=range(len(category)), labels=category, rotation=45)
        plt.title("Analysis of monthly minimum earnings of different categories")
        plt.show()

    def category_vs_highestmonthly():
        dictionary = {}
        for i,j in zip(categories,monthly_max):
            dictionary[i] = dictionary.get(i,j)+j
        category = []
        monthly_max_dist = []
        for key,values in dictionary.items():
            category.append(key)
            monthly_max_dist.append(values)
        sns.barplot(x=category, y=monthly_max_dist, palette='Set2')
        plt.xlabel("categories of the youtube channel")
        plt.ylabel("monthly maximum earnings of the channel")
        plt.xticks(ticks=range(len(category)), labels=category, rotation=45)
        plt.title("Analysis of monthly maximum earnings of different categories")
        plt.show()

    category_vs_lowest_monthly()
    category_vs_highestmonthly()

def trends_subs_gained_in_30_days():
    subs_gained = df["subscribers_for_last_30_days"].fillna(df["subscribers_for_last_30_days"].median())
    category = df["category"].dropna()
    dictionary = {}
    for i,j in zip(category,subs_gained):
        dictionary[i] = dictionary.get(i,j)+j
    sns.barplot(x=[_ for i,_ in dictionary.items()], y=[_ for _,j in dictionary.items()], palette="Set2")
    plt.xlabel("Category of trends in past month")
    plt.ylabel("Number of subscribers gained in each category")
    category = set(category)
    # plt.xticks(ticks=range(len(category)), rotation=45, labels=category)
    plt.title("Trend of subs gained in last 30 days for all the channels")
    plt.show()

def outliers_yearly_earning():
    youtubers = df["Youtuber"].dropna()
    yearly_revenue = (df["highest_yearly_earnings"].dropna())
    iqr, q1 = interquaratile_and_1stquaratile(yearly_revenue)
    print("The outliers in yearly income are")
    for i,j in zip(yearly_revenue,youtubers):
        if(i >= (q1-1.5*iqr) and i <= (q1+1.5*iqr)):
            continue
        else:
            print(f"youtuber {j} with income {i}")

def creation_of_channel_trends():
    df_temp = df[['created_year','created_month','created_date']].dropna().copy()
    df_temp["created_month"] = df_temp["created_month"].apply(lambda x: datetime.strptime(x,"%b").month)
    df_temp['created_year'] = df_temp['created_year'].astype(int)
    df_temp['created_month'] = df_temp['created_month'].astype(int)
    df_temp['created_date'] = df_temp['created_date'].astype(int)
    df_temp = df_temp.rename(columns={"created_year":'year','created_month':'month','created_date':'day'})
    df_temp["date"] = pd.to_datetime(df_temp[['year','month','day']])
    counts = df_temp["year"].value_counts()
    sns.barplot(counts, palette=["#16A085", "#F39C12"])
    plt.xlabel("dates of channel creation")
    plt.ylabel("no of channels created")
    plt.title("Trends of channel creation over years")
    plt.xticks(rotation=45)
    plt.show()

def relation_btw_grosstertiaryedu_n_noofyoutubers():
    df_temp = df[['Country','Gross tertiary education enrollment (%)']].dropna().copy()
    counts = df_temp['Country'].value_counts().to_dict()
    dictionary = df_temp.groupby("Country")["Gross tertiary education enrollment (%)"].mean().to_dict()
    df_stripped = pd.DataFrame(counts.items(), columns=['country','no_of_youtube_channels'])
    df_stripped['tertinary_education'] = df_stripped['country'].map(dictionary)
    correlation = df_stripped[['tertinary_education','no_of_youtube_channels']].corr()
    print(f"The correlation between gross tertiary education and number of youtube channels in the country is {round(correlation.loc['tertinary_education','no_of_youtube_channels'],2)}")
    sns.heatmap(correlation, annot=True, cmap='Spectral')
    plt.title("The relation between Gross tertiary education enrollment (%) and number of channels in the country")
    plt.show()

def Unemployment_vs_NoOfYoutubeChannels():
    df_temp = df[['Country','Unemployment rate']].dropna().copy()
    country_counts = df_temp['Country'].value_counts().to_dict()
    df_stripped = pd.DataFrame(country_counts.items(), columns=['country','no of youtube channels'])
    df_stripped['unemployment'] = df_stripped['country'].map(df_temp.groupby('Country')['Unemployment rate'].mean().to_dict())
    df_stripped_mini = df_stripped.head(10)
    sns.barplot(x=df_stripped_mini['country'], y=df_stripped_mini['unemployment'], palette='Set1')
    plt.xlabel('Top 10 countries with most youtube channels')
    plt.ylabel('Unemployment rate')
    plt.title('Variation of top 10 countries with most channels and unemployment rate')
    plt.show()

def avg_urban_population_in_counties_with_most_channels():
    df_temp = df[["Urban_population",'Country of origin']].dropna().copy()
    df_temp['Urban_population'] = df_temp['Urban_population'].astype(int)
    dictionary = df_temp.groupby('Country of origin')['Urban_population'].mean().to_dict()
    dictionary = dict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))
    total_population = 0
    for i,j in dictionary.items():
        total_population += j
    for i,j in dictionary.items():
        dictionary[i] = dictionary.get(i,j)+j
    for i,j in dictionary.items():
        dictionary[i] = (j/total_population)*100
    df_stripped = pd.DataFrame(dictionary.items(), columns=['country','population percentage'])
    sns.barplot(x=df_stripped['population percentage'], y=df_stripped['country'], palette='Set2')
    plt.xlabel("Countries arranged by population")
    plt.ylabel("Population percentage occupied by country")
    plt.title("The countries vs total populaton percentage")
    
    plt.show()

def latitude_longitude_number_of_channels_pattern():
    df_temp = df[['Country of origin','Latitude','Longitude']].dropna()
    country_occurences = df_temp['Country of origin'].value_counts().to_dict()
    df_temp['coords'] = df_temp[['Latitude','Longitude']].values.tolist()
    dictionary1 = df_temp.groupby('Country of origin')['coords'].first().to_dict()
    dictionary_main = {}
    for i,_ in dictionary1.items():
        for j,no_of_occurences in country_occurences.items():
            if(i == j):
                dictionary_main[tuple(_)] = no_of_occurences
    coords = [values for values in dictionary_main.values()]
    latitude = [values[0] for values in dictionary_main.keys()]
    longitude = [values[1] for values in dictionary_main.keys()]
    no_of_occurences = [key for key in dictionary_main.values()]
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs=latitude, ys=longitude, zs=no_of_occurences, c=no_of_occurences, cmap='coolwarm', s=100, alpha=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label("The color pattern quantifies the number of youtube channels in the specified co-ordinates")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_zlabel("No of youtube channels")
    ax.set_title("Analysis of the trends of no of youtube channels in different coordinates")
    plt.show()
    
def corelation_btw_subscribers_and_population():
    df_temp = pd.DataFrame()
    df_temp['subs'] = df["subscribers"].fillna(df["subscribers"].mean())
    df_temp['population'] = df['Population'].fillna(df['Population'].median())
    correlation = df_temp[['subs','population']].corr()
    if(correlation.loc['subs','population'] == 0):
        print("There is no relation between the attributes")
    elif(correlation.loc['subs','population'] > 0 and correlation.loc['subs','population'] < 0.5):
        print("There is significantly very weak correlation")
    elif(correlation.loc['subs','population'] >= 0.5 and correlation.loc['subs','population'] <= 1):
        print("There is very strong correlation")
    else:
        print("There is negative correlation")

    sns.heatmap(correlation, cmap='coolwarm', annot=True)
    plt.show()

def No_of_channels_vs_population():
    df_temp = df[['Country of origin','Population']].dropna()
    no_channels = df_temp['Country of origin'].value_counts().to_dict()
    country_population = {key:value for key,value in zip(df_temp['Country of origin'],df_temp['Population']) if key in no_channels.keys()}
    country_population = dict(islice(country_population.items(),10))
    trimmed_channelslist = {k:v for k,v in no_channels.items() if k in country_population.keys()}
    trimmed_channelslist = dict(sorted(trimmed_channelslist.items(), key=lambda x:x[1], reverse=True))
    new_dic = {}
    print(country_population)
    print(trimmed_channelslist)
    for key in trimmed_channelslist.keys():
        for i,j in country_population.items():
            if(key == i):
                new_dic[i] = j
    # print(new_dic)
    # new_dic=dict(sorted(new_dic.items(),key=lambda x:x[1],reverse=True))
    sns.barplot(x=[i for i,j in trimmed_channelslist.items()], y=[j for i,j in new_dic.items()], palette='Set2')
    plt.xlabel("Top ten countries based on number of youtube channels")
    plt.ylabel("Population of the country")
    plt.title("The comparison of counties with top 10 youtube channels and their population")
    plt.show()
    
def no_of_subs_vs_unemployment_rate():
    df_temp = pd.DataFrame()
    df_temp["subs_gained"] = df["subscribers_for_last_30_days"].fillna(df["subscribers_for_last_30_days"].median())
    df_temp["unemployment"] = df["Unemployment rate"].fillna(df["Unemployment rate"].median())
    correlation = df_temp[['subs_gained','unemployment']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.show()
    print(f"the correlation between the unemployment rate and no of subs gained in last 30 days is {round(correlation.iloc[0,1],2)}")

def video_views_in_last_30_days_in_categories():
    df_temp = pd.DataFrame()
    df_temp['views'] = df["video_views_for_the_last_30_days"].dropna()
    df_temp["category"] = df['category'].dropna()
    dictionary = {}
    for i,j in zip(df_temp["category"], df_temp["views"]):
        dictionary[i] = dictionary.get(i,j)+j
    print(dictionary)
    dictionary = dict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))
    sns.barplot(x=[j for i,j in dictionary.items()], y=[i for i,j in dictionary.items()], palette='coolwarm')
    plt.ylabel("The categories of youtube channels")
    plt.xlabel("The number of views the categories have generated in last 30 days")
    # plt.xticks(rotation=45)
    plt.title("The distribution of the video views generated by categories of youtube channels in last 30 days")
    plt.show()

# def seasonal_trends_in_no_of_videos_uploaded():
#     df_temp = df[['uploads','category']].dropna()
#     dictionary = {}
#     for key,values in zip(df_temp['uploads'], df_temp['category']):
#         dictionary[values] = key
#     dictionary = dict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))
#     sns.barplot(x=[i for i in dictionary.keys()], y=[j for j in dictionary.values()], palette='coolwarm')
#     plt.xlabel("The categories of videos uploaded")
#     plt.ylabel("The number of videos uploaded")
#     plt.title("The seasonal trends in number of video uploads")
#     plt.xticks(rotation=70)
#     plt.show()

def seasonal_trends_in_no_of_videos_uploaded():
    df_temp=df[['created_year','created_month','created_date','uploads',"Youtuber"]].dropna()
    df_temp[['created_year','created_date']]=df_temp[['created_year','created_date']].astype(int)
    # df_temp['created']
    # print(df_temp)
    df_temp['current date']=pd.to_datetime('2025-03-31')
    # print(df_temp)
    df_temp['created_month']=df_temp['created_month'].apply(lambda x: datetime.strptime(x,"%b").month)
    # print(df_temp)
    df_temp=df_temp.rename(columns={"created_year":"year","created_month":"month","created_date":"day"})
    df_temp['channel_bday']=pd.to_datetime(df_temp[['year','month','day']])
    df_temp['gap']=df_temp.apply(lambda row: abs(relativedelta(row['channel_bday'],row['current date']).years*12+relativedelta(row['channel_bday'],row['current date']).months + relativedelta(row['channel_bday'],row['current date']).days/30),axis=1)
    # print(df_temp)
    dictionary={}
    for i,j,k in zip(df_temp['uploads'],df_temp['gap'],df_temp['Youtuber']):
        dictionary[k]=round(i/j)
    dictionary=dict(sorted(dictionary.items(),key=lambda x:x[1],reverse=True))
    # print(dictionary)
    sns.barplot(x=[i for i in dictionary.values()][:40],y=[i for i in dictionary.keys()][:40],palette='Set1')
    plt.show()
seasonal_trends_in_no_of_videos_uploaded()


def avg_subs_per_month_over_lifetime():
    df_temp = df[['Youtuber','subscribers','created_year','created_month','created_date']].dropna()
    df_temp['created_year'] = df_temp['created_year'].astype(int)
    df_temp['created_date'] = df_temp['created_date'].astype(int)
    df_temp["created_month"] = df_temp["created_month"].apply(lambda x: datetime.strptime(x,"%b").month)
    df_temp['subscribers'] = df_temp['subscribers'].astype(int)
    df_temp = df_temp.rename(columns={"created_year":"year","created_month":"month","created_date":"day"})
    df_temp['creation_date'] = pd.to_datetime(df_temp[['year','month','day']])
    current_date = pd.to_datetime("2025-3-31")
    df_temp['current_date'] = current_date
    df_temp['gap'] = df_temp.apply(lambda row: abs(relativedelta(row['creation_date'],row['current_date']).years * 12 + relativedelta(row['creation_date'],row['current_date']).months + relativedelta(row['creation_date'],row['current_date']).days/30), axis=1)
    df_temp['Average monthly subscribers lifetime'] = df_temp.apply(lambda row:row['subscribers']/row['gap'], axis=1)
    sns.barplot(x=df_temp['Average monthly subscribers lifetime'].head(30), y=df_temp['Youtuber'].head(30), palette='Set1')
    plt.xlabel("Youtube channel name")
    plt.ylabel("No of average subscribers per month gained in lifetime")
    # plt.xticks(rotation=90)
    plt.title("average number of subscribers gained per month since the creation of YouTube channels till now")
    plt.show()

def main():
    top_10_youtubers()
    top_avg_category()
    upload_count()
    top_country_youtubechannel()
    distribution_of_channels_across_categories()
    correlation()
    monthly_earnings_categories()
    trends_subs_gained_in_30_days()
    outliers_yearly_earning()
    creation_of_channel_trends()
    relation_btw_grosstertiaryedu_n_noofyoutubers()
    Unemployment_vs_NoOfYoutubeChannels()
    avg_urban_population_in_counties_with_most_channels()
    latitude_longitude_number_of_channels_pattern()
    corelation_btw_subscribers_and_population()
    No_of_channels_vs_population()
    no_of_subs_vs_unemployment_rate()
    video_views_in_last_30_days_in_categories()
    seasonal_trends_in_no_of_videos_uploaded()
    avg_subs_per_month_over_lifetime()

if __name__ == "__main__":
    main()